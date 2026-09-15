from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict
import multiprocessing as mp
import gc
import os
import re
import hashlib
import time

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd


# =========================
# Configuration
# =========================

BASE_DIR   = Path(__file__).resolve().parents[1]
DATA_ROOT  = Path(os.getenv("GDELT_DATA_ROOT", str(BASE_DIR / "data/processed")))
RAW_DATA_ROOT = Path(os.getenv("GDELT_RAW_DATA_ROOT", str(BASE_DIR / "data/raw")))
INPUT_FILE = RAW_DATA_ROOT / "datasets" / "02_articles_raw.csv"

SUMMARIES_DIR = DATA_ROOT / "summaries"
SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

OUT_ARTICLES  = DATA_ROOT / "datasets" / "02b_articles_deduped.csv"
OUT_QUALITY_ARTICLES = DATA_ROOT / "datasets" / "02c_articles_quality_filtered.csv"
OUT_DUP_PAIRS = SUMMARIES_DIR / "dedup_pairs.csv"      # diagnostic only
OUT_SUMMARY   = SUMMARIES_DIR / "dedup_quality_summary.txt"

# Dedup settings
SHINGLE_SIZE = 5
NEAR_DUP_JACCARD_THRESHOLD = 0.85
MIN_WORDS_FOR_NEAR_DUP = 80  # skip near duplicate check for very short texts


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


DEDUP_WORKERS = max(1, _env_int("GDELT_DEDUP_WORKERS", min(os.cpu_count() or 1, 8)))
DEDUP_WORKER_CAP = max(1, _env_int("GDELT_DEDUP_MAX_WORKERS", min(os.cpu_count() or 1, 8)))
REQUESTED_DEDUP_WORKERS = DEDUP_WORKERS
DEDUP_WORKERS = min(DEDUP_WORKERS, DEDUP_WORKER_CAP)
DEDUP_FEATURE_CHUNKSIZE = max(1, _env_int("GDELT_DEDUP_FEATURE_CHUNKSIZE", 100))
DEDUP_PAIR_CHUNKSIZE = max(1, _env_int("GDELT_DEDUP_PAIR_CHUNKSIZE", 25))
DEDUP_ALL_PAIRS_LIMIT = max(0, _env_int("GDELT_DEDUP_ALL_PAIRS_LIMIT", 5_000_000))
DEDUP_SIGNATURE_SIZE = max(8, _env_int("GDELT_DEDUP_SIGNATURE_SIZE", 64))
DEDUP_LSH_BAND_SIZE = max(2, _env_int("GDELT_DEDUP_LSH_BAND_SIZE", 8))
DEDUP_RARE_ANCHORS = max(0, _env_int("GDELT_DEDUP_RARE_ANCHORS", 8))
DEDUP_MAX_BUCKET_SIZE = max(2, _env_int("GDELT_DEDUP_MAX_BUCKET_SIZE", 250))
DEDUP_MAX_LSH_CANDIDATE_PAIRS = max(1, _env_int("GDELT_DEDUP_MAX_LSH_CANDIDATE_PAIRS", 2_000_000))

_NEAR_DUP_STATS = {}
_MULTIPROCESSING_FALLBACK_USED = False


def _process_pool_kwargs() -> dict:
    method = os.getenv("GDELT_PROCESS_START_METHOD")
    if method is None:
        method = "fork" if os.name == "posix" else ""
    return {"mp_context": mp.get_context(method)} if method else {}

# Hard coded deterministic quality rules
MIN_TEXT_LEN = 500
MIN_SENTENCES = 3
MAX_BOILERPLATE_HITS = 2       # 3 or more will be flagged bad
MAX_BOILERPLATE_RATIO = 0.03   # share of words that match boilerplate terms

BOILERPLATE_TERMS = [
    "cookie",
    "privacy",
    "subscribe",
    "newsletter",
    "javascript",
    "advertisement",
    "advertising",
    "ad choices",
    "sign up",
    "login",
    "log in",
    "terms of service",
    "accept cookies",
    "enable javascript",
    "all rights reserved",
    "read more",
    "breaking news alerts",
    "copyright",
]

REBEL_KEYWORDS = [
    "rebel",
    "rebels",
    "insurgent",
    "insurgents",
    "militia",
    "militias",
    "armed group",
    "armed groups",
    "clash",
    "clashes",
    "attack",
    "attacks",
    "fighting",
    "conflict",
    "ambush",
    "raid",
    "gunmen",
]

ACTOR_STOPWORDS = {
    "government",
    "police",
    "army",
    "military",
    "state",
    "unknown",
    "none",
    "nan",
    "",
}

MOJIBAKE_MARKERS = set("ÃÂâ€š„™œØÙÐÑ")
SEVERE_MOJIBAKE_MARKERS = set("ØÙÐÑ")
MOJIBAKE_SEQUENCE_PATTERN = re.compile(r"Ã.|Â.|â[\u0080-\u009f€]|[ØÙÐÑ]")


# =========================
# Helpers
# =========================

def clean_text(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def has_mojibake_artifacts(text):
    t = clean_text(text)
    if not t:
        return False
    marker_count = sum(ch in MOJIBAKE_MARKERS for ch in t)
    severe_count = sum(ch in SEVERE_MOJIBAKE_MARKERS for ch in t)
    sequence_count = len(MOJIBAKE_SEQUENCE_PATTERN.findall(t))
    text_len = max(len(t), 1)
    return (
        (sequence_count >= 3 and marker_count >= 3)
        or
        (marker_count >= 20 and marker_count / text_len >= 0.02)
        or (severe_count >= 8 and severe_count / text_len >= 0.01)
    )


def normalize_for_hash(text):
    t = clean_text(text).lower()
    t = re.sub(r"_", " ", t)
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_for_similarity(text):
    # same normalization for shingling
    return normalize_for_hash(text)


def tokenize(text):
    t = normalize_for_similarity(text)
    if not t:
        return []
    return t.split()


def stable_shingle_hash(text):
    return int.from_bytes(
        hashlib.blake2b(text.encode("utf8"), digest_size=8).digest(),
        "big",
    )


def shingle_set(tokens, k=5):
    if len(tokens) < k:
        return set()
    return {
        stable_shingle_hash(" ".join(tokens[i:i+k]))
        for i in range(len(tokens) - k + 1)
    }


def jaccard_similarity(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return inter / union if union else 0.0


def count_sentences(text):
    t = clean_text(text)
    if not t:
        return 0
    parts = re.split(r"[.!?]+", t)
    parts = [p.strip() for p in parts if p.strip()]
    return len(parts)


def count_words(text):
    return len(tokenize(text))


def count_boilerplate_hits(text):
    t = clean_text(text).lower()
    return sum(term in t for term in BOILERPLATE_TERMS)


def boilerplate_ratio(text):
    words = count_words(text)
    if words == 0:
        return 0.0
    # count raw occurrences for a slightly stronger signal
    t = clean_text(text).lower()
    hit_count = 0
    for term in BOILERPLATE_TERMS:
        hit_count += t.count(term)
    return hit_count / max(words, 1)


def contains_rebel_keyword(text):
    t = clean_text(text).lower()
    return any(term in t for term in REBEL_KEYWORDS)


def normalize_name(name):
    n = clean_text(name).lower()
    n = re.sub(r"\s+", " ", n).strip()
    return n


def is_useful_actor_name(name):
    n = normalize_name(name)
    if n in ACTOR_STOPWORDS:
        return False
    if len(n) < 4:
        return False
    if " " not in n and len(n) < 6:
        return False
    return True


def actor_name_in_text(name, text):
    n = normalize_name(name)
    if not is_useful_actor_name(n):
        return False
    t = clean_text(text).lower()
    return n in t


def get_domain(url):
    try:
        return urlparse(str(url)).netloc.lower()
    except Exception:
        return ""


def stable_text_hash(text, max_chars=4000):
    norm = normalize_for_hash(text)[:max_chars]
    return hashlib.sha1(norm.encode("utf8")).hexdigest() if norm else ""


def extract_article_features(args):
    text, actor1, actor2 = args
    cleaned = clean_text(text)
    normalized = normalize_for_similarity(cleaned)
    tokens = normalized.split() if normalized else []
    word_count = len(tokens)
    text_len = len(cleaned)

    sentence_parts = re.split(r"[.!?]+", cleaned) if cleaned else []
    sentence_count = len([p.strip() for p in sentence_parts if p.strip()])

    lower = cleaned.lower()
    boilerplate_hits = sum(term in lower for term in BOILERPLATE_TERMS)
    boilerplate_occurrences = 0
    for term in BOILERPLATE_TERMS:
        boilerplate_occurrences += lower.count(term)
    bp_ratio = boilerplate_occurrences / max(word_count, 1) if word_count else 0.0

    has_rebel_keyword = any(term in lower for term in REBEL_KEYWORDS)
    has_actor1_signal = actor_name_in_text(actor1, cleaned)
    has_actor2_signal = actor_name_in_text(actor2, cleaned)
    text_hash = hashlib.sha1(normalized[:4000].encode("utf8")).hexdigest() if normalized else ""
    shingles = shingle_set(tokens, SHINGLE_SIZE) if word_count >= MIN_WORDS_FOR_NEAR_DUP else set()

    return (
        cleaned,
        text_len,
        word_count,
        sentence_count,
        boilerplate_hits,
        bp_ratio,
        has_rebel_keyword,
        has_actor1_signal,
        has_actor2_signal,
        text_hash,
        shingles,
    )


def extract_features_parallel(feature_inputs):
    global _MULTIPROCESSING_FALLBACK_USED

    if DEDUP_WORKERS <= 1 or len(feature_inputs) <= 1:
        return [extract_article_features(value) for value in feature_inputs]

    try:
        with ProcessPoolExecutor(max_workers=DEDUP_WORKERS, **_process_pool_kwargs()) as executor:
            return list(
                executor.map(
                    extract_article_features,
                    feature_inputs,
                    chunksize=DEDUP_FEATURE_CHUNKSIZE,
                )
            )
    except PermissionError as exc:
        _MULTIPROCESSING_FALLBACK_USED = True
        print(f"[DEDUP] Multiprocessing unavailable ({exc}); falling back to 1 worker", flush=True)
        return [extract_article_features(value) for value in feature_inputs]


_PAIR_CANDIDATES = []
_PAIR_WORD_COUNTS = []
_PAIR_TEXT_HASHES = []
_PAIR_SHINGLES = []


def init_near_pair_worker(candidates, word_counts, text_hashes, shingles):
    global _PAIR_CANDIDATES, _PAIR_WORD_COUNTS, _PAIR_TEXT_HASHES, _PAIR_SHINGLES
    _PAIR_CANDIDATES = candidates
    _PAIR_WORD_COUNTS = word_counts
    _PAIR_TEXT_HASHES = text_hashes
    _PAIR_SHINGLES = shingles


def find_near_pairs_for_range(bounds):
    start_pos, end_pos = bounds
    pairs = []
    total = len(_PAIR_CANDIDATES)

    for pos_i in range(start_pos, end_pos):
        i = _PAIR_CANDIDATES[pos_i]
        wi = _PAIR_WORD_COUNTS[i]
        hash_i = _PAIR_TEXT_HASHES[i]
        shingles_i = _PAIR_SHINGLES[i]

        for pos_j in range(pos_i + 1, total):
            j = _PAIR_CANDIDATES[pos_j]
            hash_j = _PAIR_TEXT_HASHES[j]
            if hash_i == hash_j and hash_i != "":
                continue

            wj = _PAIR_WORD_COUNTS[j]
            if min(wi, wj) == 0:
                continue
            len_ratio = max(wi, wj) / min(wi, wj)
            if len_ratio > 1.6:
                continue

            sim = jaccard_similarity(shingles_i, _PAIR_SHINGLES[j])
            if sim >= NEAR_DUP_JACCARD_THRESHOLD:
                pairs.append((i, j, round(sim, 4)))

    return pairs


def _lsh_band_keys(signature):
    usable_len = min(len(signature), DEDUP_SIGNATURE_SIZE)
    for start in range(0, usable_len, DEDUP_LSH_BAND_SIZE):
        band = signature[start:start + DEDUP_LSH_BAND_SIZE]
        if len(band) == DEDUP_LSH_BAND_SIZE:
            yield ("sig", start // DEDUP_LSH_BAND_SIZE, band)


def _can_compare_pair(i, j, word_counts, text_hashes):
    hash_i = text_hashes[i]
    hash_j = text_hashes[j]
    if hash_i == hash_j and hash_i != "":
        return False

    wi = word_counts[i]
    wj = word_counts[j]
    if min(wi, wj) == 0:
        return False

    return max(wi, wj) / min(wi, wj) <= 1.6


def find_near_duplicate_pairs_lsh(candidates, word_counts, text_hashes, shingles):
    global _NEAR_DUP_STATS

    print("[DEDUP] Candidate pairs exceed all-pairs limit; using bounded LSH blocking", flush=True)
    print(f"[DEDUP] All-pairs limit: {DEDUP_ALL_PAIRS_LIMIT}", flush=True)
    print(
        "[DEDUP] LSH settings: "
        f"signature={DEDUP_SIGNATURE_SIZE}, band={DEDUP_LSH_BAND_SIZE}, "
        f"rare_anchors={DEDUP_RARE_ANCHORS}, max_bucket={DEDUP_MAX_BUCKET_SIZE}, "
        f"max_pairs={DEDUP_MAX_LSH_CANDIDATE_PAIRS}",
        flush=True,
    )

    shingle_freq = Counter()
    if DEDUP_RARE_ANCHORS:
        for row_idx in candidates:
            shingle_freq.update(shingles[row_idx])

    buckets = defaultdict(list)
    for row_idx in candidates:
        row_shingles = shingles[row_idx]
        if not row_shingles:
            continue

        signature = tuple(sorted(row_shingles)[:DEDUP_SIGNATURE_SIZE])
        for key in _lsh_band_keys(signature):
            buckets[key].append(row_idx)

        if DEDUP_RARE_ANCHORS:
            rare_anchors = sorted(
                row_shingles,
                key=lambda value: (shingle_freq[value], value),
            )[:DEDUP_RARE_ANCHORS]
            for anchor in rare_anchors:
                buckets[("rare", anchor)].append(row_idx)

    candidate_pairs = set()
    skipped_large_buckets = 0
    limit_reached = False

    for rows in buckets.values():
        if len(rows) < 2:
            continue
        if len(rows) > DEDUP_MAX_BUCKET_SIZE:
            skipped_large_buckets += 1
            continue

        rows = sorted(rows)
        for pos_i, i in enumerate(rows[:-1]):
            for j in rows[pos_i + 1:]:
                if not _can_compare_pair(i, j, word_counts, text_hashes):
                    continue
                candidate_pairs.add((i, j))
                if len(candidate_pairs) >= DEDUP_MAX_LSH_CANDIDATE_PAIRS:
                    limit_reached = True
                    break
            if limit_reached:
                break
        if limit_reached:
            break

    print(f"[DEDUP] LSH buckets built: {len(buckets)}", flush=True)
    print(f"[DEDUP] LSH buckets skipped as too large: {skipped_large_buckets}", flush=True)
    print(f"[DEDUP] LSH candidate pairs after filters: {len(candidate_pairs)}", flush=True)
    if limit_reached:
        print("[DEDUP] LSH candidate-pair cap reached; results may miss some near duplicates", flush=True)

    pair_tuples = []
    ordered_pairs = sorted(candidate_pairs)
    for completed, (i, j) in enumerate(ordered_pairs, start=1):
        sim = jaccard_similarity(shingles[i], shingles[j])
        if sim >= NEAR_DUP_JACCARD_THRESHOLD:
            pair_tuples.append((i, j, round(sim, 4)))

        if completed % 100000 == 0 or completed == len(ordered_pairs):
            print(f"[DEDUP] LSH exact checks complete: {completed}/{len(ordered_pairs)}", flush=True)

    pair_tuples.sort(key=lambda item: (item[0], item[1]))
    _NEAR_DUP_STATS = {
        "strategy": "lsh",
        "all_pairs_limit": DEDUP_ALL_PAIRS_LIMIT,
        "lsh_buckets": len(buckets),
        "lsh_skipped_large_buckets": skipped_large_buckets,
        "lsh_candidate_pairs": len(candidate_pairs),
        "lsh_pair_cap_reached": limit_reached,
        "near_pairs_found": len(pair_tuples),
    }

    return [
        {
            "row_i": i,
            "row_j": j,
            "dup_kind": "near",
            "similarity": sim,
            "text_hash": "",
        }
        for i, j, sim in pair_tuples
    ]


def find_near_duplicate_pairs_parallel(candidates, word_counts, text_hashes, shingles):
    global _NEAR_DUP_STATS, _MULTIPROCESSING_FALLBACK_USED

    if len(candidates) < 2:
        _NEAR_DUP_STATS = {
            "strategy": "none",
            "all_pairs_limit": DEDUP_ALL_PAIRS_LIMIT,
            "near_pairs_found": 0,
        }
        return []

    n_candidate_pairs = len(candidates) * (len(candidates) - 1) // 2
    if n_candidate_pairs > DEDUP_ALL_PAIRS_LIMIT:
        return find_near_duplicate_pairs_lsh(candidates, word_counts, text_hashes, shingles)

    _NEAR_DUP_STATS = {
        "strategy": "all_pairs",
        "all_pairs_limit": DEDUP_ALL_PAIRS_LIMIT,
        "near_pairs_found": 0,
    }

    pair_end = len(candidates) - 1
    ranges = [
        (start, min(start + DEDUP_PAIR_CHUNKSIZE, pair_end))
        for start in range(0, pair_end, DEDUP_PAIR_CHUNKSIZE)
    ]

    if DEDUP_WORKERS <= 1 or len(ranges) <= 1:
        init_near_pair_worker(candidates, word_counts, text_hashes, shingles)
        pair_tuples = []
        for bounds in ranges:
            pair_tuples.extend(find_near_pairs_for_range(bounds))
    else:
        pair_tuples = []
        try:
            with ProcessPoolExecutor(
                max_workers=DEDUP_WORKERS,
                initializer=init_near_pair_worker,
                initargs=(candidates, word_counts, text_hashes, shingles),
                **_process_pool_kwargs(),
            ) as executor:
                future_to_bounds = {
                    executor.submit(find_near_pairs_for_range, bounds): bounds
                    for bounds in ranges
                }
                completed = 0
                for future in as_completed(future_to_bounds):
                    pair_tuples.extend(future.result())
                    completed += 1
                    if completed % 25 == 0 or completed == len(ranges):
                        print(f"[DEDUP] Near-pair chunks complete: {completed}/{len(ranges)}", flush=True)
        except PermissionError as exc:
            _MULTIPROCESSING_FALLBACK_USED = True
            print(f"[DEDUP] Multiprocessing unavailable ({exc}); falling back to 1 worker", flush=True)
            init_near_pair_worker(candidates, word_counts, text_hashes, shingles)
            for bounds in ranges:
                pair_tuples.extend(find_near_pairs_for_range(bounds))

    pair_tuples.sort(key=lambda item: (item[0], item[1]))
    _NEAR_DUP_STATS["near_pairs_found"] = len(pair_tuples)
    return [
        {
            "row_i": i,
            "row_j": j,
            "dup_kind": "near",
            "similarity": sim,
            "text_hash": "",
        }
        for i, j, sim in pair_tuples
    ]


def find_column(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing expected column. Tried {candidates}")
    return None


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# =========================
# Main
# =========================

def main():
    start_time = time.time()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    # Prefer text_cleaned (boilerplate-filtered) produced by 05_clean_articles.
    # Falls back to raw text if the cleaning step was skipped.
    text_col = find_column(df, ["text_cleaned", "article_text", "text", "extracted_text", "content"])
    url_col = find_column(df, ["SOURCEURL", "scraped_url", "url"], required=False)
    title_col = find_column(df, ["title"], required=False)
    actor1_col = find_column(df, ["Actor1Name"], required=False)
    actor2_col = find_column(df, ["Actor2Name"], required=False)
    event_id_col = find_column(df, ["GlobalEventID"], required=False)

    print(f"[DEDUP] Workers: {DEDUP_WORKERS}")
    if REQUESTED_DEDUP_WORKERS != DEDUP_WORKERS:
        print(f"[DEDUP] Requested workers capped: {REQUESTED_DEDUP_WORKERS} -> {DEDUP_WORKERS}")
    print(f"[DEDUP] Feature chunksize: {DEDUP_FEATURE_CHUNKSIZE}")
    print(f"[DEDUP] Pair chunksize: {DEDUP_PAIR_CHUNKSIZE}")
    print(f"[DEDUP] All-pairs limit: {DEDUP_ALL_PAIRS_LIMIT}")

    # Basic cleaning and feature prep
    actor1_values = df[actor1_col].tolist() if actor1_col else [""] * len(df)
    actor2_values = df[actor2_col].tolist() if actor2_col else [""] * len(df)
    feature_inputs = list(zip(df[text_col].tolist(), actor1_values, actor2_values))
    feature_rows = extract_features_parallel(feature_inputs)

    if feature_rows:
        (
            cleaned_texts,
            text_lengths,
            word_counts,
            sentence_counts,
            boilerplate_hits,
            boilerplate_ratios,
            rebel_keyword_flags,
            actor1_signal_flags,
            actor2_signal_flags,
            text_hashes,
            shingle_sets,
        ) = zip(*feature_rows)
    else:
        cleaned_texts = []
        text_lengths = []
        word_counts = []
        sentence_counts = []
        boilerplate_hits = []
        boilerplate_ratios = []
        rebel_keyword_flags = []
        actor1_signal_flags = []
        actor2_signal_flags = []
        text_hashes = []
        shingle_sets = []

    df[text_col] = list(cleaned_texts)
    df["text_len"] = list(text_lengths)
    df["word_count"] = list(word_counts)
    df["sentence_count"] = list(sentence_counts)
    df["boilerplate_hits"] = list(boilerplate_hits)
    df["boilerplate_ratio"] = list(boilerplate_ratios)
    df["has_rebel_keyword"] = list(rebel_keyword_flags)

    if url_col:
        df["domain"] = df[url_col].apply(get_domain)
    else:
        df["domain"] = ""

    df["has_actor1_signal"] = list(actor1_signal_flags)
    df["has_actor2_signal"] = list(actor2_signal_flags)

    df["has_actor_signal"] = df["has_actor1_signal"] | df["has_actor2_signal"]
    df["has_relevance_signal"] = df["has_actor_signal"] | df["has_rebel_keyword"]

    # =========================
    # Duplicate detection
    # =========================

    n = len(df)
    uf = UnionFind(n)
    duplicate_pairs = []

    # Exact duplicate pass using hash
    df["text_hash"] = list(text_hashes)
    hash_groups = df.groupby("text_hash").indices

    for text_hash, idxs in hash_groups.items():
        idx_list = list(idxs)
        if text_hash and len(idx_list) >= 2:
            base = idx_list[0]
            for j in idx_list[1:]:
                uf.union(base, j)
                duplicate_pairs.append({
                    "row_i": base,
                    "row_j": j,
                    "dup_kind": "exact",
                    "similarity": 1.0,
                    "text_hash": text_hash,
                })

    # Near duplicate pass using shingles and Jaccard
    # Only compare rows with enough words and not already exact same hash
    candidates = [i for i, wc in enumerate(word_counts) if wc >= MIN_WORDS_FOR_NEAR_DUP]
    n_candidate_pairs = len(candidates) * (len(candidates) - 1) // 2
    print(f"[DEDUP] Near-duplicate candidates: {len(candidates)}")
    print(f"[DEDUP] Candidate pairs before filters: {n_candidate_pairs}")

    near_pairs = find_near_duplicate_pairs_parallel(
        candidates,
        list(word_counts),
        list(text_hashes),
        list(shingle_sets),
    )

    for pair in near_pairs:
        uf.union(pair["row_i"], pair["row_j"])
        duplicate_pairs.append(pair)

    del near_pairs
    del shingle_sets
    del feature_rows
    gc.collect()

    # Build duplicate groups
    roots = [uf.find(i) for i in range(n)]
    df["dup_root"] = roots

    # Assign readable group ids
    root_to_gid = {}
    gid_counter = 1
    for r in sorted(set(roots)):
        root_to_gid[r] = f"DUP_{gid_counter:04d}"
        gid_counter += 1

    df["dup_group_id"] = df["dup_root"].map(root_to_gid)
    group_sizes = df.groupby("dup_group_id").size().to_dict()
    df["dup_group_size"] = df["dup_group_id"].map(group_sizes).astype(int)

    # Canonical row choice
    # Keep the best row by longest text first, then fewer boilerplate hits
    df["canonical_rank_key_1"] = -df["text_len"]  # longer is better
    df["canonical_rank_key_2"] = df["boilerplate_hits"]  # fewer is better

    canonical_indices = []
    for gid, part in df.groupby("dup_group_id", sort=False):
        part_sorted = part.sort_values(
            by=["canonical_rank_key_1", "canonical_rank_key_2"],
            ascending=[True, True]
        )
        canonical_indices.append(part_sorted.index[0])

    canonical_set = set(canonical_indices)
    df["is_canonical"] = df.index.isin(canonical_set)
    df["drop_as_duplicate"] = (~df["is_canonical"]) & (df["dup_group_size"] >= 2)

    # =========================
    # Deterministic quality score
    # =========================

    # Important point
    # Duplicate articles are not automatically bad
    # They can be valid syndicated copies
    # So quality_bad_score is independent of duplicate status
    df["flag_short_text"] = df["text_len"] < MIN_TEXT_LEN
    df["flag_few_sentences"] = df["sentence_count"] < MIN_SENTENCES
    df["flag_boilerplate_hits"] = df["boilerplate_hits"] > MAX_BOILERPLATE_HITS
    df["flag_boilerplate_ratio"] = df["boilerplate_ratio"] > MAX_BOILERPLATE_RATIO
    df["flag_mojibake_text"] = df[text_col].apply(has_mojibake_artifacts)

    # Optional weak relevance warning
    # Not part of quality_bad_score for now
    df["flag_no_relevance_signal"] = ~df["has_relevance_signal"]

    # 0 good, 1 bad
    df["quality_bad_score"] = (
        df["flag_short_text"] |
        df["flag_few_sentences"] |
        df["flag_boilerplate_hits"] |
        df["flag_boilerplate_ratio"] |
        df["flag_mojibake_text"]
    ).astype(int)

    quality_reasons = []
    for short_text, few_sentences, boilerplate_hits_flag, boilerplate_ratio_flag, mojibake_flag in zip(
        df["flag_short_text"].to_numpy(),
        df["flag_few_sentences"].to_numpy(),
        df["flag_boilerplate_hits"].to_numpy(),
        df["flag_boilerplate_ratio"].to_numpy(),
        df["flag_mojibake_text"].to_numpy(),
    ):
        reasons = []
        if short_text:
            reasons.append("short_text")
        if few_sentences:
            reasons.append("few_sentences")
        if boilerplate_hits_flag:
            reasons.append("boilerplate_hits")
        if boilerplate_ratio_flag:
            reasons.append("boilerplate_ratio")
        if mojibake_flag:
            reasons.append("mojibake_text")
        quality_reasons.append(",".join(reasons))
    df["quality_bad_reason"] = quality_reasons

    # =========================
    # Duplicate pairs export with metadata
    # =========================

    if duplicate_pairs:
        dup_pairs_df = pd.DataFrame(duplicate_pairs)

        # Attach URL and title columns for easy inspection
        def safe_take(row_idx, col):
            if col is None or col not in df.columns:
                return ""
            return df.iloc[row_idx][col]

        dup_pairs_df["url_i"] = dup_pairs_df["row_i"].apply(lambda x: safe_take(x, url_col))
        dup_pairs_df["url_j"] = dup_pairs_df["row_j"].apply(lambda x: safe_take(x, url_col))
        dup_pairs_df["title_i"] = dup_pairs_df["row_i"].apply(lambda x: safe_take(x, title_col))
        dup_pairs_df["title_j"] = dup_pairs_df["row_j"].apply(lambda x: safe_take(x, title_col))
        dup_pairs_df["group_i"] = dup_pairs_df["row_i"].apply(lambda x: df.iloc[x]["dup_group_id"])
        dup_pairs_df["group_j"] = dup_pairs_df["row_j"].apply(lambda x: df.iloc[x]["dup_group_id"])
    else:
        dup_pairs_df = pd.DataFrame(columns=[
            "row_i", "row_j", "dup_kind", "similarity", "text_hash",
            "url_i", "url_j", "title_i", "title_j", "group_i", "group_j"
        ])

    # =========================
    # Save outputs
    # =========================

    df_out = df.copy()

    # Drop internal helper columns before writing
    helper_cols = ["dup_root", "canonical_rank_key_1", "canonical_rank_key_2"]
    for c in helper_cols:
        if c in df_out.columns:
            df_out = df_out.drop(columns=[c])

    # Write only canonical rows to the deduped output file
    df_out = df_out[df_out["is_canonical"]].reset_index(drop=True)
    df_out.to_csv(OUT_ARTICLES, index=False, encoding="utf8")

    df_quality_out = df_out[df_out["quality_bad_score"] == 0].reset_index(drop=True)
    if text_col != "text" and "text" in df_quality_out.columns:
        df_quality_out = df_quality_out.drop(columns=["text"])
    df_quality_out.to_csv(OUT_QUALITY_ARTICLES, index=False, encoding="utf8")

    # Duplicate pairs go to summaries/ for inspection only
    dup_pairs_df.to_csv(OUT_DUP_PAIRS, index=False, encoding="utf8")

    # =========================
    # Summary
    # =========================

    input_rows = len(df)
    output_rows = len(df_out)
    duplicate_rows = int((df["dup_group_size"] >= 2).sum())
    dropped_rows = int(df["drop_as_duplicate"].sum())
    canonical_rows = int(df["is_canonical"].sum())
    bad_rows = int(df_out["quality_bad_score"].sum())
    good_rows = output_rows - bad_rows
    elapsed = time.time() - start_time

    lines = []
    lines.append("ARTICLE DEDUP AND QUALITY SUMMARY")
    lines.append("")
    lines.append(f"Input rows: {input_rows}")
    lines.append(f"Rows in duplicate groups: {duplicate_rows}")
    lines.append(f"Rows marked drop_as_duplicate: {dropped_rows}")
    lines.append(f"Canonical rows after dedup: {canonical_rows}")
    lines.append(f"Rows written to deduped output: {output_rows}")
    lines.append(f"Rows written to quality-filtered output: {len(df_quality_out)}")
    lines.append(f"Dedup workers: {DEDUP_WORKERS}")
    lines.append(f"Multiprocessing fallback used: {_MULTIPROCESSING_FALLBACK_USED}")
    if REQUESTED_DEDUP_WORKERS != DEDUP_WORKERS:
        lines.append(f"Dedup requested workers: {REQUESTED_DEDUP_WORKERS}")
        lines.append(f"Dedup worker cap: {DEDUP_WORKER_CAP}")
    lines.append(f"Dedup feature chunksize: {DEDUP_FEATURE_CHUNKSIZE}")
    lines.append(f"Dedup pair chunksize: {DEDUP_PAIR_CHUNKSIZE}")
    lines.append(f"Near duplicate strategy: {_NEAR_DUP_STATS.get('strategy', 'unknown')}")
    lines.append(f"Near duplicate pairs found: {_NEAR_DUP_STATS.get('near_pairs_found', 0)}")
    if _NEAR_DUP_STATS.get("strategy") == "lsh":
        lines.append(f"LSH buckets: {_NEAR_DUP_STATS.get('lsh_buckets', 0)}")
        lines.append(f"LSH candidate pairs checked: {_NEAR_DUP_STATS.get('lsh_candidate_pairs', 0)}")
        lines.append(f"LSH large buckets skipped: {_NEAR_DUP_STATS.get('lsh_skipped_large_buckets', 0)}")
        lines.append(f"LSH pair cap reached: {_NEAR_DUP_STATS.get('lsh_pair_cap_reached', False)}")
    lines.append(f"Elapsed seconds: {elapsed:.2f}")
    lines.append("")
    lines.append("QUALITY BAD SCORE")
    lines.append("0 means good")
    lines.append("1 means bad")
    lines.append(f"Good rows: {good_rows}")
    lines.append(f"Bad rows: {bad_rows}")
    lines.append("")
    lines.append("QUALITY FLAG COUNTS")
    lines.append(f"short_text: {int(df_out['flag_short_text'].sum())}")
    lines.append(f"few_sentences: {int(df_out['flag_few_sentences'].sum())}")
    lines.append(f"boilerplate_hits: {int(df_out['flag_boilerplate_hits'].sum())}")
    lines.append(f"boilerplate_ratio: {int(df_out['flag_boilerplate_ratio'].sum())}")
    lines.append(f"mojibake_text: {int(df_out['flag_mojibake_text'].sum())}")
    lines.append(f"no_relevance_signal (not used in score): {int(df_out['flag_no_relevance_signal'].sum())}")
    lines.append("")
    lines.append("DUPLICATE TYPES")
    if not dup_pairs_df.empty:
        lines.append(dup_pairs_df["dup_kind"].value_counts().to_string())
    else:
        lines.append("No duplicate pairs found")
    lines.append("")
    lines.append(f"Saved deduped articles to  {OUT_ARTICLES}")
    lines.append(f"Saved quality-filtered articles to {OUT_QUALITY_ARTICLES}")
    lines.append(f"Saved duplicate pairs to   {OUT_DUP_PAIRS}")
    lines.append(f"Saved summary to           {OUT_SUMMARY}")

    summary_text = "\n".join(lines)
    print(summary_text)

    with open(OUT_SUMMARY, "w", encoding="utf8") as f:
        f.write(summary_text)


if __name__ == "__main__":
    main()
