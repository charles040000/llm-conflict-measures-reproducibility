from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import time
import threading
from typing import Dict, Tuple
from urllib.parse import urlparse

# NumPy wheels on macOS can hang during Accelerate initialization with the
# default thread settings. Set this before importing pandas/numpy.
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

print("[BOOT] Starting article scraper module", flush=True)
import pandas as pd
import requests
print("[BOOT] Article scraper imports complete", flush=True)


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.getenv("GDELT_DATA_ROOT", str(BASE_DIR / "data" / "raw")))

EVENTS_PATH    = DATA_ROOT / "gdelt" / "events_base.csv"
CANDIDATES_PATH = DATA_ROOT / "gdelt" / "event_url_candidates.csv"

DATASETS_DIR  = DATA_ROOT / "datasets"
SUMMARIES_DIR = DATA_ROOT / "summaries"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

ARTICLES_PATH = DATASETS_DIR / "02_articles_raw.csv"
FAILURES_PATH = SUMMARIES_DIR / "scraping_failures.csv"
PROGRESS_PATH = SUMMARIES_DIR / "scraping_progress.csv"
SUMMARY_PATH = SUMMARIES_DIR / "scraping_summary.txt"
COMPLETE_FLAG_PATH = SUMMARIES_DIR / "scraping_complete.flag"

ARTICLE_COLUMNS = [
    "GlobalEventID", "SQLDATE", "DATEADDED", "Actor1Name", "Actor2Name",
    "Actor1CountryCode", "Actor2CountryCode", "ActionGeo_CountryCode",
    "EventCode", "EventBaseCode", "EventRootCode", "QuadClass",
    "NumArticles", "NumMentions", "NumSources", "AvgTone", "GoldsteinScale",
    "match_rad_term", "match_ucdp_actor_term", "match_gdelt_reb_type",
    "match_sources", "match_rebel_term", "match_rebel_type", "involves_rebel",
    "used_url", "used_candidate_rank", "used_primary_sourceurl",
    "n_url_attempts_for_event", "final_url", "http_status", "content_type",
    "text_len", "word_count", "text",
]
FAILURE_COLUMNS = [
    "failure_scope", "GlobalEventID", "SQLDATE", "candidate_rank",
    "candidate_url", "is_primary_sourceurl", "ok", "reason", "http_status",
    "content_type", "final_url", "text_len", "error",
    "n_url_attempts_for_event",
]
PROGRESS_COLUMNS = [
    "GlobalEventID", "SQLDATE", "ok", "reason", "n_candidate_urls",
    "n_url_attempts_for_event", "used_candidate_rank", "used_url",
    "final_url", "text_len", "finished_at",
]

def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


SCRAPE_N_EVENTS = _env_int("GDELT_SCRAPE_N_EVENTS", 100)
SCRAPE_SEED = _env_int("GDELT_SCRAPE_SEED", 42)
SCRAPE_SLEEP_SECONDS = float(os.getenv("GDELT_SCRAPE_SLEEP_SECONDS", "0.8"))
SCRAPE_REQUEST_TIMEOUT_SECONDS = _env_int("GDELT_SCRAPE_REQUEST_TIMEOUT", 10)
SCRAPE_WORKERS = max(1, _env_int("GDELT_SCRAPE_WORKERS", 8))
SCRAPE_CHECKPOINT_EVERY = max(1, _env_int("GDELT_SCRAPE_CHECKPOINT_EVERY", 100))
SCRAPE_RESUME = _env_bool("GDELT_SCRAPE_RESUME", True)
SCRAPE_REBEL_ONLY = _env_bool("GDELT_SCRAPE_REBEL_ONLY", True)
SCRAPE_MATCH_FILTER = os.getenv("GDELT_SCRAPE_MATCH_FILTER", "any_rebel").strip().lower()
SCRAPE_EVENT_ROOT_FILTER = {
    x.strip()
    for x in os.getenv("GDELT_SCRAPE_EVENT_ROOT_FILTER", "").split(",")
    if x.strip()
}

MAX_CANDIDATES_TO_TRY_PER_EVENT = _env_int("GDELT_MAX_CANDIDATES_TO_TRY_PER_EVENT", 5)

MIN_TEXT_CHARS = 200
MAX_TEXT_CHARS = 8000

_THREAD_LOCAL = threading.local()


def normalize_url(url: str) -> str:
    return str(url or "").strip()


def normalize_text(s: str) -> str:
    s = str(s or "")
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def is_probably_pdf(url: str) -> bool:
    u = normalize_url(url).lower()
    return u.endswith(".pdf") or ".pdf?" in u


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en,en-US;q=0.9",
    })
    return s


def get_thread_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = make_session()
        _THREAD_LOCAL.session = session
    return session


def fetch_html(session: requests.Session, url: str, timeout: int = SCRAPE_REQUEST_TIMEOUT_SECONDS) -> Tuple[int, str, str, str]:
    r = session.get(url, timeout=timeout, allow_redirects=True)
    status = int(r.status_code)
    final_url = str(r.url)
    content_type = (r.headers.get("Content-Type") or "").lower()

    html = ""
    if "text/html" in content_type or "application/xhtml" in content_type or content_type == "":
        r.encoding = r.encoding or "utf-8"
        html = r.text

    return status, final_url, content_type, html


def try_trafilatura(html: str) -> str:
    try:
        import trafilatura
        txt = trafilatura.extract(html, include_comments=False, include_tables=False)
        return (txt or "").strip()
    except Exception:
        return ""


def try_bs4(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        txt = soup.get_text(separator=" ", strip=True)
        return (txt or "").strip()
    except Exception:
        return ""


def extract_article_text(session: requests.Session, url: str) -> Dict:
    url_n = normalize_url(url)

    if not url_n:
        return {"ok": False, "reason": "empty_url", "final_url": ""}

    if is_probably_pdf(url_n):
        return {"ok": False, "reason": "pdf_url_skipped", "final_url": url_n}

    try:
        status, final_url, content_type, html = fetch_html(session, url_n)
    except Exception as e:
        return {"ok": False, "reason": "request_error", "final_url": url_n, "error": str(e)}

    if status >= 400:
        return {
            "ok": False,
            "reason": "http_error",
            "final_url": final_url,
            "http_status": status,
            "content_type": content_type,
        }

    if not html:
        return {
            "ok": False,
            "reason": "no_html",
            "final_url": final_url,
            "http_status": status,
            "content_type": content_type,
        }

    text = try_trafilatura(html)
    if len(text) < MIN_TEXT_CHARS:
        text = try_bs4(html)

    text = normalize_text(text)

    if len(text) < MIN_TEXT_CHARS:
        return {
            "ok": False,
            "reason": "extraction_too_short",
            "final_url": final_url,
            "http_status": status,
            "content_type": content_type,
            "text_len": len(text),
        }

    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]

    return {
        "ok": True,
        "reason": "",
        "final_url": final_url,
        "http_status": status,
        "content_type": content_type,
        "text": text,
        "text_len": len(text),
        "word_count": len(text.split()),
    }


def summarize_failures(fail_df: pd.DataFrame) -> pd.DataFrame:
    if fail_df.empty:
        return pd.DataFrame(columns=["reason", "n"])
    return (
        fail_df["reason"]
        .fillna("")
        .value_counts()
        .rename_axis("reason")
        .reset_index(name="n")
    )


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    extra_cols = [col for col in out.columns if col not in columns]
    return out[columns + extra_cols].copy()


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"[RESUME] Could not read {path}: {exc}")
        return pd.DataFrame()


def global_event_ids(df: pd.DataFrame) -> set[int]:
    if df.empty or "GlobalEventID" not in df.columns:
        return set()
    ids = pd.to_numeric(df["GlobalEventID"], errors="coerce").dropna().astype(int)
    return set(ids.tolist())


def build_candidates_lookup(candidates_df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    if candidates_df.empty or "GlobalEventID" not in candidates_df.columns:
        return {}
    out = {}
    for geid, group in candidates_df.groupby("GlobalEventID", sort=False):
        if pd.isna(geid):
            continue
        out[int(geid)] = group.copy()
    return out


def get_event_candidates_for_scrape(event_row: pd.Series, candidates_df) -> pd.DataFrame:
    geid = pd.to_numeric(event_row.get("GlobalEventID", None), errors="coerce")
    if pd.isna(geid):
        return pd.DataFrame(columns=["candidate_rank", "candidate_url", "is_primary_sourceurl", "score"])

    geid = int(geid)

    if isinstance(candidates_df, dict):
        c0 = candidates_df.get(geid)
        c = c0.copy() if c0 is not None else pd.DataFrame()
    else:
        c = candidates_df[candidates_df["GlobalEventID"] == geid].copy() if len(candidates_df) else pd.DataFrame()

    if len(c) > 0:
        c["candidate_url"] = c["candidate_url"].astype(str).map(normalize_url)
        c = c[c["candidate_url"] != ""].copy()

        if "is_primary_sourceurl" not in c.columns:
            c["is_primary_sourceurl"] = False
        c["is_primary_sourceurl"] = c["is_primary_sourceurl"].fillna(False).astype(bool)

        if "candidate_rank" not in c.columns:
            c["candidate_rank"] = 9999
        c["candidate_rank"] = pd.to_numeric(c["candidate_rank"], errors="coerce").fillna(9999).astype(int)

        if "score" not in c.columns:
            c["score"] = 0.0
        c["score"] = pd.to_numeric(c["score"], errors="coerce").fillna(0.0)

        c = c.sort_values(["candidate_rank", "score"], ascending=[True, False]).copy()
    else:
        c = pd.DataFrame(columns=["candidate_rank", "candidate_url", "is_primary_sourceurl", "score"])

    primary_url = normalize_url(event_row.get("SOURCEURL", ""))
    if primary_url:
        if len(c) == 0 or not (c["candidate_url"] == primary_url).any():
            primary_row = pd.DataFrame([{
                "GlobalEventID": geid,
                "candidate_rank": 0,
                "candidate_url": primary_url,
                "is_primary_sourceurl": True,
                "score": 9999.0,
            }])
            c = pd.concat([primary_row, c], ignore_index=True)
        else:
            c.loc[c["candidate_url"] == primary_url, "is_primary_sourceurl"] = True

    c = c.drop_duplicates(subset=["candidate_url"]).copy()
    c = c.sort_values(["is_primary_sourceurl", "candidate_rank", "score"], ascending=[False, True, False]).copy()
    c["candidate_rank"] = range(1, len(c) + 1)

    c = c.head(MAX_CANDIDATES_TO_TRY_PER_EVENT).copy()
    return c.reset_index(drop=True)


def as_bool(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def as_int(value, default: int) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def split_failure_rows(fail_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fail_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    if "failure_scope" in fail_df.columns:
        event_mask = fail_df["failure_scope"].fillna("").astype(str).eq("event")
    else:
        event_mask = fail_df["reason"].fillna("").isin(["no_candidate_urls", "all_candidates_failed"])
        if "candidate_url" in fail_df.columns:
            event_mask = event_mask & fail_df["candidate_url"].fillna("").astype(str).eq("")
    return fail_df[~event_mask].copy(), fail_df[event_mask].copy()


def build_scraping_summary(ok_df: pd.DataFrame, fail_df: pd.DataFrame, progress_df: pd.DataFrame, status: str) -> str:
    fail_attempt_df, fail_event_df = split_failure_rows(fail_df)

    if not progress_df.empty and "GlobalEventID" in progress_df.columns:
        progress_unique = progress_df.drop_duplicates(subset=["GlobalEventID"], keep="last")
        n_events_attempted = len(progress_unique)
        if "ok" in progress_unique.columns:
            n_success = int(progress_unique["ok"].map(as_bool).sum())
        else:
            n_success = len(ok_df)
        n_failed = n_events_attempted - n_success
    else:
        n_success = len(ok_df)
        n_failed = len(fail_event_df)
        n_events_attempted = n_success + n_failed

    success_rate = n_success / max(1, n_events_attempted)
    fail_by_reason = summarize_failures(fail_attempt_df)

    summary_lines = [
        "SCRAPING SUMMARY",
        "",
        f"Status               : {status}",
        f"Events completed     : {n_events_attempted}",
        f"Successfully scraped : {n_success}",
        f"Failed events        : {n_failed}",
        f"Event success rate   : {success_rate:.3f}",
        f"Total URL attempts   : {len(fail_attempt_df) + n_success}",
        f"Workers              : {SCRAPE_WORKERS}",
        f"Timeout seconds      : {SCRAPE_REQUEST_TIMEOUT_SECONDS}",
        "",
        "FAILURES BY REASON",
    ]
    if not fail_by_reason.empty:
        for _, row in fail_by_reason.iterrows():
            summary_lines.append(f"  {row['reason']:<30} {row['n']}")
    else:
        summary_lines.append("  None")

    return "\n".join(summary_lines)


def write_scrape_outputs(ok_rows: list[dict], fail_rows: list[dict], progress_rows: list[dict], status: str):
    ok_df = pd.DataFrame(ok_rows)
    fail_df = pd.DataFrame(fail_rows)
    progress_df = pd.DataFrame(progress_rows)

    ok_df = ensure_columns(ok_df, ARTICLE_COLUMNS)
    fail_df = ensure_columns(fail_df, FAILURE_COLUMNS)
    progress_df = ensure_columns(progress_df, PROGRESS_COLUMNS)

    if not ok_df.empty and "GlobalEventID" in ok_df.columns:
        ok_df = ok_df.drop_duplicates(subset=["GlobalEventID"], keep="last").copy()
    if not progress_df.empty and "GlobalEventID" in progress_df.columns:
        progress_df = progress_df.drop_duplicates(subset=["GlobalEventID"], keep="last").copy()

    if not fail_df.empty:
        dedup_cols = [c for c in ["GlobalEventID", "failure_scope", "candidate_rank", "candidate_url", "reason"] if c in fail_df.columns]
        if dedup_cols:
            fail_df = fail_df.drop_duplicates(subset=dedup_cols, keep="last").copy()

    write_csv_atomic(ok_df, ARTICLES_PATH)
    write_csv_atomic(fail_df, FAILURES_PATH)
    write_csv_atomic(progress_df, PROGRESS_PATH)

    summary_text = build_scraping_summary(ok_df, fail_df, progress_df, status)
    SUMMARY_PATH.write_text(summary_text, encoding="utf-8")

    if status == "complete":
        COMPLETE_FLAG_PATH.write_text(
            f"complete_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            encoding="utf-8",
        )
    elif COMPLETE_FLAG_PATH.exists():
        COMPLETE_FLAG_PATH.unlink()

    fail_attempt_df, fail_event_df = split_failure_rows(fail_df)
    return ok_df, fail_attempt_df, fail_event_df, summary_text


def load_resume_state() -> tuple[list[dict], list[dict], list[dict], set[int]]:
    if not SCRAPE_RESUME:
        if COMPLETE_FLAG_PATH.exists():
            COMPLETE_FLAG_PATH.unlink()
        return [], [], [], set()

    ok_df = read_csv_if_exists(ARTICLES_PATH)
    fail_df = read_csv_if_exists(FAILURES_PATH)
    progress_df = read_csv_if_exists(PROGRESS_PATH)

    if progress_df.empty:
        progress_rows = []
        for geid in sorted(global_event_ids(ok_df)):
            progress_rows.append({
                "GlobalEventID": geid,
                "ok": True,
                "reason": "",
                "resume_source": "articles_file",
            })
        _, fail_event_df = split_failure_rows(fail_df)
        for geid in sorted(global_event_ids(fail_event_df)):
            progress_rows.append({
                "GlobalEventID": geid,
                "ok": False,
                "reason": "failed_previous_run",
                "resume_source": "failures_file",
            })
        progress_df = pd.DataFrame(progress_rows)

    completed_ids = global_event_ids(progress_df)
    if completed_ids:
        print(f"[RESUME] Loaded {len(completed_ids)} completed event(s) from previous checkpoints")

    return (
        ok_df.to_dict("records"),
        fail_df.to_dict("records"),
        progress_df.to_dict("records"),
        completed_ids,
    )


def scrape_one_event(position: int, total: int, event_row: dict, candidates_lookup: dict[int, pd.DataFrame]) -> dict:
    session = get_thread_session()
    geid = int(event_row["GlobalEventID"])
    cands = get_event_candidates_for_scrape(event_row, candidates_lookup)

    ok_rows = []
    fail_attempt_rows = []
    fail_event_rows = []
    log_lines = [f"[SCRAPE] Event {position}/{total}  GlobalEventID={geid}  candidates={len(cands)}"]

    if len(cands) == 0:
        fail_event_rows.append({
            "failure_scope": "event",
            "GlobalEventID": geid,
            "SQLDATE": event_row.get("SQLDATE", ""),
            "ok": False,
            "reason": "no_candidate_urls",
            "n_url_attempts_for_event": 0,
        })
        progress_row = {
            "GlobalEventID": geid,
            "SQLDATE": event_row.get("SQLDATE", ""),
            "ok": False,
            "reason": "no_candidate_urls",
            "n_candidate_urls": 0,
            "n_url_attempts_for_event": 0,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return {
            "ok_rows": ok_rows,
            "fail_attempt_rows": fail_attempt_rows,
            "fail_event_rows": fail_event_rows,
            "progress_row": progress_row,
            "log_lines": log_lines,
        }

    success = False
    n_attempts = 0
    progress_row = None

    for _, cand in cands.iterrows():
        n_attempts += 1
        url = normalize_url(cand.get("candidate_url", ""))
        candidate_rank = as_int(cand.get("candidate_rank", n_attempts), n_attempts)
        is_primary = as_bool(cand.get("is_primary_sourceurl", False))

        res = extract_article_text(session, url)

        attempt_row = {
            "failure_scope": "attempt",
            "GlobalEventID": geid,
            "SQLDATE": event_row.get("SQLDATE", ""),
            "candidate_rank": candidate_rank,
            "candidate_url": url,
            "is_primary_sourceurl": is_primary,
            "ok": bool(res.get("ok", False)),
            "reason": res.get("reason", ""),
            "http_status": res.get("http_status", ""),
            "content_type": res.get("content_type", ""),
            "final_url": res.get("final_url", ""),
            "text_len": res.get("text_len", ""),
            "error": res.get("error", ""),
        }

        if res.get("ok"):
            ok_rows.append({
                "GlobalEventID": geid,
                "SQLDATE": event_row.get("SQLDATE", ""),
                "DATEADDED": event_row.get("DATEADDED", ""),
                "Actor1Name": event_row.get("Actor1Name", ""),
                "Actor2Name": event_row.get("Actor2Name", ""),
                "Actor1CountryCode": event_row.get("Actor1CountryCode", ""),
                "Actor2CountryCode": event_row.get("Actor2CountryCode", ""),
                "ActionGeo_CountryCode": event_row.get("ActionGeo_CountryCode", ""),
                "EventCode": event_row.get("EventCode", ""),
                "EventBaseCode": event_row.get("EventBaseCode", ""),
                "EventRootCode": event_row.get("EventRootCode", ""),
                "QuadClass": event_row.get("QuadClass", ""),
                "NumArticles": event_row.get("NumArticles", ""),
                "NumMentions": event_row.get("NumMentions", ""),
                "NumSources": event_row.get("NumSources", ""),
                "AvgTone": event_row.get("AvgTone", ""),
                "GoldsteinScale": event_row.get("GoldsteinScale", ""),
                "match_rad_term": event_row.get("match_rad_term", False),
                "match_ucdp_actor_term": event_row.get("match_ucdp_actor_term", False),
                "match_gdelt_reb_type": event_row.get("match_gdelt_reb_type", False),
                "match_sources": event_row.get("match_sources", ""),
                "match_rebel_term": event_row.get("match_rebel_term", False),
                "match_rebel_type": event_row.get("match_rebel_type", False),
                "involves_rebel": event_row.get("involves_rebel", False),
                "used_url": url,
                "used_candidate_rank": candidate_rank,
                "used_primary_sourceurl": is_primary,
                "n_url_attempts_for_event": n_attempts,
                "final_url": res.get("final_url", ""),
                "http_status": res.get("http_status", ""),
                "content_type": res.get("content_type", ""),
                "text_len": res.get("text_len", ""),
                "word_count": res.get("word_count", ""),
                "text": res.get("text", ""),
            })

            log_lines.append(
                f"[SCRAPE]   SUCCESS rank={candidate_rank} "
                f"status={res.get('http_status', '')} len={res.get('text_len', '')}"
            )
            progress_row = {
                "GlobalEventID": geid,
                "SQLDATE": event_row.get("SQLDATE", ""),
                "ok": True,
                "reason": "",
                "n_candidate_urls": len(cands),
                "n_url_attempts_for_event": n_attempts,
                "used_candidate_rank": candidate_rank,
                "used_url": url,
                "final_url": res.get("final_url", ""),
                "text_len": res.get("text_len", ""),
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            success = True
            break

        fail_attempt_rows.append(attempt_row)
        log_lines.append(
            f"[SCRAPE]   FAIL rank={candidate_rank} "
            f"reason={res.get('reason', '')} status={res.get('http_status', '')}"
        )

        if SCRAPE_SLEEP_SECONDS > 0:
            time.sleep(SCRAPE_SLEEP_SECONDS)

    if not success:
        fail_event_rows.append({
            "failure_scope": "event",
            "GlobalEventID": geid,
            "SQLDATE": event_row.get("SQLDATE", ""),
            "ok": False,
            "reason": "all_candidates_failed",
            "n_url_attempts_for_event": n_attempts,
        })
        progress_row = {
            "GlobalEventID": geid,
            "SQLDATE": event_row.get("SQLDATE", ""),
            "ok": False,
            "reason": "all_candidates_failed",
            "n_candidate_urls": len(cands),
            "n_url_attempts_for_event": n_attempts,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    return {
        "ok_rows": ok_rows,
        "fail_attempt_rows": fail_attempt_rows,
        "fail_event_rows": fail_event_rows,
        "progress_row": progress_row,
        "log_lines": log_lines,
    }


def scrape_events_with_fallback(events: pd.DataFrame, candidates_df: pd.DataFrame):
    df = events.copy()
    print(f"[SCRAPE] Input event rows: {len(df)}")

    if SCRAPE_EVENT_ROOT_FILTER:
        if "EventRootCode" not in df.columns:
            raise ValueError("EventRootCode column missing in events file")
        before = len(df)
        root = df["EventRootCode"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
        df = df[root.isin(SCRAPE_EVENT_ROOT_FILTER)].copy()
        print(
            f"[SCRAPE] Event root filter {sorted(SCRAPE_EVENT_ROOT_FILTER)} rows: "
            f"{len(df)} / {before}"
        )

    if SCRAPE_REBEL_ONLY:
        if "involves_rebel" not in df.columns:
            raise ValueError("involves_rebel column missing in events file")
        df["involves_rebel"] = df["involves_rebel"].fillna(False).astype(bool)
        df = df[df["involves_rebel"] == True].copy()
        print(f"[SCRAPE] Rebel only event rows: {len(df)}")

    if SCRAPE_MATCH_FILTER != "any_rebel":
        before = len(df)
        rad = df.get("match_rad_term", False)
        ucdp = df.get("match_ucdp_actor_term", False)
        gdelt_type = df.get("match_gdelt_reb_type", False)
        if not hasattr(rad, "fillna"):
            rad = pd.Series(False, index=df.index)
        if not hasattr(ucdp, "fillna"):
            ucdp = pd.Series(False, index=df.index)
        if not hasattr(gdelt_type, "fillna"):
            gdelt_type = pd.Series(False, index=df.index)
        rad = rad.fillna(False).astype(bool)
        ucdp = ucdp.fillna(False).astype(bool)
        gdelt_type = gdelt_type.fillna(False).astype(bool)

        if SCRAPE_MATCH_FILTER == "actor_terms_only":
            keep = rad | ucdp
        elif SCRAPE_MATCH_FILTER == "ucdp_only":
            keep = ucdp
        elif SCRAPE_MATCH_FILTER == "rad_only":
            keep = rad
        elif SCRAPE_MATCH_FILTER == "both_terms":
            keep = rad & ucdp
        elif SCRAPE_MATCH_FILTER == "gdelt_type_only":
            keep = gdelt_type & ~(rad | ucdp)
        else:
            raise ValueError(
                "GDELT_SCRAPE_MATCH_FILTER must be one of: "
                "any_rebel, actor_terms_only, ucdp_only, rad_only, both_terms, gdelt_type_only"
            )

        df = df[keep].copy()
        print(f"[SCRAPE] Match filter {SCRAPE_MATCH_FILTER} rows: {len(df)} / {before}")

    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Drop duplicate events
    df["GlobalEventID"] = pd.to_numeric(df["GlobalEventID"], errors="coerce")
    df = df[df["GlobalEventID"].notna()].drop_duplicates(subset=["GlobalEventID"]).copy()
    df["GlobalEventID"] = df["GlobalEventID"].astype(int)

    n_target = min(SCRAPE_N_EVENTS, len(df))
    print(f"[SCRAPE] Requested events: {SCRAPE_N_EVENTS}")
    print(f"[SCRAPE] Actually sampled events: {n_target}")
    print(f"[SCRAPE] Request timeout seconds: {SCRAPE_REQUEST_TIMEOUT_SECONDS}")
    print(f"[SCRAPE] Match filter: {SCRAPE_MATCH_FILTER}")
    print(f"[SCRAPE] Event root filter: {sorted(SCRAPE_EVENT_ROOT_FILTER) if SCRAPE_EVENT_ROOT_FILTER else 'disabled'}")
    print(f"[SCRAPE] Max candidate URLs per event: {MAX_CANDIDATES_TO_TRY_PER_EVENT}")
    print(f"[SCRAPE] Parallel workers: {SCRAPE_WORKERS}")
    print(f"[SCRAPE] Checkpoint every events: {SCRAPE_CHECKPOINT_EVERY}")
    print(f"[SCRAPE] Resume from checkpoint: {SCRAPE_RESUME}")

    if n_target == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = df.sample(n=n_target, random_state=SCRAPE_SEED).reset_index(drop=True)

    ok_rows, fail_rows, progress_rows, completed_ids = load_resume_state()
    if completed_ids:
        before = len(df)
        df = df[~df["GlobalEventID"].isin(completed_ids)].reset_index(drop=True)
        print(f"[RESUME] Skipping {before - len(df)} already completed event(s); remaining={len(df)}")

    if len(df) == 0:
        ok_df, fail_attempt_df, fail_event_df, summary_text = write_scrape_outputs(
            ok_rows,
            fail_rows,
            progress_rows,
            status="complete",
        )
        print(summary_text)
        return ok_df, fail_attempt_df, fail_event_df

    candidates_lookup = build_candidates_lookup(candidates_df)
    print(f"[SCRAPE] Candidate lookup events: {len(candidates_lookup)}")

    completed_this_run = 0
    total_remaining = len(df)
    total_completed_start = len(global_event_ids(pd.DataFrame(progress_rows)))

    with ThreadPoolExecutor(max_workers=SCRAPE_WORKERS) as executor:
        future_to_geid = {}
        for pos, (_, ev) in enumerate(df.iterrows(), start=1):
            ev_dict = ev.to_dict()
            future = executor.submit(scrape_one_event, pos, total_remaining, ev_dict, candidates_lookup)
            future_to_geid[future] = int(ev_dict["GlobalEventID"])

        for future in as_completed(future_to_geid):
            geid = future_to_geid[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "ok_rows": [],
                    "fail_attempt_rows": [],
                    "fail_event_rows": [{
                        "failure_scope": "event",
                        "GlobalEventID": geid,
                        "ok": False,
                        "reason": "worker_error",
                        "error": str(exc),
                    }],
                    "progress_row": {
                        "GlobalEventID": geid,
                        "ok": False,
                        "reason": "worker_error",
                        "error": str(exc),
                        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    "log_lines": [f"[SCRAPE] Event GlobalEventID={geid} failed in worker: {exc}"],
                }

            ok_rows.extend(result["ok_rows"])
            fail_rows.extend(result["fail_attempt_rows"])
            fail_rows.extend(result["fail_event_rows"])
            if result.get("progress_row") is not None:
                progress_rows.append(result["progress_row"])

            completed_this_run += 1
            completed_total = total_completed_start + completed_this_run

            for line in result["log_lines"]:
                print(line)
            print(f"[SCRAPE] Completed {completed_this_run}/{total_remaining} this run ({completed_total}/{n_target} sampled)")

            if completed_this_run % SCRAPE_CHECKPOINT_EVERY == 0:
                ok_df, fail_attempt_df, fail_event_df, _ = write_scrape_outputs(
                    ok_rows,
                    fail_rows,
                    progress_rows,
                    status="checkpoint",
                )
                print(
                    f"[CHECKPOINT] completed={completed_total}/{n_target} "
                    f"articles={len(ok_df)} failures={len(fail_event_df)} -> {ARTICLES_PATH}"
                )

    ok_df, fail_attempt_df, fail_event_df, summary_text = write_scrape_outputs(
        ok_rows,
        fail_rows,
        progress_rows,
        status="complete",
    )
    print(summary_text)

    return ok_df, fail_attempt_df, fail_event_df

def reorder_candidates_domain_first(cands: pd.DataFrame) -> pd.DataFrame:
    c = cands.copy()

    if len(c) == 0:
        return c

    # Required columns with safe defaults
    if "GlobalEventID" not in c.columns:
        raise ValueError("GlobalEventID column missing in candidates dataframe")
    if "candidate_url" not in c.columns:
        raise ValueError("candidate_url column missing in candidates dataframe")

    c["GlobalEventID"] = pd.to_numeric(c["GlobalEventID"], errors="coerce")
    c = c[c["GlobalEventID"].notna()].copy()
    c["GlobalEventID"] = c["GlobalEventID"].astype(int)

    if "candidate_rank" not in c.columns:
        c["candidate_rank"] = 9999
    c["candidate_rank"] = pd.to_numeric(c["candidate_rank"], errors="coerce").fillna(9999).astype(int)

    if "score" not in c.columns:
        c["score"] = 0.0
    c["score"] = pd.to_numeric(c["score"], errors="coerce").fillna(0.0)

    if "is_primary_sourceurl" not in c.columns:
        c["is_primary_sourceurl"] = False
    c["is_primary_sourceurl"] = c["is_primary_sourceurl"].fillna(False).astype(bool)

    # Domain extraction
    def _domain(u: str) -> str:
        try:
            d = (urlparse(str(u)).netloc or "").lower()
            if d.startswith("www."):
                d = d[4:]
            return d
        except Exception:
            return ""

    c["candidate_domain"] = c["candidate_url"].astype(str).map(_domain)

    out_parts = []

    for geid, g in c.groupby("GlobalEventID", sort=False):
        g = g.copy()

        # Start from existing rank and score ordering
        g = g.sort_values(
            ["is_primary_sourceurl", "candidate_rank", "score"],
            ascending=[False, True, False]
        ).copy()

        # Keep exactly one primary first if present
        primary = g[g["is_primary_sourceurl"] == True].head(1).copy()
        rest = g.loc[~g.index.isin(primary.index)].copy()

        # Domain diversity on the fallback candidates only
        first_per_domain = rest.drop_duplicates(subset=["candidate_domain"]).copy()
        remaining = rest.loc[~rest.index.isin(first_per_domain.index)].copy()

        g_out = pd.concat([primary, first_per_domain, remaining], ignore_index=True)

        # Recompute ranks per event so downstream sorting keeps this order
        g_out["candidate_rank"] = range(1, len(g_out) + 1)

        out_parts.append(g_out)

    out = pd.concat(out_parts, ignore_index=True)

    # Keep original max candidates if your fetch script already limited it
    return out

def main():
    start_time = time.time()
    print("Executing article scraper")
    print("Base dir:", BASE_DIR)

    if not EVENTS_PATH.exists():
        raise FileNotFoundError(f"Missing events file: {EVENTS_PATH}")

    if not CANDIDATES_PATH.exists():
        raise FileNotFoundError(
            f"Missing candidates file: {CANDIDATES_PATH}\n"
            f"Run 03_fetch_eventmentions.py first"
        )

    print(f"[INPUT] Reading events from {EVENTS_PATH}", flush=True)
    events = pd.read_csv(EVENTS_PATH)
    print(f"[INPUT] Reading candidates from {CANDIDATES_PATH}", flush=True)
    candidates = pd.read_csv(CANDIDATES_PATH)

    print(f"[INPUT] Events rows loaded: {len(events)}")
    print(f"[INPUT] Candidate rows loaded: {len(candidates)}")

    print("[INPUT] Reordering candidates", flush=True)
    candidates = reorder_candidates_domain_first(candidates)
    print("[INPUT] Candidate reorder complete", flush=True)

    # Normalize types
    if "GlobalEventID" in candidates.columns:
        candidates["GlobalEventID"] = pd.to_numeric(candidates["GlobalEventID"], errors="coerce")
        candidates = candidates[candidates["GlobalEventID"].notna()].copy()
        candidates["GlobalEventID"] = candidates["GlobalEventID"].astype(int)

    scrape_events_with_fallback(events, candidates)

    print(f"\n[OUTPUT] Articles  -> {ARTICLES_PATH}")
    print(f"[OUTPUT] Failures  -> {FAILURES_PATH}")
    print(f"[OUTPUT] Progress  -> {PROGRESS_PATH}")
    print(f"[OUTPUT] Summary   -> {SUMMARY_PATH}")
    print(f"[OUTPUT] Complete  -> {COMPLETE_FLAG_PATH}")

    elapsed = time.time() - start_time
    print(f"Finished article scraper in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
