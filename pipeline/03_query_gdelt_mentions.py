from pathlib import Path
import os
import re
import time
from datetime import datetime, timezone, timedelta, date

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.getenv("GDELT_DATA_ROOT", str(BASE_DIR / "data" / "raw")))
SERVICE_ACCOUNT_PATH = Path(
    os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS",
        str(BASE_DIR / "config" / "service_account_key.json"),
    )
)

GDELT_OUT_DIR = DATA_ROOT / "gdelt"
GDELT_OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENTS_PATH = GDELT_OUT_DIR / "events_base.csv"
MENTIONS_RAW_PATH = GDELT_OUT_DIR / "eventmentions_raw.csv"
CANDIDATES_PATH = GDELT_OUT_DIR / "event_url_candidates.csv"
MENTION_AGG_PATH = GDELT_OUT_DIR / "event_mentions_agg.csv"

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


CHUNK_SIZE = _env_int("GDELT_MENTIONS_CHUNK_SIZE", 5000)
MAX_EVENTS_FOR_MENTIONS = _env_int("GDELT_MAX_EVENTS_FOR_MENTIONS", 5000)
MAX_CANDIDATES_PER_EVENT = _env_int("GDELT_MAX_CANDIDATES_PER_EVENT", 10)
FETCH_MENTION_AGG = _env_bool("GDELT_FETCH_MENTION_AGG", True)
MENTIONS_MODE = os.getenv("GDELT_MENTIONS_MODE", "").strip().lower()
if not MENTIONS_MODE:
    MENTIONS_MODE = "both" if FETCH_MENTION_AGG else "candidates_only"
if MENTIONS_MODE not in {"both", "candidates_only", "agg_only"}:
    raise ValueError("GDELT_MENTIONS_MODE must be one of: both, candidates_only, agg_only")

# Safety belt: hard cap per query
# 50 GiB is a reasonable starting point. Increase if your date window is larger.
# separate safety belts
MAX_BYTES_BILLED_CANDIDATES = _env_int("GDELT_MAX_BYTES_BILLED_CANDIDATES_GB", 30) * 1024**3
MAX_BYTES_BILLED_AGG = _env_int("GDELT_MAX_BYTES_BILLED_AGG_GB", 60) * 1024**3

# Optional: estimate bytes for every query before running it
DO_DRY_RUN_ESTIMATE = _env_bool("GDELT_DO_DRY_RUN_ESTIMATE", True)

# Use the partitioned GDELT table
EVENTMENTIONS_TABLE = "gdelt-bq.gdeltv2.eventmentions_partitioned"


def make_bq_client() -> bigquery.Client:
    if SERVICE_ACCOUNT_PATH.exists():
        credentials = service_account.Credentials.from_service_account_file(str(SERVICE_ACCOUNT_PATH))
        return bigquery.Client(credentials=credentials, project=credentials.project_id)
    return bigquery.Client()


def normalize_url(url: str) -> str:
    u = str(url or "").strip()
    if not u:
        return ""
    u = re.sub(r"^http://", "https://", u, flags=re.IGNORECASE)

    try:
        p = urlparse(u)
        netloc = (p.netloc or "").lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]

        keep_params = []
        for k, v in parse_qsl(p.query, keep_blank_values=False):
            kl = k.lower()
            if kl.startswith("utm_"):
                continue
            if kl in {"fbclid", "gclid", "mc_cid", "mc_eid"}:
                continue
            keep_params.append((k, v))

        query = urlencode(keep_params, doseq=True)
        path = p.path or "/"
        if path != "/" and path.endswith("/"):
            path = path[:-1]

        return urlunparse(("https", netloc, path, "", query, ""))
    except Exception:
        return u


def extract_domain(url: str) -> str:
    try:
        d = (urlparse(str(url)).netloc or "").lower()
        if d.startswith("www."):
            d = d[4:]
        return d
    except Exception:
        return ""


def _to_event_date(row_value) -> date | None:
    """
    Accepts either EventTimeDate (YYYYMMDDHHMMSS) or SQLDATE (YYYYMMDD).
    Returns a python date or None.
    """
    if pd.isna(row_value):
        return None
    s = str(row_value).strip()
    s = re.sub(r"\.0$", "", s)  # handle csv floats like 20200101123045.0
    s = re.sub(r"\D", "", s)    # keep digits only
    if len(s) >= 8:
        try:
            return datetime.strptime(s[:8], "%Y%m%d").date()
        except Exception:
            return None
    return None


def _midnight_utc(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)


def _run_query_to_df(client: bigquery.Client, query: str, job_config: bigquery.QueryJobConfig, label: str) -> pd.DataFrame:
    if DO_DRY_RUN_ESTIMATE:
        dry = bigquery.QueryJobConfig(
            query_parameters=job_config.query_parameters,
            dry_run=True,
            use_query_cache=False,
        )
        dry_job = client.query(query, job_config=dry)
        # dry_job.total_bytes_processed is available without executing
        est = getattr(dry_job, "total_bytes_processed", None)
        if est is not None:
            print(f"[DRY RUN] {label} estimated bytes processed: {est:,}")

    job = client.query(query, job_config=job_config)
    return job.to_dataframe()


def fetch_first_window_mentions_chunk(
    client: bigquery.Client,
    event_ids: list[int],
    start_ts: datetime,
    end_ts: datetime,
    max_candidates_per_event: int = MAX_CANDIDATES_PER_EVENT,
) -> pd.DataFrame:
    """
    First window mentions for backup urls.
    Partition pruned by _PARTITIONTIME.
    """
    query = f"""
    SELECT
      GlobalEventID,
      EventTimeDate,
      MentionTimeDate,
      MentionType,
      MentionSourceName,
      MentionIdentifier,
      Confidence,
      MentionDocLen,
      MentionDocTone,
      SentenceID
    FROM `{EVENTMENTIONS_TABLE}`
    WHERE _PARTITIONTIME >= @start_ts
      AND _PARTITIONTIME <  @end_ts
      AND GlobalEventID IN UNNEST(@event_ids)
      AND MentionType = 1
      AND MentionTimeDate = EventTimeDate
    QUALIFY ROW_NUMBER() OVER (
      PARTITION BY GlobalEventID
      ORDER BY
        IFNULL(Confidence, 0) DESC,
        IFNULL(MentionDocLen, 0) DESC,
        IFNULL(SentenceID, 9999) ASC
    ) <= @max_candidates_per_event
    """

    job_config = bigquery.QueryJobConfig(
        maximum_bytes_billed=MAX_BYTES_BILLED_CANDIDATES,
        query_parameters=[
            bigquery.ScalarQueryParameter("start_ts", "TIMESTAMP", start_ts),
            bigquery.ScalarQueryParameter("end_ts", "TIMESTAMP", end_ts),
            bigquery.ArrayQueryParameter("event_ids", "INT64", event_ids),
            bigquery.ScalarQueryParameter("max_candidates_per_event", "INT64", int(max_candidates_per_event)),
        ],
    )
    return _run_query_to_df(client, query, job_config, label="candidates")


def fetch_mentions_agg_chunk(
    client: bigquery.Client,
    event_ids: list[int],
    start_ts: datetime,
    end_ts: datetime,
) -> pd.DataFrame:
    """
    Mention aggregates for 15m, 1h, 24h.
    Partition pruned by _PARTITIONTIME.
    Restrict to MentionType = 1 so counts align with article urls.
    """
    query = f"""
    WITH m AS (
      SELECT
        GlobalEventID,
        MentionIdentifier,
        MentionSourceName,
        SAFE.PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(MentionTimeDate AS STRING)) AS mention_ts,
        SAFE.PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(EventTimeDate AS STRING)) AS event_ts
      FROM `{EVENTMENTIONS_TABLE}`
      WHERE _PARTITIONTIME >= @start_ts
        AND _PARTITIONTIME <  @end_ts
        AND GlobalEventID IN UNNEST(@event_ids)
        AND MentionType = 1
    ),
    x AS (
      SELECT
        GlobalEventID,
        MentionIdentifier,
        MentionSourceName,
        TIMESTAMP_DIFF(mention_ts, event_ts, MINUTE) AS delta_min
      FROM m
      WHERE mention_ts IS NOT NULL
        AND event_ts IS NOT NULL
        AND mention_ts >= event_ts
        AND mention_ts <= TIMESTAMP_ADD(event_ts, INTERVAL 24 HOUR)
    )
    SELECT
      GlobalEventID,

      COUNTIF(delta_min <= 15) AS mentions_15m,
      COUNT(DISTINCT IF(delta_min <= 15, MentionIdentifier, NULL)) AS articles_15m,
      COUNT(DISTINCT IF(delta_min <= 15, MentionSourceName, NULL)) AS sources_15m,

      COUNTIF(delta_min <= 60) AS mentions_1h,
      COUNT(DISTINCT IF(delta_min <= 60, MentionIdentifier, NULL)) AS articles_1h,
      COUNT(DISTINCT IF(delta_min <= 60, MentionSourceName, NULL)) AS sources_1h,

      COUNTIF(delta_min <= 1440) AS mentions_24h,
      COUNT(DISTINCT IF(delta_min <= 1440, MentionIdentifier, NULL)) AS articles_24h,
      COUNT(DISTINCT IF(delta_min <= 1440, MentionSourceName, NULL)) AS sources_24h

    FROM x
    GROUP BY GlobalEventID
    """

    job_config = bigquery.QueryJobConfig(
        maximum_bytes_billed=MAX_BYTES_BILLED_AGG,
        query_parameters=[
            bigquery.ScalarQueryParameter("start_ts", "TIMESTAMP", start_ts),
            bigquery.ScalarQueryParameter("end_ts", "TIMESTAMP", end_ts),
            bigquery.ArrayQueryParameter("event_ids", "INT64", event_ids),
        ],
    )
    return _run_query_to_df(client, query, job_config, label="aggregates")


def build_url_candidates(events: pd.DataFrame, mentions_first_window: pd.DataFrame) -> pd.DataFrame:
    e = events.copy()
    m = mentions_first_window.copy()

    e["GlobalEventID"] = pd.to_numeric(e["GlobalEventID"], errors="coerce").astype("Int64")
    e = e[e["GlobalEventID"].notna()].copy()

    if m.empty:
        m = pd.DataFrame(columns=[
            "GlobalEventID", "MentionIdentifier", "MentionType",
            "MentionSourceName", "Confidence", "MentionDocLen", "SentenceID"
        ])

    m["GlobalEventID"] = pd.to_numeric(m["GlobalEventID"], errors="coerce").astype("Int64")
    m = m[m["GlobalEventID"].notna()].copy()

    e["primary_sourceurl"] = e["SOURCEURL"].map(normalize_url)

    if len(m) > 0:
        m["candidate_url"] = m["MentionIdentifier"].astype(str).map(normalize_url)
        m = m[m["candidate_url"].str.len() > 0].copy()

        m = m[~m["candidate_url"].str.contains(r"\.(?:jpg|jpeg|png|gif|webp)(?:\?|$)", case=False, regex=True)]
        m = m[~m["candidate_url"].str.contains(r"\.pdf(?:\?|$)", case=False, regex=True)]

        for col in ["Confidence", "MentionDocLen", "SentenceID", "MentionType"]:
            if col in m.columns:
                m[col] = pd.to_numeric(m[col], errors="coerce")

        m = (
            m.sort_values(
                by=["GlobalEventID", "Confidence", "MentionDocLen", "SentenceID"],
                ascending=[True, False, False, True]
            )
            .drop_duplicates(subset=["GlobalEventID", "candidate_url"])
            .copy()
        )
    else:
        m["candidate_url"] = pd.Series(dtype="string")

    c = m.merge(
        e[["GlobalEventID", "primary_sourceurl", "involves_rebel"]],
        on="GlobalEventID",
        how="right"
    )

    primary_rows = e[["GlobalEventID", "primary_sourceurl", "involves_rebel"]].copy()
    primary_rows = primary_rows.rename(columns={"primary_sourceurl": "candidate_url"})
    primary_rows["MentionType"] = 0
    primary_rows["MentionSourceName"] = ""
    primary_rows["Confidence"] = 0.0
    primary_rows["MentionDocLen"] = 0
    primary_rows["SentenceID"] = 9999

    c = pd.concat([c, primary_rows], ignore_index=True)

    c["candidate_url"] = c["candidate_url"].fillna("").map(normalize_url)
    c = c[c["candidate_url"] != ""].copy()

    c["primary_sourceurl"] = c["primary_sourceurl"].fillna("").map(normalize_url)
    c["is_primary_sourceurl"] = c["candidate_url"] == c["primary_sourceurl"]

    c = c.drop_duplicates(subset=["GlobalEventID", "candidate_url"]).copy()

    c["Confidence"] = pd.to_numeric(c["Confidence"], errors="coerce").fillna(0)
    c["MentionDocLen"] = pd.to_numeric(c["MentionDocLen"], errors="coerce").fillna(0)
    c["SentenceID"] = pd.to_numeric(c["SentenceID"], errors="coerce").fillna(9999)
    c["MentionType"] = pd.to_numeric(c["MentionType"], errors="coerce").fillna(0)

    c["score"] = 0.0
    c["score"] += c["is_primary_sourceurl"].astype(int) * 100.0
    c["score"] += (c["MentionType"] == 1).astype(int) * 20.0
    c["score"] += c["Confidence"]
    c["score"] += (c["MentionDocLen"] / 1000.0).clip(0, 20)
    c["score"] += (100 - c["SentenceID"].clip(0, 100)) * 0.1

    c["candidate_domain"] = c["candidate_url"].map(extract_domain)
    c = c.sort_values(["GlobalEventID", "score"], ascending=[True, False]).copy()

    best_per_domain = c.drop_duplicates(subset=["GlobalEventID", "candidate_domain"]).copy()

    remaining = c.merge(
        best_per_domain[["GlobalEventID", "candidate_url"]],
        on=["GlobalEventID", "candidate_url"],
        how="left",
        indicator=True
    )
    remaining = remaining[remaining["_merge"] == "left_only"].drop(columns=["_merge"])

    combined = pd.concat([best_per_domain, remaining], ignore_index=True)
    combined = combined.sort_values(["GlobalEventID", "score"], ascending=[True, False]).copy()
    combined["candidate_rank"] = combined.groupby("GlobalEventID").cumcount() + 1

    c = combined[combined["candidate_rank"] <= MAX_CANDIDATES_PER_EVENT].copy()

    cols = [
        "GlobalEventID",
        "candidate_rank",
        "candidate_url",
        "candidate_domain",
        "score",
        "is_primary_sourceurl",
        "MentionType",
        "MentionSourceName",
        "Confidence",
        "MentionDocLen",
        "SentenceID",
        "involves_rebel",
    ]
    cols = [x for x in cols if x in c.columns]

    return c[cols].reset_index(drop=True)


def main():
    start_time = time.time()
    print("Executing mentions fetch")
    print("Base dir:", BASE_DIR)
    print("Mentions mode:", MENTIONS_MODE)

    if not EVENTS_PATH.exists():
        raise FileNotFoundError(f"Missing events file: {EVENTS_PATH}")

    events = pd.read_csv(EVENTS_PATH)

    if "GlobalEventID" not in events.columns:
        raise RuntimeError("events_base.csv must include GlobalEventID")

    # Derive an event date for partition pruning
    if "EventTimeDate" in events.columns:
        events["_event_date"] = events["EventTimeDate"].apply(_to_event_date)
    elif "SQLDATE" in events.columns:
        events["_event_date"] = events["SQLDATE"].apply(_to_event_date)
    else:
        raise RuntimeError("events_base.csv must include EventTimeDate or SQLDATE so we can set _PARTITIONTIME filters safely")

    events["GlobalEventID"] = pd.to_numeric(events["GlobalEventID"], errors="coerce")
    events = events[events["GlobalEventID"].notna()].copy()
    events = events[events["_event_date"].notna()].copy()

    if "involves_rebel" in events.columns:
        before = len(events)
        events = events[events["involves_rebel"].astype(bool)].copy()
        print(f"[INPUT] Filtered to involves_rebel=True: {len(events)} / {before}")

    event_ids = events["GlobalEventID"].astype(int).unique().tolist()

    print(f"[INPUT] Loaded events rows: {len(events)}")
    print(f"[INPUT] Unique GlobalEventID count: {len(event_ids)}")

    if len(event_ids) == 0:
        raise RuntimeError("No GlobalEventID values found in events file")

    if len(event_ids) > MAX_EVENTS_FOR_MENTIONS:
        raise RuntimeError(
            f"Refusing mentions fetch for {len(event_ids)} events. "
            f"MAX_EVENTS_FOR_MENTIONS is {MAX_EVENTS_FOR_MENTIONS}"
        )

    client = make_bq_client()

    # Batch by day so partition filters stay tight
    day_to_ids = (
        events.groupby("_event_date")["GlobalEventID"]
        .apply(lambda s: s.astype(int).unique().tolist())
        .to_dict()
    )

    day_chunks = []
    for d, ids in sorted(day_to_ids.items(), key=lambda x: x[0]):
        for i in range(0, len(ids), CHUNK_SIZE):
            day_chunks.append((d, ids[i:i + CHUNK_SIZE]))

    print(f"[PLAN] Day chunks: {len(day_chunks)}")

    if MENTIONS_MODE in {"both", "candidates_only"}:
        # 1) First window mention rows for url fallback candidates
        mention_parts = []
        for i, (d, chunk_ids) in enumerate(day_chunks, 1):
            start_ts = _midnight_utc(d)
            end_ts = start_ts + timedelta(days=1)  # same day partitions only
            print(f"[CANDIDATES] {i}/{len(day_chunks)} date={d} events={len(chunk_ids)} partition=[{start_ts} , {end_ts})")
            part = fetch_first_window_mentions_chunk(client, chunk_ids, start_ts, end_ts, MAX_CANDIDATES_PER_EVENT)
            mention_parts.append(part)

        mentions_raw = pd.concat(mention_parts, ignore_index=True) if mention_parts else pd.DataFrame()
        print(f"[CANDIDATES] Mention rows fetched: {len(mentions_raw)}")
        mentions_raw.to_csv(MENTIONS_RAW_PATH, index=False)
        print(f"[CANDIDATES] Saved mention rows to {MENTIONS_RAW_PATH}")

        candidates = build_url_candidates(events, mentions_raw)
        candidates.to_csv(CANDIDATES_PATH, index=False)
        print(f"[CANDIDATES] Saved url candidates to {CANDIDATES_PATH}")
        print(f"[CANDIDATES] Candidate rows: {len(candidates)}")
        print(f"[CANDIDATES] Unique events in candidates: {candidates['GlobalEventID'].nunique() if len(candidates) else 0}")
    elif CANDIDATES_PATH.exists():
        print(f"[CANDIDATES] agg_only mode: preserving existing {CANDIDATES_PATH}")
        candidates = pd.read_csv(CANDIDATES_PATH, low_memory=False)
    else:
        print("[CANDIDATES] agg_only mode: no existing candidate file; using primary SOURCEURL only in links output")
        candidates = build_url_candidates(events, pd.DataFrame())

    # 2) Mention aggregates for 15m, 1h, 24h. These are useful diagnostics and
    # possible media-salience controls, but not required for URL scraping.
    if MENTIONS_MODE in {"both", "agg_only"}:
        agg_parts = []
        for i, (d, chunk_ids) in enumerate(day_chunks, 1):
            start_ts = _midnight_utc(d)
            end_ts = start_ts + timedelta(days=2)  # include next day for 24h window
            print(f"[AGG] {i}/{len(day_chunks)} date={d} events={len(chunk_ids)} partition=[{start_ts} , {end_ts})")
            part = fetch_mentions_agg_chunk(client, chunk_ids, start_ts, end_ts)
            agg_parts.append(part)

        mention_agg = pd.concat(agg_parts, ignore_index=True) if agg_parts else pd.DataFrame()
    else:
        print("[AGG] Skipping mention aggregates")
        mention_agg = pd.DataFrame()

    if len(mention_agg) == 0:
        mention_agg = pd.DataFrame({"GlobalEventID": event_ids})

    mention_agg["GlobalEventID"] = pd.to_numeric(mention_agg["GlobalEventID"], errors="coerce")
    mention_agg = pd.DataFrame({"GlobalEventID": event_ids}).merge(
        mention_agg, on="GlobalEventID", how="left"
    )

    for col in mention_agg.columns:
        if col != "GlobalEventID":
            mention_agg[col] = pd.to_numeric(mention_agg[col], errors="coerce").fillna(0).astype(int)

    if MENTIONS_MODE in {"both", "agg_only"}:
        mention_agg.to_csv(MENTION_AGG_PATH, index=False)
        print(f"[AGG] Saved mention aggregates to {MENTION_AGG_PATH}")
    elif MENTION_AGG_PATH.exists():
        print(f"[AGG] Preserving existing mention aggregates at {MENTION_AGG_PATH}")
    else:
        print("[AGG] No mention aggregate file written")

    # ------------------------------------------------------------------
    # Build dataset 01: article links joined with mention counts
    # One row per (event, candidate URL) with event context attached.
    # This is the input-selection dataset for the scraping step.
    # ------------------------------------------------------------------
    DATASETS_DIR = DATA_ROOT / "datasets"
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    links = candidates.merge(mention_agg, on="GlobalEventID", how="left")

    event_meta_cols = [c for c in [
        "GlobalEventID", "SQLDATE", "ActionGeo_CountryCode",
        "Actor1Name", "Actor2Name", "GoldsteinScale", "AvgTone",
        "match_rad_term", "match_ucdp_actor_term", "match_gdelt_reb_type",
        "match_sources",
        "involves_rebel", "SOURCEURL",
    ] if c in events.columns]
    event_meta = events[event_meta_cols].drop_duplicates(subset=["GlobalEventID"])
    links = links.merge(event_meta, on="GlobalEventID", how="left")

    ordered_cols = [c for c in [
        "GlobalEventID", "SQLDATE", "ActionGeo_CountryCode",
        "Actor1Name", "Actor2Name", "GoldsteinScale", "AvgTone",
        "match_rad_term", "match_ucdp_actor_term", "match_gdelt_reb_type",
        "match_sources",
        "involves_rebel", "SOURCEURL",
        "candidate_rank", "candidate_url", "candidate_domain",
        "score", "is_primary_sourceurl", "MentionType", "Confidence", "MentionDocLen",
        "mentions_15m", "articles_15m", "sources_15m",
        "mentions_1h", "articles_1h", "sources_1h",
        "mentions_24h", "articles_24h", "sources_24h",
    ] if c in links.columns]
    extra_cols = [c for c in links.columns if c not in ordered_cols]
    links = links[ordered_cols + extra_cols]

    links_path = DATASETS_DIR / "01_links_and_mentions.csv"
    links.to_csv(links_path, index=False)
    print(f"[DATASETS] Saved {len(links)} rows to {links_path}")

    elapsed = time.time() - start_time
    print(f"Finished mentions fetch in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
