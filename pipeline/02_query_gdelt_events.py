from pathlib import Path

print("[BOOT] Starting GDELT event query", flush=True)

import re
import os
import time
from datetime import datetime
import random
import requests


import pandas as pd

print("[BOOT] Importing google.cloud.bigquery and credentials", flush=True)
from google.cloud import bigquery
from google.oauth2 import service_account
print("[BOOT] Google BigQuery imports complete", flush=True)

print("CWD:", Path.cwd(), flush=True)

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


SCRAPE_ARTICLES = _env_bool("GDELT_SCRAPE_ARTICLES", False)
SCRAPE_N_UNIQUE_URLS = _env_int("GDELT_SCRAPE_N_UNIQUE_URLS", 100)
SCRAPE_SLEEP_SECONDS = float(os.getenv("GDELT_SCRAPE_SLEEP_SECONDS", "0.8"))
SCRAPE_SEED = _env_int("GDELT_SCRAPE_SEED", 42)

SAMPLE_MODE = _env_bool("GDELT_SAMPLE_MODE", True)
SAMPLE_LIMIT = _env_int("GDELT_SAMPLE_LIMIT", 1000)
SAMPLE_SEED = _env_int("GDELT_SAMPLE_SEED", 42)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.getenv("GDELT_DATA_ROOT", str(BASE_DIR / "data" / "raw")))

SERVICE_ACCOUNT_PATH = Path(
    os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS",
        str(BASE_DIR / "config" / "service_account_key.json"),
    )
)
LAST_RUN_PATH = DATA_ROOT / "gdelt" / "last_run_date.txt"

ARTICLES_DIR = DATA_ROOT / "articles"
ARTICLES_DIR.mkdir(parents=True, exist_ok=True)

GDELT_OUT_DIR = DATA_ROOT / "gdelt"
GDELT_OUT_DIR.mkdir(parents=True, exist_ok=True)

PREPROCESS_OUT_DIR = DATA_ROOT / "preprocessing"
PREPROCESS_OUT_DIR.mkdir(parents=True, exist_ok=True)

ACTOR_TERMS_PATH = Path(os.getenv(
    "GDELT_ACTOR_TERMS_PATH",
    str(PREPROCESS_OUT_DIR / "cleaned_actor_terms.csv"),
))
ACTOR_TERMS_FALLBACK = BASE_DIR / "data" / "reference" / "cleaned_actor_terms.csv"
REBEL_TERMS_PATH_FALLBACK = BASE_DIR / "data" / "reference" / "cleaned_rebel_terms.csv"

# Apply actor names inside the BigQuery WHERE clause as well as after fetch.
# This prevents UCDP actors with missing/incomplete GDELT actor-type codes from
# being missed before pandas matching can see them.
GDELT_USE_BQ_NAME_FILTER = _env_bool("GDELT_USE_BQ_NAME_FILTER", True)




type_codes = [
    "reb", "sep", "ins", "rad", "uaf", "crm", "img", "mil", "spy", "opp", "uis", "set"
]

event_root_codes = ["19", "20"]

event_base_codes = [
    "036", "037", "062", "064", "072", "074", "084", "087", "093", "094", "130",
    "133", "138", "139", "145", "150", "151", "152", "153", "154", "170", "171",
    "175", "180", "181", "182",
]

event_codes = [
    "0212", "0214", "0232", "0234", "0252", "0253", "0255", "0256", "0312", "0314",
    "0332", "0334", "0353", "0356", "0831", "0834", "1012", "1014", "1032", "1034",
    "1056", "1123", "1124", "1212", "1222", "1224", "1324", "1383", "1384", "1622",
    "1711",
]


def load_actor_terms() -> pd.DataFrame:
    terms_path = ACTOR_TERMS_PATH
    if not terms_path.exists():
        terms_path = ACTOR_TERMS_FALLBACK if ACTOR_TERMS_FALLBACK.exists() else REBEL_TERMS_PATH_FALLBACK

    if not terms_path.exists():
        raise FileNotFoundError(
            f"No actor term file found. Run pipeline/01_build_actor_terms.py first. "
            f"Expected {ACTOR_TERMS_PATH}"
        )

    df = pd.read_csv(terms_path, low_memory=False)
    if "term_lower" not in df.columns:
        term_col = "term" if "term" in df.columns else df.columns[0]
        df["term_lower"] = df[term_col].fillna("").astype(str).str.lower()

    if "term_source_dataset" not in df.columns:
        df["term_source_dataset"] = "RAD"
    if "use_for_matching" not in df.columns:
        df["use_for_matching"] = True

    df["term_lower"] = (
        df["term_lower"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df[df["term_lower"].ne("")].copy()
    df = df[df["use_for_matching"].astype(bool)].copy()

    return df.reset_index(drop=True)


def _unique_terms(terms_df: pd.DataFrame, source: str | None = None) -> list[str]:
    df = terms_df
    if source is not None:
        df = df[df["term_source_dataset"].astype(str).eq(source)]
    return sorted(set(df["term_lower"].dropna().astype(str)))


def build_pandas_pattern(terms: list[str]) -> str:
    if not terms:
        return r"(?!x)x"
    escaped = [re.escape(t) for t in sorted(set(terms), key=len, reverse=True)]
    return r"(?<![A-Za-z0-9])(?:%s)(?![A-Za-z0-9])" % "|".join(escaped)


def build_bq_pattern(terms: list[str]) -> str:
    if not terms:
        return r"(?!x)x"

    def escape_re2(term: str) -> str:
        # BigQuery uses RE2. Escape regex metacharacters, but leave spaces as
        # literal spaces; Python's re.escape emits "\ ", which RE2 rejects.
        return re.sub(r"([\\.^$*+?{}\[\]|()])", r"\\\1", term)

    escaped = [escape_re2(t) for t in sorted(set(terms), key=len, reverse=True)]
    return r"(^|[^A-Za-z0-9])(?:%s)($|[^A-Za-z0-9])" % "|".join(escaped)


def get_date_window() -> tuple[int, int]:
    start_override = os.getenv("GDELT_START_DATE", "").strip()
    end_override = os.getenv("GDELT_END_DATE", "").strip()
    if start_override or end_override:
        if not (start_override and end_override):
            raise ValueError("Set both GDELT_START_DATE and GDELT_END_DATE, or neither")
        return int(start_override), int(end_override)

    today = datetime.today()
    today_date = int(today.strftime("%Y%m%d"))

    if SAMPLE_MODE:
        return 20210101, 20210103

    if LAST_RUN_PATH.exists():
        last_run_str = LAST_RUN_PATH.read_text().strip()
        if last_run_str:
            return int(last_run_str), today_date

    return 20180101, today_date


def save_last_run_date(end_date: int) -> None:
    LAST_RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_RUN_PATH.write_text(str(end_date))


def make_bq_client() -> bigquery.Client:
    if SERVICE_ACCOUNT_PATH.exists():
        print(f"[BQ] Loading service account from {SERVICE_ACCOUNT_PATH}", flush=True)
        credentials = service_account.Credentials.from_service_account_file(str(SERVICE_ACCOUNT_PATH))
        return bigquery.Client(credentials=credentials, project=credentials.project_id)
    print("[BQ] Using Google Application Default Credentials", flush=True)
    return bigquery.Client()


def write_rebel_summary(
    actor_terms: pd.DataFrame,
    events: pd.DataFrame,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    term_values = sorted(set(actor_terms["term_lower"].dropna().astype(str)))
    n_terms = len(term_values)
    longest_terms = sorted(term_values, key=len, reverse=True)[:20]

    rad_match = int(events["match_rad_term"].sum()) if "match_rad_term" in events.columns else 0
    ucdp_match = int(events["match_ucdp_actor_term"].sum()) if "match_ucdp_actor_term" in events.columns else 0
    type_match = int(events["match_gdelt_reb_type"].sum()) if "match_gdelt_reb_type" in events.columns else 0
    either = int(events["involves_rebel"].sum())
    total = len(events)

    lines = []
    lines.append("Actor term loading summary")
    lines.append(f"Number of actor terms loaded for matching: {n_terms}")
    lines.append("")
    lines.append("Rows by source dataset")
    if "term_source_dataset" in actor_terms.columns:
        lines.append(actor_terms["term_source_dataset"].value_counts().to_string())
    lines.append("")
    lines.append("Top 20 longest terms")
    for t in longest_terms:
        lines.append(f"{t}  |  length={len(t)}")
    lines.append("")
    lines.append("Event matching summary")
    lines.append(f"Total events fetched: {total}")
    lines.append(f"Match by RAD term: {rad_match}")
    lines.append(f"Match by UCDP actor term: {ucdp_match}")
    lines.append(f"Match by GDELT REB type: {type_match}")
    lines.append(f"Match by either (involves_rebel): {either}")
    lines.append("")
    if "match_sources" in events.columns and total > 0:
        lines.append("Match source combinations")
        lines.append(events["match_sources"].replace("", "NONE").value_counts().head(30).to_string())
        lines.append("")
    if total > 0:
        lines.append("Shares")
        lines.append(f"Share RAD term: {rad_match / total:.4f}")
        lines.append(f"Share UCDP actor term: {ucdp_match / total:.4f}")
        lines.append(f"Share GDELT REB type: {type_match / total:.4f}")
        lines.append(f"Share either: {either / total:.4f}")

    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")

    print(text)

def export_match_examples(
    events: pd.DataFrame,
    out_csv: Path,
    n_each: int = 50,
    seed: int = 42,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "GlobalEventID",
        "SQLDATE",
        "Actor1Name",
        "Actor2Name",
        "Actor1CountryCode",
        "Actor2CountryCode",
        "ActionGeo_CountryCode",
        "EventCode",
        "EventBaseCode",
        "EventRootCode",
        "QuadClass",
        "NumArticles",
        "NumMentions",
        "AvgTone",
        "SOURCEURL",
        "match_rad_term",
        "match_ucdp_actor_term",
        "match_gdelt_reb_type",
        "match_sources",
        "involves_rebel",
    ]
    cols = [c for c in cols if c in events.columns]

    pos = events[events["involves_rebel"] == True]
    neg = events[events["involves_rebel"] == False]

    pos_s = pos.sample(n=min(n_each, len(pos)), random_state=seed) if len(pos) else pos
    neg_s = neg.sample(n=min(n_each, len(neg)), random_state=seed) if len(neg) else neg

    out = pd.concat([pos_s, neg_s], ignore_index=True)[cols]
    out.to_csv(out_csv, index=False)

    print(f"Saved match examples to {out_csv} with {len(out)} rows")


# 10 GiB hard cap — raises google.api_core.exceptions.Forbidden if exceeded.
# GDELT events_partitioned is ~200 MB/day, so a 1-year window costs ~70 GB;
# raise this limit only after running a dry-run estimate first.
MAX_BYTES_BILLED = _env_int("GDELT_MAX_BYTES_BILLED_GB", 10) * 1024 ** 3

# Set to True to print the estimated bytes before running the real query.
DO_DRY_RUN_ESTIMATE = _env_bool("GDELT_DO_DRY_RUN_ESTIMATE", True)


def _sqldate_to_timestamp(sqldate: int) -> str:
    """Convert GDELT SQLDATE integer (YYYYMMDD) to a TIMESTAMP string for _PARTITIONTIME."""
    s = str(sqldate)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]} 00:00:00 UTC"


def fetch_events_base(
    client: bigquery.Client,
    start_date: int,
    end_date: int,
    actor_name_bq_pattern: str,
) -> pd.DataFrame:
    # Use events_partitioned so _PARTITIONTIME filters actually prune partitions.
    # events (non-partitioned) causes a full-table scan (~500 GB) regardless of WHERE clauses.
    query = """
    SELECT
      GlobalEventID,
      SQLDATE,
      DATEADDED,

      Actor1Name,
      Actor1CountryCode,
      Actor1Type1Code,
      Actor1Type2Code,
      Actor1Type3Code,

      Actor2Name,
      Actor2CountryCode,
      Actor2Type1Code,
      Actor2Type2Code,
      Actor2Type3Code,

      CAST(EventCode AS STRING) AS EventCode,
      CAST(EventBaseCode AS STRING) AS EventBaseCode,
      CAST(EventRootCode AS STRING) AS EventRootCode,
      QuadClass,
      GoldsteinScale,

      NumMentions,
      NumSources,
      NumArticles,
      AvgTone,

      ActionGeo_FullName,
      ActionGeo_CountryCode,
      ActionGeo_ADM1Code,
      ActionGeo_ADM2Code,
      ActionGeo_Lat,
      ActionGeo_Long,
      ActionGeo_FeatureID,

      SOURCEURL

    FROM `gdelt-bq.gdeltv2.events_partitioned`
    WHERE
      _PARTITIONTIME >= TIMESTAMP(@start_ts)
      AND _PARTITIONTIME < TIMESTAMP(@end_ts)
      AND NumArticles > 1
      AND (
        LOWER(Actor1Type1Code) IN UNNEST(@type_codes) OR
        LOWER(Actor1Type2Code) IN UNNEST(@type_codes) OR
        LOWER(Actor1Type3Code) IN UNNEST(@type_codes) OR
        LOWER(Actor2Type1Code) IN UNNEST(@type_codes) OR
        LOWER(Actor2Type2Code) IN UNNEST(@type_codes) OR
        LOWER(Actor2Type3Code) IN UNNEST(@type_codes)
        OR (
          @use_name_filter = TRUE AND (
            REGEXP_CONTAINS(LOWER(IFNULL(Actor1Name, '')), @actor_name_pattern) OR
            REGEXP_CONTAINS(LOWER(IFNULL(Actor2Name, '')), @actor_name_pattern)
          )
        )
      )
      AND (
        CAST(EventRootCode AS STRING) IN UNNEST(@event_root_codes) OR
        CAST(EventBaseCode AS STRING) IN UNNEST(@event_base_codes) OR
        CAST(EventCode AS STRING) IN UNNEST(@event_codes)
      )
    """

    if SAMPLE_MODE:
        query += """
    ORDER BY FARM_FINGERPRINT(CONCAT(CAST(GlobalEventID AS STRING), CAST(@sample_seed AS STRING)))
    LIMIT @sample_limit
    """

    params = [
        bigquery.ScalarQueryParameter("start_ts", "STRING", _sqldate_to_timestamp(start_date)),
        bigquery.ScalarQueryParameter("end_ts",   "STRING", _sqldate_to_timestamp(end_date)),
        bigquery.ArrayQueryParameter("type_codes", "STRING", type_codes),
        bigquery.ArrayQueryParameter("event_root_codes", "STRING", event_root_codes),
        bigquery.ArrayQueryParameter("event_base_codes", "STRING", event_base_codes),
        bigquery.ArrayQueryParameter("event_codes", "STRING", event_codes),
        bigquery.ScalarQueryParameter("actor_name_pattern", "STRING", actor_name_bq_pattern),
        bigquery.ScalarQueryParameter("use_name_filter", "BOOL", GDELT_USE_BQ_NAME_FILTER),
    ]
    if SAMPLE_MODE:
        params.append(bigquery.ScalarQueryParameter("sample_limit", "INT64", SAMPLE_LIMIT))
        params.append(bigquery.ScalarQueryParameter("sample_seed", "INT64", SAMPLE_SEED))

    if DO_DRY_RUN_ESTIMATE:
        print("[BQ] Starting dry run for events query", flush=True)
        dry_cfg = bigquery.QueryJobConfig(query_parameters=params, dry_run=True, use_query_cache=False)
        dry_job = client.query(query, job_config=dry_cfg)
        gb = dry_job.total_bytes_processed / 1024 ** 3
        cost_usd = gb / 1024 * 6.25   # $6.25 per TiB
        print(f"[DRY RUN] events_partitioned estimate: {gb:.2f} GB  (~${cost_usd:.4f})")

    print("[BQ] Starting events query", flush=True)
    job_config = bigquery.QueryJobConfig(
        query_parameters=params,
        maximum_bytes_billed=MAX_BYTES_BILLED,
    )
    df = client.query(query, job_config=job_config).to_dataframe()
    print(f"[BQ] Events query returned {len(df)} rows before GlobalEventID dedup", flush=True)

    df = df.drop_duplicates(subset=["GlobalEventID"]).reset_index(drop=True)
    return df


def add_rebel_flags(
    df: pd.DataFrame,
    rad_pattern: str,
    ucdp_pattern: str,
) -> pd.DataFrame:
    df = df.copy()

    df["event_date"] = pd.to_datetime(df["SQLDATE"].astype(str), format="%Y%m%d", errors="coerce")

    actor1 = df["Actor1Name"].fillna("").astype(str).str.lower()
    actor2 = df["Actor2Name"].fillna("").astype(str).str.lower()

    df["actor1_match_rad_term"] = actor1.str.contains(rad_pattern, case=False, regex=True)
    df["actor2_match_rad_term"] = actor2.str.contains(rad_pattern, case=False, regex=True)
    df["match_rad_term"] = df["actor1_match_rad_term"] | df["actor2_match_rad_term"]

    df["actor1_match_ucdp_actor_term"] = actor1.str.contains(ucdp_pattern, case=False, regex=True)
    df["actor2_match_ucdp_actor_term"] = actor2.str.contains(ucdp_pattern, case=False, regex=True)
    df["match_ucdp_actor_term"] = (
        df["actor1_match_ucdp_actor_term"] | df["actor2_match_ucdp_actor_term"]
    )

    # Backward-compatible name-match field for old diagnostics.
    df["match_rebel_term"] = df["match_rad_term"] | df["match_ucdp_actor_term"]

    type_cols = [
        "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
        "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    ]
    rebel_type = pd.Series(False, index=df.index)
    for c in type_cols:
        rebel_type = rebel_type | df[c].fillna("").str.lower().eq("reb")
    df["match_gdelt_reb_type"] = rebel_type
    df["match_rebel_type"] = rebel_type  # backward-compatible alias

    def sources(row: pd.Series) -> str:
        out = []
        if bool(row["match_rad_term"]):
            out.append("RAD")
        if bool(row["match_ucdp_actor_term"]):
            out.append("UCDP_ACTOR")
        if bool(row["match_gdelt_reb_type"]):
            out.append("GDELT_REB")
        return ";".join(out)

    df["involves_rebel"] = (
        df["match_rad_term"] | df["match_ucdp_actor_term"] | df["match_gdelt_reb_type"]
    )
    df["match_sources"] = df.apply(sources, axis=1)

    return df


def build_country_week_panel(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()

    df = df[df["event_date"].notna()].copy()
    df["week_start_date"] = df["event_date"] - pd.to_timedelta(df["event_date"].dt.weekday, unit="D")
    df["country_code"] = df["ActionGeo_CountryCode"].fillna("").astype(str)

    df = df[df["country_code"] != ""].copy()

    g = df.groupby(["country_code", "week_start_date"], as_index=False)

    panel = g.agg(
        n_events_total=("GlobalEventID", "count"),
        n_events_rebel=("involves_rebel", "sum"),
        mean_goldstein=("GoldsteinScale", "mean"),
        mean_tone=("AvgTone", "mean"),
        median_tone=("AvgTone", "median"),
        total_nummentions_firstwindow=("NumMentions", "sum"),
        total_numsources_firstwindow=("NumSources", "sum"),
        total_numarticles_firstwindow=("NumArticles", "sum"),
        max_nummentions_firstwindow=("NumMentions", "max"),
    )

    panel["share_events_rebel"] = panel["n_events_rebel"] / panel["n_events_total"]

    quad = (
        df.pivot_table(
            index=["country_code", "week_start_date"],
            columns="QuadClass",
            values="GlobalEventID",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )
    quad.columns = [str(c) for c in quad.columns]
    quad = quad.rename(
        columns={
            "1": "n_events_quadclass_1",
            "2": "n_events_quadclass_2",
            "3": "n_events_quadclass_3",
            "4": "n_events_quadclass_4",
        }
    )

    panel = panel.merge(quad, on=["country_code", "week_start_date"], how="left")
    for c in ["n_events_quadclass_1", "n_events_quadclass_2", "n_events_quadclass_3", "n_events_quadclass_4"]:
        if c not in panel.columns:
            panel[c] = 0

    panel["year"] = panel["week_start_date"].dt.year
    panel["week_of_year"] = panel["week_start_date"].dt.isocalendar().week.astype(int)

    return panel


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en,en-US;q=0.9",
    })
    return s


def normalize_url(url: str) -> str:
    return (url or "").strip()


def is_probably_pdf(url: str) -> bool:
    u = (url or "").lower().strip()
    return u.endswith(".pdf") or ".pdf?" in u


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


def fetch_html(session: requests.Session, url: str, timeout: int = 20) -> tuple[int, str, str, str]:
    r = session.get(url, timeout=timeout, allow_redirects=True)
    status = int(r.status_code)
    final_url = str(r.url)
    content_type = (r.headers.get("Content-Type") or "").lower()
    html = ""
    if "text/html" in content_type or "application/xhtml" in content_type or content_type == "":
        r.encoding = r.encoding or "utf-8"
        html = r.text
    return status, final_url, content_type, html


def extract_article_text(session: requests.Session, url: str) -> dict:
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
        return {"ok": False, "reason": "http_error", "final_url": final_url, "http_status": status, "content_type": content_type}

    if not html:
        return {"ok": False, "reason": "no_html", "final_url": final_url, "http_status": status, "content_type": content_type}

    text = try_trafilatura(html)
    if len(text) < 200:
        text = try_bs4(html)

    text = (text or "").strip()
    if len(text) < 200:
        return {
            "ok": False,
            "reason": "extraction_too_short",
            "final_url": final_url,
            "http_status": status,
            "content_type": content_type,
            "text_len": len(text),
        }

    max_chars = 20000
    if len(text) > max_chars:
        text = text[:max_chars]

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


def scrape_sample_articles(events: pd.DataFrame) -> None:
    if "SOURCEURL" not in events.columns:
        raise ValueError("SOURCEURL column not found in events dataframe")

    df = events.copy()
    print(f"[SCRAPE] Input event rows: {len(df)}")

    if "involves_rebel" in df.columns:
        n_rebel = int(df["involves_rebel"].fillna(False).astype(bool).sum())
        print(f"[SCRAPE] Rebel flagged event rows: {n_rebel}")
        print(f"[SCRAPE] Non rebel event rows: {len(df) - n_rebel}")

    # Normalize and keep non empty URLs
    df["SOURCEURL"] = df["SOURCEURL"].fillna("").astype(str).map(normalize_url)
    n_nonempty = int((df["SOURCEURL"] != "").sum())
    print(f"[SCRAPE] Rows with non empty SOURCEURL: {n_nonempty}")

    # Optional visibility on duplicates before dropping them
    n_unique_before = int(df["SOURCEURL"][df["SOURCEURL"] != ""].nunique())
    print(f"[SCRAPE] Unique non empty SOURCEURL before dedup: {n_unique_before}")

    df = df[df["SOURCEURL"] != ""].copy()

    # Show top duplicate URLs (helps explain why 100 events can become much fewer URLs)
    dup_counts = df["SOURCEURL"].value_counts()
    n_dupe_rows = int((dup_counts > 1).sum())
    print(f"[SCRAPE] Number of URLs appearing more than once: {n_dupe_rows}")
    if n_dupe_rows > 0:
        print("[SCRAPE] Top duplicated URLs (count):")
        for url, cnt in dup_counts.head(10).items():
            if cnt > 1:
                print(f"  {cnt}  {url}")

    # Drop duplicate URLs so we scrape each URL once
    df = df.drop_duplicates(subset=["SOURCEURL"]).reset_index(drop=True)
    print(f"[SCRAPE] Rows after URL dedup (one row per URL): {len(df)}")

    # Sample up to configured number of unique URLs
    random.seed(SCRAPE_SEED)
    n_target = min(SCRAPE_N_UNIQUE_URLS, len(df))
    print(f"[SCRAPE] Requested unique URLs: {SCRAPE_N_UNIQUE_URLS}")
    print(f"[SCRAPE] Actually sampled for scraping: {n_target}")

    if n_target == 0:
        print("[SCRAPE] No URLs available to scrape")
        return

    df = df.sample(n=n_target, random_state=SCRAPE_SEED).reset_index(drop=True)

    session = make_session()

    ok_rows = []
    fail_rows = []

    for i, row in df.iterrows():
        url = row["SOURCEURL"]
        res = extract_article_text(session, url)

        base = {
            "GlobalEventID": str(row.get("GlobalEventID", "")),
            "SQLDATE": str(row.get("SQLDATE", "")),
            "SOURCEURL": url,
            "involves_rebel": bool(row.get("involves_rebel", False)),
        }

        out = {**base, **res}
        if res.get("ok"):
            ok_rows.append(out)
        else:
            fail_rows.append(out)

        print(
            f"[SCRAPE] {i+1}/{len(df)} "
            f"ok={res.get('ok')} "
            f"reason={res.get('reason')} "
            f"status={res.get('http_status', '')}"
        )

        time.sleep(SCRAPE_SLEEP_SECONDS)

    ok_df = pd.DataFrame(ok_rows)
    fail_df = pd.DataFrame(fail_rows)

    ok_path = ARTICLES_DIR / "articles_sample_100.csv"
    fail_path = ARTICLES_DIR / "articles_sample_100_failures.csv"

    ok_df.to_csv(ok_path, index=False)
    fail_df.to_csv(fail_path, index=False)

    print(f"[SCRAPE] Saved {len(ok_df)} scraped articles to {ok_path}")
    print(f"[SCRAPE] Saved {len(fail_df)} failures to {fail_path}")
    print(f"[SCRAPE] Total attempted URLs: {len(df)}")
    print(f"[SCRAPE] Success rate: {len(ok_df) / len(df):.3f}" if len(df) > 0 else "[SCRAPE] Success rate: n/a")


def main():
    start_time = time.time()
    print("Executing UCDP-augmented GDELT event fetch")

    actor_terms = load_actor_terms()
    rad_terms = _unique_terms(actor_terms, "RAD")
    ucdp_terms = _unique_terms(actor_terms, "UCDP_ACTOR")
    all_terms = sorted(set(rad_terms) | set(ucdp_terms))

    print(f"Actor terms loaded: {len(actor_terms)} rows")
    print(f"Unique RAD terms: {len(rad_terms)}")
    print(f"Unique UCDP actor terms: {len(ucdp_terms)}")
    print(f"BigQuery actor-name filter: {GDELT_USE_BQ_NAME_FILTER}")

    rad_pattern = build_pandas_pattern(rad_terms)
    ucdp_pattern = build_pandas_pattern(ucdp_terms)
    actor_name_bq_pattern = build_bq_pattern(all_terms)

    start_date, end_date = get_date_window()
    print(f"Date window {start_date} to {end_date}")

    client = make_bq_client()

    events = fetch_events_base(client, start_date, end_date, actor_name_bq_pattern)
    events = add_rebel_flags(events, rad_pattern, ucdp_pattern)

    summary_path = GDELT_OUT_DIR / "actor_matching_summary.txt"
    write_rebel_summary(actor_terms, events, summary_path)

    examples_path = GDELT_OUT_DIR / "actor_match_examples.csv"
    export_match_examples(events, examples_path, n_each=50)

    events_out_parquet = GDELT_OUT_DIR / "events_base.parquet"
    events_out_csv = GDELT_OUT_DIR / "events_base.csv"
    events.to_parquet(events_out_parquet, index=False)
    events.to_csv(events_out_csv, index=False)

    panel = build_country_week_panel(events)
    panel_out_parquet = GDELT_OUT_DIR / "country_week_panel.parquet"
    panel_out_csv = GDELT_OUT_DIR / "country_week_panel.csv"
    panel.to_parquet(panel_out_parquet, index=False)
    panel.to_csv(panel_out_csv, index=False)

    if SCRAPE_ARTICLES:
        scrape_sample_articles(events)


    if not SAMPLE_MODE:
        save_last_run_date(end_date)

    elapsed = time.time() - start_time
    print("Finished UCDP-augmented GDELT event fetch")
    print(f"Elapsed seconds {elapsed:.2f}")
    print(f"Events rows {len(events)}")
    print(f"Panel rows {len(panel)}")


if __name__ == "__main__":
    main()
