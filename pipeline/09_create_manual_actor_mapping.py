from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT = REPO_ROOT / "data/reports/strict_rebel_actor_group_metadata.csv"
OUT = REPO_ROOT / "data/reports/proposed_rebel_actor_manual_mapping.csv"
REPORT = REPO_ROOT / "data/reports/proposed_rebel_actor_manual_mapping_summary.txt"


PROPOSALS = {
    "HAMAS": ("209", "Hamas", "Israel/Palestine", "Middle East", "single_exact", "Single unambiguous alias match."),
    "TALIBAN": ("8079", "Taliban", "Afghanistan", "South/Central Asia", "merged_spelling_variant", "Taleban and Taliban are spelling/name variants; use Taliban as canonical. Haiti metadata on ActorId 8079 treated as source artifact and removed from derived metadata."),
    "FARC": ("743", "FARC umbrella", "Colombia", "Latin America", "umbrella_group", "Alias maps to FARC and successor/splinter groups; use FARC umbrella for country/month aggregation."),
    "KURDISTAN WORKERS PARTY": ("323", "PKK", "Turkey/Iraq", "Middle East", "single_exact", "Matched long name of PKK."),
    "JANJAWEED": ("630", "Janjaweed umbrella", "Sudan", "Africa", "umbrella_group", "Alias maps to Janjaweed factions; use Janjaweed umbrella."),
    "KHMER ROUGE": ("270", "Khmer Rouge", "Cambodia", "Asia", "single_exact", "Single unambiguous alias match."),
    "BALOCHISTAN LIBERATION ARMY": ("287", "BLA", "Pakistan", "South/Central Asia", "single_exact", "Single unambiguous alias match."),
    "MILF": ("276", "MILF", "Philippines", "Asia", "manual_prefer_main_group", "Alias also maps to BIFM as previous name; prefer MILF for this actor label."),
    "AL QAEDA": ("769", "al-Qaida", "Transnational", "Transnational", "single_exact_transnational", "Transnational actor; event location should drive country/month aggregation."),
    "AL QAIDA": ("769", "al-Qaida", "Transnational", "Transnational", "spelling_variant", "Spelling variant of al-Qaida."),
    "LIBERATION TIGERS OF TAMIL EELAM": ("320", "LTTE", "Sri Lanka", "Asia", "single_exact", "Single unambiguous alias match."),
    "VISHWA HINDU PARISHAD": ("397", "VHP", "India", "Asia", "single_exact_review", "Matched UCDP actor; verify whether this should be included as rebel/non-state actor for your design."),
    "LASHKAR E TAIBA": ("1070", "Lashkar-e-Taiba", "Pakistan/India", "South/Central Asia", "single_exact", "Single unambiguous alias match."),
    "RENAMO": ("498", "Renamo", "Mozambique", "Africa", "single_exact", "Single unambiguous alias match."),
    "PALESTINE LIBERATION ORGANIZATION": ("204", "PLO", "Israel/Palestine", "Middle East", "single_exact_review", "Historical actor; verify inclusion for your study period."),
    "JKLF": ("817", "JKLF", "India/Pakistan", "South/Central Asia", "single_exact", "Single unambiguous alias match."),
    "MORO ISLAMIC LIBERATION FRONT": ("276", "MILF", "Philippines", "Asia", "single_exact", "Long-name alias for MILF."),
    "FRELIMO": ("428", "Frelimo", "Mozambique", "Africa", "single_exact_review", "Matched UCDP actor; verify whether government-party status in study period matters."),
    "MNLF": ("275", "MNLF umbrella", "Philippines", "Asia", "umbrella_group", "Alias maps to MNLF and factions; use MNLF umbrella."),
    "MORO NATIONAL LIBERATION FRONT": ("275", "MNLF", "Philippines", "Asia", "single_exact", "Long-name alias for MNLF."),
    "SUDAN LIBERATION MOVEMENT": ("469", "SLM/A umbrella", "Sudan", "Africa", "umbrella_group", "Alias maps to SLM/A factions; use umbrella."),
    "ISLAMIC FRONT": ("5575", "Islamic Front umbrella", "Syria", "Middle East", "umbrella_group", "Alliance label; use Islamic Front umbrella or review if you want faction-level assignment."),
    "REVOLUTIONARY UNITED FRONT": ("532", "RUF", "Sierra Leone", "Africa", "single_exact", "Single unambiguous alias match."),
    "OROMO LIBERATION FRONT": ("551", "OLF", "Ethiopia", "Africa", "single_exact", "Single unambiguous alias match."),
    "NATIONAL PATRIOTIC FRONT OF LIBERIA": ("507", "NPFL", "Liberia", "Africa", "single_exact", "Single unambiguous alias match."),
    "INDEPENDENT NATIONAL PATRIOTIC FRONT OF LIBERIA": ("508", "INPFL", "Liberia", "Africa", "single_exact", "Single unambiguous alias match."),
    "INPFL": ("508", "INPFL", "Liberia", "Africa", "single_exact", "Short-name alias for INPFL."),
    "UNITED LIBERATION FRONT OF ASSAM": ("326", "ULFA", "India", "Asia", "single_exact", "Single unambiguous alias match."),
    "ULIMO": ("697", "ULIMO umbrella", "Liberia/Sierra Leone", "Africa", "umbrella_group", "Alias maps to ULIMO and J/K factions; use umbrella."),
    "ERITREAN LIBERATION FRONT": ("415", "ELF", "Eritrea/Ethiopia", "Africa", "single_exact", "Single unambiguous alias match."),
    "LORDS RESISTANCE ARMY": ("488", "LRA", "Uganda/Central Africa", "Africa", "single_exact", "Single unambiguous alias match."),
    "LIBERIANS UNITED FOR RECONCILIATION AND DEMOCRACY": ("509", "LURD", "Liberia", "Africa", "single_exact", "Single unambiguous alias match."),
    "MOZAMBICAN NATIONAL RESISTANCE": ("498", "Renamo", "Mozambique", "Africa", "single_exact", "Long-name alias for Renamo."),
    "UGANDA NATIONAL LIBERATION FRONT": ("479", "UNLF", "Uganda", "Africa", "single_exact", "Single unambiguous alias match."),
    "UNITED FRONT FOR DEMOCRATIC CHANGE": ("453", "FUCD", "Chad", "Africa", "single_exact", "Single unambiguous alias match."),
}


def main() -> None:
    df = pd.read_csv(INPUT)
    rows = []
    for _, row in df.iterrows():
        actor = row["matched_gdelt_actor"]
        proposal = PROPOSALS.get(actor, ("", "", "", "", "needs_review", "No proposal available."))
        rows.append(
            {
                "matched_gdelt_actor": actor,
                "unique_events": row.get("unique_events"),
                "top_event_country": row.get("top_event_country"),
                "possible_rebel_ids": row.get("possible_rebel_ids"),
                "possible_rebel_names": row.get("possible_rebel_names"),
                "metadata_locations": row.get("metadata_locations"),
                "metadata_regions": row.get("metadata_regions"),
                "proposed_rebel_id": proposal[0],
                "proposed_rebel_name": proposal[1],
                "proposed_home_country_or_area": proposal[2],
                "proposed_macro_region": proposal[3],
                "assignment_type": proposal[4],
                "assignment_note": proposal[5],
                "user_confirmed": "",
                "user_override_rebel_id": "",
                "user_override_rebel_name": "",
                "user_override_home_country_or_area": "",
                "user_note": "",
            }
        )

    out = pd.DataFrame(rows).sort_values("unique_events", ascending=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    lines = [
        "Proposed rebel actor manual mapping",
        f"Rows: {len(out)}",
        f"Output: {OUT}",
        "",
        out[
            [
                "matched_gdelt_actor",
                "unique_events",
                "possible_rebel_names",
                "proposed_rebel_name",
                "proposed_home_country_or_area",
                "assignment_type",
            ]
        ].to_string(index=False),
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
