# Taliban pre/post-takeover diagnostic

- Exact article cutoff: 2021-08-15
- Pre-takeover articles: 9,427
- Post-takeover articles: 14,214
- Monthly comparison: 19 pre months and 52 post months; August 2021 excluded.

## Monthly level shifts

- log(1 + articles): pre mean 6.056, post mean 5.232; HAC difference -0.824 (p=4.635e-05); trend/season-adjusted difference 0.026 (p=0.9414).
- log(1 + next-month fatalities): pre mean 7.741, post mean 3.199; HAC difference -4.542 (p=4.207e-44); trend/season-adjusted difference -3.310 (p=1.71e-10).

## Largest article-level positive-share changes

- Combat context: 52.6% to 34.9% (-17.7 percentage points).
- Physical violence occurred: 58.7% to 43.4% (-15.2 percentage points).
- Direct conflict relevance: 84.1% to 69.7% (-14.4 percentage points).
- Actor as perpetrator: 55.0% to 41.7% (-13.3 percentage points).
- Main-story fatalities: 48.3% to 35.2% (-13.1 percentage points).
- Current or recent event: 80.0% to 67.5% (-12.6 percentage points).
- Violence mentioned anywhere: 88.2% to 77.9% (-10.3 percentage points).
- Main-story injuries: 31.5% to 21.4% (-10.1 percentage points).

## Multiclass distribution shifts

- event_time_relation: Jensen-Shannon distance 0.177; largest category change 14.4 percentage points.
- event_context: Jensen-Shannon distance 0.167; largest category change 17.7 percentage points.
- conflict_relevance: Jensen-Shannon distance 0.157; largest category change 14.4 percentage points.
- matched_actor_role: Jensen-Shannon distance 0.139; largest category change 13.3 percentage points.
- event_modality: Jensen-Shannon distance 0.045; largest category change 2.5 percentage points.

HAC p-values describe a known pre/post level break and are not causal tests. The adjusted specification includes a linear trend and annual sine/cosine terms. Article-level significance tests are intentionally omitted because the very large article count would make substantively small changes appear highly significant.
