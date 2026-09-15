# Prompt development notes

The article-labeling prompt was developed in two evaluated versions before the production run. These notes document both versions and the changes between them. The final prompt files in this directory are exact production artifacts. The initial prompt was edited in place and was not retained as a separate full-text snapshot.

## Version 1: initial schema prompt

The initial version translated the manual annotation schema into an LLM system prompt. It:

- required one JSON object with the complete annotation schema;
- separated main-story labels from article-wide mention labels;
- defined the event-context, actor-role, violence, fatality, injury, weapon, and diagnostic categories;
- included consistency rules and examples covering protest, policy, armed-clash, advocacy, and historical-background articles; and
- used the same article-specific metadata fields as the subsequent production prompt.

The first diagnostic evaluation used 28 manually labeled articles. Article-wide mention detection was comparatively reliable, but labels requiring interpretation of the main story were less consistent. In particular, the model did not use the `irrelevant` category for any of the six manually irrelevant cases. The evaluation also exposed ambiguity in the treatment of approximate casualty counts and generic weapon references.

## Version 2: finalized production prompt

Version 2 retained the same overall schema but made the following rules more explicit:

- An incidental matched-actor reference does not by itself make the main story conflict-relevant.
- An explicit irrelevant-article example shows how main-story fields become `not_applicable` while article-wide mentions remain independently codable.
- Numeric casualty fields accept exact counts only; approximations, ranges, bounds, percentages, and vague quantities are excluded.
- Article-wide casualty counts mechanically record every exact count instance rather than attempting to infer a deduplicated event total.
- A positive weapon-mention label requires a specific weapon, weapon system, or means of attack; generic references to arms, weapons, violence, or disarmament are insufficient.
- The output includes an article-language diagnostic field.

The revised version was evaluated on an expanded set of 40 manually labeled articles. It was then fixed before the production run and used for all released article labels. Remaining disagreement was evaluated using the larger manual-validation sample rather than through further prompt optimization.

The exact finalized files are:

- `system_prompt.txt`
- `user_prompt.txt`

## Archival limitation

Because Version 1 was revised in place, its exact complete prompt text cannot be reproduced from the archived files. The description above is based on the contemporaneous evaluation and change notes. This limitation is stated explicitly to avoid presenting a reconstructed prompt as the original artifact.
