# Saved extraction report: exact target features

Rescored 10 November 2026. Same five shoe and five job conversations, same saved model outputs. No model or generation calls were made for this re-evaluation.

## Acceptance rule

- Match the target feature name, constraint operator and values exactly after lowercasing strings. Do not trim whitespace, substitute synonyms, infer equivalence across features or convert strings into numbers.
- Numeric bounds retain their operators: `price <= 100` does not match `price = 100`. Array values are unordered sets; duplicate values do not add credit.
- A missing target feature is an error. A present target feature with a missing/changed value or operator is an error.
- Additional features or values are warnings only. They never block acceptance. A replacement already classified as a changed target is not counted again as an extra warning.
- A turn passes when it has no errors, including turns that have warnings.
- Scope, strength, evidence, task/item state, natural-language intent and native output-validity checks remain separate diagnostics. They do not erase correctly extracted feature values or gate this feature/value check.

## Results on the same ten conversations

46 user turns. The player uses rollout: each model receives its own previous valid prediction. These existing input chains are unchanged; only scoring and display were revised.

| Model | Passed turns | Error turns | Warning turns | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Luna | 41/46 | 5 | 9 | 96.4% | 97.9% |
| Qwen 4B zero-shot | 18/46 | 28 | 13 | 80.3% | 80.7% |
| Qwen 4B fine-tuned | 22/46 | 24 | 3 | 97.1% | 84.0% |

Precision = matched target atoms / extracted atoms. Recall = matched target atoms / target atoms. Extra atoms reduce precision but do not make a turn fail. Error and warning counts are feature-level occurrences; precision and recall count individual operator/value atoms. Every available feature is scored even when a separate state-validity check fails; there is no blanket zeroing.

### Supplied previous correct state

| Model | Passed turns | Error turns | Warning turns | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Luna | 43/46 | 3 | 3 | 98.8% | 98.8% |
| Qwen 4B zero-shot | 24/46 | 22 | 5 | 95.3% | 83.5% |
| Qwen 4B fine-tuned | 31/46 | 15 | 1 | 98.7% | 91.4% |

## Luna: per-conversation outcomes in the player

| Conversation | Turns | Passed | Error turns | Warning turns |
| --- | ---: | ---: | ---: | ---: |
| sft-shoes-1 | 3 | 2 | 1 | 0 |
| sft-shoes-2 | 4 | 4 | 0 | 0 |
| sft-shoes-3 | 4 | 4 | 0 | 0 |
| sft-shoes-4 | 4 | 4 | 0 | 3 |
| sft-shoes-5 | 5 | 2 | 3 | 3 |
| sft-jobs-1 | 7 | 7 | 0 | 0 |
| sft-jobs-2 | 5 | 5 | 0 | 0 |
| sft-jobs-3 | 6 | 5 | 1 | 0 |
| sft-jobs-4 | 3 | 3 | 0 | 0 |
| sft-jobs-5 | 5 | 5 | 0 | 3 |

The feature `sport = basketball` does not satisfy a target `primary_use = Basketball`: the latter remains missing, and sport is an extra-feature warning if absent from the target. A supported extra category such as `Trail running shoes` is still an extra warning, and passes when all target values are present. No semantic reinterpretation is used.

## Experiment and sampling

- The ten conversations were sampled from held-out test data with seed 20261110, five per vertical, independently of model outcomes. They are for inspection, not a replacement for the full test set.
- Qwen base model: Qwen/Qwen3.5-4B, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Fine-tuning used 800 training and 100 validation conversations, one epoch, 256 optimizer steps; LoRA rank 16, alpha 32, dropout 0.05, learning rate 0.0001, accumulation 16, BF16, completion-only loss and gradient checkpointing. Checkpoint 256 was selected on validation loss.
- Qwen zero-shot and fine-tuned outputs retain the same extraction prompt, schema, tokenizer and serving configuration; temperature 0, a seed schedule starting at 7, 4096 output tokens, thinking disabled. Supplied-state input hashes match.
- Luna was evaluated on these ten only, using the same logical extraction prompt and schema. Its OpenAI Responses API uses provider-default decoding; provider/runtime/sampling controls differ from Qwen.
- The previous full-100 test figures used the older validity-gated scorer. They must not be compared directly with the newly rescored figures in this report. This revision rescored the ten saved comparison conversations only.
- Targets remain procedural labels from automated data generation. This score measures exact target feature fidelity, not an independent assessment of meaning.

## Player

Choose **Play pre-extracted**, select a shoes or jobs conversation, and enable **Compare side by side**. Select the two models independently. Full-trace pass/error/warning counts and precision/recall appear at the top of each extractor panel; links jump to error or warning turns. Each extracted feature has a compact title with its expected/extracted values and evidence collapsed underneath. Assistant product and job listings are also collapsed individually. Playback and model switching make no API calls. Live extraction remains Luna-only.

## Footwear schema v2

Generic activity names now have one canonical field: `sport`. The duplicate `primary_use` values Road running, Trail running, Basketball, Football, Tennis, Volleyball, Sprinting, Lifestyle and Hiking have been removed. Specific uses such as Marathon racing, Easy runs, Technical trail running and Weightlifting remain in `primary_use`. Catalogs no longer repeat generic activities in both fields.

The ten saved traces above were extracted with schema v1. Their raw outputs, target labels and scores retain that version for reproducibility; they have not been rerun with v2, and no score-time alias was introduced. The player labels historical shoe traces accordingly. New live conversations and newly generated footwear scenarios use v2. Re-extraction of historical saved conversations retains their original embedded schema.

## Five additional job conversations

Added Jobs 6–10 from the same held-out test pool, excluding all original examples, sampled with seed 20261111. All three models use the original extraction prompt and embedded job schema. Qwen zero-shot and LoRA predictions come from the frozen test run; Luna was newly evaluated on these five in both rollout and supplied-correct-state modes. Original ten traces are unchanged. No pending footwear schema changes are included.

| Model | Passed turns | Error turns | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Luna | 28/30 | 2 | 98.9% | 98.4% |
| Qwen 4B zero-shot | 19/30 | 11 | 87.0% | 82.8% |
| Qwen 4B fine-tuned | 19/30 | 11 | 98.2% | 89.2% |
