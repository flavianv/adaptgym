# Qwen 4B intent extraction: zero-shot and LoRA SFT

Completed 10 November 2026. All evaluations finished successfully.

## Experiment

- Base model: `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Data: 1,000 synthetic conversations, balanced between shoes and jobs; 800 train / 100 validation / 100 test. Turn-expanded rows: 4,082 / 507 / 522.
- Training: one epoch, 256 optimizer steps; ordinary LoRA rank 16, alpha 32, dropout 0.05; learning rate 0.0001, gradient accumulation 16, microbatch 1, BF16, completion-only loss and gradient checkpointing. 32,464,896 trainable parameters.
- Checkpoint selection uses validation loss only. The selected checkpoint is step 256. Test results were not used for checkpoint selection.
- Base and tuned models use the same frozen system prompt, output schema, tokenizer/chat template, serving configuration, temperature 0, seed schedule starting at 7, 4,096 output-token cap and no thinking.
- All 522 test turns are evaluated twice: with supplied previous correct state, and in rollout with each model carrying its own previous valid prediction. Teacher-forced request hashes match before and after training.
- No input truncation. Longest training sequence: 22,385 tokens. The full forward/backward smoke passed with finite gradients and 20.93 GiB peak allocated GPU memory.

## Full test results: 100 conversations, 522 turns per mode

| Mode | Model | Valid outputs | Value precision* | Value recall | Complete value coverage | Exact structured state |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Supplied previous state | Qwen 4B zero-shot | 313/522 (60.0%) | 98.3% | 44.2% | 29.9% | 12.3% |
| Supplied previous state | Qwen 4B LoRA | 379/522 (72.6%) | 99.6% | 66.1% | 52.5% | 27.2% |
| Own previous prediction | Qwen 4B zero-shot | 239/522 (45.8%) | 84.9% | 34.9% | 22.8% | 4.4% |
| Own previous prediction | Qwen 4B LoRA | 341/522 (65.3%) | 98.0% | 53.3% | 30.7% | 13.2% |

*Precision counts value atoms from state-valid outputs. Invalid outputs contribute all expected values as misses to recall and fail complete coverage / exact state; their predicted atoms are excluded from precision. Precision therefore needs to be read together with validity and recall.

LoRA improves recall by 21.9 percentage points with supplied state and by 18.4 points in rollout. Exact structured-state agreement rises from 12.3% to 27.2% and from 4.4% to 13.2%, respectively. This is an uplift, but remaining state errors and invalid outputs are substantial.

## Ten precomputed comparison conversations

Five shoes and five jobs were sampled from the held-out test set using seed `20261110`, independently of model outcomes. They contain 46 user turns. The same exact conversations are shown for all three extractors; the Qwen outputs are reused from the full frozen test run. Luna was newly evaluated on these ten only, with the same logical extraction prompt and schema, in both context modes. Luna uses OpenAI Responses with provider-default decoding; its provider, sampling controls and runtime differ from Qwen.

| Context mode | Model | Valid outputs | Value precision* | Value recall | Exact structured state |
| --- | --- | ---: | ---: | ---: | ---: |
| Supplied previous state | Luna | 46/46 | 98.8% | 98.8% | 89.1% |
| Supplied previous state | Qwen 4B zero-shot | 24/46 | 96.1% | 30.5% | 8.7% |
| Supplied previous state | Qwen 4B LoRA | 28/46 | 100.0% | 54.3% | 23.9% |
| Own previous prediction | Luna | 46/46 | 96.4% | 97.9% | 65.2% |
| Own previous prediction | Qwen 4B zero-shot | 21/46 | 83.3% | 39.1% | 2.2% |
| Own previous prediction | Qwen 4B LoRA | 26/46 | 98.1% | 42.0% | 10.9% |

The ten-case subset is for trace inspection. It is too small to replace the full test comparison; Luna has not been evaluated on all 100 test conversations.

| Saved conversation | Vertical | User turns |
| --- | --- | ---: |
| sft-shoes-1 | footwear | 3 |
| sft-shoes-2 | footwear | 4 |
| sft-shoes-3 | footwear | 4 |
| sft-shoes-4 | footwear | 4 |
| sft-shoes-5 | footwear | 5 |
| sft-jobs-1 | job_search | 7 |
| sft-jobs-2 | job_search | 5 |
| sft-jobs-3 | job_search | 6 |
| sft-jobs-4 | job_search | 3 |
| sft-jobs-5 | job_search | 5 |

## Using the conversation demo

Open **Live chat**, then choose **Play pre-extracted**. Choose a shoes or jobs conversation, then enable **Compare side by side**. The left and right dropdowns independently select Luna, Qwen 4B zero-shot or Qwen 4B fine-tuned. Play, pause, previous and next keep both panels on the same turn. The player uses rollout traces: each model receives its own prior valid prediction. Saved playback makes no model calls. Live extraction remains Luna-only.

Each bundle contains the original demo sample shape (`plan`, `result.conversation`, reference checkpoints), and three models with native saved replay records (`completed`, `checkpoints`, `profile`, `reference_profile`, `score`, `metrics`, evidence and prompt). Both supplied-state and rollout results are retained in the downloadable JSON.

## Limits and interpretation

- Reference labels come from procedural states and an automated generation gate, not a complete human audit. The gate permits invented additions when intended meaning is preserved. Some extra extracted values can therefore be grounded in text but absent from the reference.
- Value metrics match case-insensitive attribute/value atoms; arrays are sets. They do not establish correctness of operators, scope, strength, evidence, task status, selections or search constraints. Exact structured state adds these deterministic checks, while free-text intent and evidence semantics are not independently judged here.
- Invalidity often comes from item comparisons or selections without displayed-item/user evidence, inconsistent shortlists, or task/scope metadata. These are retained in the traces rather than removed.
- Training loss is not an extraction accuracy metric. The final reported training runtime covers the last resumed segment, not necessarily the entire epoch; the completion claim rests on step 256, epoch 1 and the saved checkpoint.
- Identical teacher-forced inputs support the base-versus-SFT comparison. Rollout inputs intentionally diverge as model predictions differ.
