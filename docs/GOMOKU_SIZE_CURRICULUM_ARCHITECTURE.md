# Gomoku Size Curriculum Architecture

## Goal

Establish a unified Gomoku size-curriculum architecture where:

- `gomoku9_curriculum` is a fast first-class curriculum lane;
- `gomoku11`, `gomoku13`, `gomoku15`, `gomoku16` continue through the same neural contract;
- serving safety prevents curriculum artifacts from leaking into production variants;
- training, replay, evaluation, and checkpoint metadata become explicitly size-aware.

This document records the minimal-safe phase adapted from [`michaelnny/alpha_zero`](https://github.com/michaelnny/alpha_zero).

## Audit Summary

- The repo already had the right neural contract for a unified size family:
  - input planes are encoded as `[6, 16, 16]`
  - policy targets are padded to `[256]`
  - augmentation preserves top-left geometry
- The main missing piece was not a new model family; it was a shared size/spec abstraction and safe metadata flow.
- The biggest size-related risks were:
  - scattered board-size parsing
  - implicit large-board defaults in trainer-lab helpers
  - replay visibility that ignored board size and curriculum stage
  - registry/serving logic that relied mostly on variant path names rather than structured compatibility metadata

## Final Architecture Decision

- `[ADAPT]` Keep one padded neural contract for `9..16`:
  - `padded_size = 16`
  - `policy_size = 256`
  - top-left placement stays unchanged in the minimal-safe phase
- `[ADAPT]` Introduce a shared `VariantSpec` as the runtime/training source of truth.
- `[ADAPT]` Represent curriculum lanes as explicit variants such as `gomoku9_curriculum`, not as hidden flags on production variants.
- `[LOCAL]` Preserve champion-only serving and guard it with structured variant compatibility checks.
- `[LOCAL]` Keep teacher/tactical/user/failure/anchor sources intact; do not replace them with pure upstream uniform self-play replay.

## Upstream Mapping Table

| Proposed change | Classification | Upstream source file | Current repo target file(s) | Why it fits | Why it must be adapted or preserved |
|---|---|---|---|---|---|
| Shared game/variant spec as the center of training wiring | `[ADAPT]` | `alpha_zero/training_gomoku.py`, `alpha_zero/core/pipeline.py` | `trainer-lab/src/trainer_lab/specs.py`, `apps/api/src/gomoku_api/ws/train_service_ws.py`, `apps/api/src/gomoku_api/ws/predict_service.py` | Upstream builds training around explicit env/network wiring per game | Local repo already has product variants, serving modes, and a padded `16x16/256` contract that must remain stable |
| Replay as the center of learning | `[UPSTREAM]` | `alpha_zero/core/replay.py` | `trainer-lab/src/trainer_lab/self_play/mixed_replay.py` | Upstream correctly centers learning around replay state and resume behavior | Local replay must remain mixed-source rather than uniform self-play only |
| Size-aware replay summaries | `[ADAPT]` | `alpha_zero/core/replay.py` | `trainer-lab/src/trainer_lab/self_play/mixed_replay.py` | Upstream replay is already the canonical state container | Local repo needs explicit visibility into both source mix and board-size mix |
| Process/role separation as the target topology | `[UPSTREAM]` | `alpha_zero/training_gomoku.py`, `alpha_zero/core/pipeline.py` | Existing worker/evaluator seams in `apps/api/src/gomoku_api/ws/` and `trainer-lab/src/trainer_lab/self_play/` | Upstream actor/learner/evaluator separation is the correct blueprint | Minimal-safe phase only stabilizes interfaces and metadata; it does not rewrite the full training host again |
| MCTS visit-count targets and deterministic evaluator semantics | `[UPSTREAM]` | `alpha_zero/core/mcts_v2.py`, `alpha_zero/core/pipeline.py` | Existing `trainer-lab/src/trainer_lab/self_play/player.py`, `trainer-lab/src/trainer_lab/evaluation/eval_script.py` | Upstream search discipline is already the right learning target | Local runtime still needs `pure/hybrid/mcts` separation and tactical safety layers |
| First-class 9x9 curriculum profile under the same model family | `[ADAPT]` | `alpha_zero/core/network.py`, `alpha_zero/training_gomoku.py` | `apps/api/src/gomoku_api/ws/model_profiles.py`, `apps/api/src/gomoku_api/ws/train_service_ws.py` | Upstream trains one Gomoku family and scales by board configuration | Local repo must isolate curriculum variants from production serving while preserving weight carry-over |
| Structured checkpoint compatibility metadata | `[LOCAL]` | none | `apps/api/src/gomoku_api/ws/model_registry.py`, `apps/api/src/gomoku_api/ws/train_service_ws.py`, `apps/api/src/gomoku_api/ws/predict_service.py` | Local registry already owns lifecycle and manifest history | Upstream does not need cross-variant serving guardrails because it does not multiplex product variants this way |
| Keep padded geometry unchanged in minimal-safe phase | `[LOCAL]` | none | `trainer-lab/src/trainer_lab/data/encoder.py`, `trainer-lab/src/trainer_lab/data/augmentation.py`, `trainer-lab/src/trainer_lab/data/policy.py` | Stability is more important than geometry experiments in this phase | Changing offset/padding would require a migration plan and new validation packs |
| Evaluate Gomoku-specific first-conv padding later, not now | `[ADAPT]` | `alpha_zero/core/network.py` | `trainer-lab/src/trainer_lab/models/resnet.py` | Upstream’s larger first-layer padding is a real idea worth testing | Local repo cannot safely adopt it in the same phase as size-aware lifecycle and curriculum metadata |

## Minimal-Safe Phase Boundaries

Implemented now:

1. unified `VariantSpec`
2. removal of dangerous scattered size defaults where safe
3. explicit `gomoku9_curriculum` model/training profile
4. size-aware replay/checkpoint/manifest/eval metadata
5. curriculum-vs-production separation by variant identity and serving guardrails
6. targeted tests

Not implemented yet:

- full upstream multi-process actor farm for Gomoku sizes
- board-size mixing policy inside learner sampling beyond visibility/summary
- network geometry changes
- production rollout defaults for `11x11+` self-play

## Neural Contract

- `[LOCAL]` Input remains `[B, 6, 16, 16]`
- `[LOCAL]` Policy head remains `[B, 256]`
- `[LOCAL]` Top-left padding remains the canonical board placement
- `[ADAPT]` Curriculum and production variants share the same contract so weight carry-over stays possible

## Safety Rules

- `[LOCAL]` Curriculum variants are stored under distinct variant IDs, for example `gomoku9_curriculum`
- `[LOCAL]` Serving resolves only promoted checkpoints and now rejects structured variant mismatches
- `[LOCAL]` `candidate_working -> candidate -> champion` lifecycle stays intact
- `[LOCAL]` `pure`, `hybrid`, and `mcts` runtime modes remain unchanged

## Phase-2 Roadmap

1. Port more upstream actor/learner/evaluator wiring into dedicated Gomoku worker roles.
2. Make learner sampling explicitly board-size aware, not only source aware.
3. Add curriculum-to-production transfer workflow with explicit resume-from-checkpoint semantics.
4. Add evaluator bundles separated by board size and promotion scope.
5. Evaluate upstream Gomoku-specific first-conv padding behind a separate benchmark gate.
