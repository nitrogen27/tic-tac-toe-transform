# AlphaZero Integration Plan For TTT5 / Gomoku Stack

## Goal

Take the strongest reusable parts of [`michaelnny/alpha_zero`](https://github.com/michaelnny/alpha_zero) and integrate them deeply into this repository without throwing away the product features that are already valuable here:

- champion-only serving
- hybrid runtime with tactical safety and exact late search
- user-game relabel corpus
- exact/user-trap validation
- UI-driven training control and diagnostics

The target is not "rewrite everything into another repo". The target is:

1. adopt the proven AlphaZero training topology,
2. preserve the current product and evaluation strengths,
3. create a stronger and more stable training system for `ttt5`,
4. keep `pure` and `hybrid` measured separately.


## Executive Summary

`michaelnny/alpha_zero` is most valuable to this project as a **training architecture reference**, not as a drop-in product runtime.

What it already does well:

- explicit separation of `self-play actors`, `learner`, `evaluator`, and `replay`
- checkpoint-driven training progression
- evaluator that compares latest checkpoint against previous checkpoint
- replay-first learning instead of ad-hoc fine-tuning
- MCTS visit-count policy targets
- large-scale parallel self-play structure
- clean driver-level hyperparameter organization

What this repository already does better:

- product-facing runtime modes: `pure`, `hybrid`, `mcts`
- tactical safety layer and exact late-game search in serving
- champion/candidate/working-candidate lifecycle
- user-game post-hoc analysis and corpus building
- exact validation packs and product-centric diagnostics
- engine-backed teacher contour

Therefore the best integration strategy is:

- **import the AlphaZero loop**
- **do not throw away the existing teacher/hybrid/product contour**
- **run both contours side-by-side**
- **let product metrics decide what gets promoted**


## Current Project Analysis

### Current strengths

The current repository already has strong product and lifecycle pieces:

- `apps/api/src/gomoku_api/ws/model_registry.py`
  - separate `candidate_working.pt`, `candidate.pt`, `champion.pt`
  - serving resolves through verified checkpoint, not arbitrary working state
- `apps/api/src/gomoku_api/ws/predict_service.py`
  - multiple runtime paths: `pure`, `hybrid`, `mcts`
  - tactical forced win/block logic
  - exact late-game search
- `apps/api/src/gomoku_api/ws/user_game_corpus.py`
  - post-hoc analysis of user games
  - teacher-backed relabel
  - recent / mistake / conversion / weak-side buckets
- `apps/api/src/gomoku_api/ws/train_service_ws.py`
  - staged trainer
  - quick exam / confirm exam
  - frozen tactical suites
  - exact pack support
  - checkpoint selection and promotion gate
- `trainer-lab`
  - already contains a lightweight self-play implementation
  - already has replay, self-play player, mini bench, trainer modules

### Current weaknesses

The main bottlenecks are not "missing MCTS code". They are:

- training remains partly split between product contour and trainer contour
- self-play is present but not yet the dominant, stable data source
- frozen metrics still overestimate live strength
- user-found trap coverage is incomplete
- `pure` strength still lags `hybrid`
- `ttt5` side-balance and conversion remain unstable
- replay is too light compared to a full AlphaZero learner pipeline


## External Repo Analysis: `michaelnny/alpha_zero`

### Core architecture worth importing

From the upstream code:

- `alpha_zero/core/pipeline.py`
  - explicit `run_selfplay_actor_loop`
  - explicit `run_learner_loop`
  - explicit `run_evaluator_loop`
  - MCTS player factory
  - checkpoint/event synchronization between actors and learner
- `alpha_zero/core/replay.py`
  - replay as the center of learning
  - fixed-size circular buffer
  - support for serialization / restoration
- `alpha_zero/core/rating.py`
  - Elo-based checkpoint comparison
- `alpha_zero/training_gomoku.py`
  - driver that wires:
    - env builder
    - network builder
    - replay
    - actor processes
    - learner
    - evaluator
- `alpha_zero/core/network.py`
  - standard AlphaZero policy-value ResNet
  - notable Gomoku-specific fix: bigger padding on first conv for edge cases
- `alpha_zero/core/mcts_v2.py`
  - proper search policy from visit counts
  - tree reuse
  - optional batched parallel leaf evaluation

### Why it is a strong fit

It is a good fit because our project already wants:

- stronger self-play generation
- better replay discipline
- challenger-vs-serving comparison
- less reliance on one-shot fine-tune
- clearer separation of training roles

### What does **not** fit directly

It is **not** a drop-in replacement for this project because:

- it assumes a purer AlphaZero world than our product currently uses
- it does not know about `pure` vs `hybrid`
- it does not know about user-game relabel pipelines
- it does not know about exact trap packs and product-specific tactical rescue metrics
- it does not know about our serving lifecycle requirements
- its evaluator is checkpoint-vs-previous-checkpoint, not product-vs-engine/product-vs-user-trap-suite


## Mapping: Current Repo vs AlphaZero Repo

### Role mapping

| AlphaZero upstream | Current repo equivalent | Gap |
|---|---|---|
| self-play actor loop | `trainer-lab/self_play/player.py`, optional self-play inside `train_service_ws.py` | lacks process-level actor architecture and cleaner isolation |
| learner loop | `_run_training_steps()` + `train_variant()` | too monolithic; replay is not yet the primary center |
| evaluator loop | `_run_engine_exam()` + confirm exam + frozen suites | stronger on product metrics, weaker on checkpoint-vs-checkpoint discipline |
| replay | `trainer-lab/self_play/replay_buffer.py`, position banks in `train_service_ws.py`, user corpus | fragmented across multiple stores |
| Elo/rating | promotion metrics + engine winrate + side metrics | no clean rating layer between self-play checkpoints |
| self-play visit targets | optional MCTS/self-play branch | not the default, not deeply integrated into model selection |

### What we should preserve from current repo

Do not discard these:

- `model_registry.py` lifecycle
- `predict_service.py` hybrid runtime
- `user_game_corpus.py`
- exact/user-trap validation in `train_service_ws.py`
- product metrics and UI telemetry
- current engine-backed teacher contour

### What we should import conceptually from AlphaZero

- actor/learner/evaluator separation
- checkpoint/event coordination
- replay-first training
- visit-count policy targets as first-class training data
- explicit challenger evaluation loop
- persistent replay restore/resume behavior


## Integration Strategy

This should be a **dense integration**, not a side experiment and not a rewrite.

The integration should produce a training system with **two data contours**:

1. `teacher/product contour`
   - engine teacher
   - tactical curriculum
   - user-game relabel corpus
   - exact/user-trap validation

2. `alpha-zero contour`
   - MCTS self-play actors
   - replay buffer
   - visit-count targets
   - checkpoint-vs-checkpoint evaluator

Then the learner should train on a controlled mixture of both.


## Target End State

### Serving

- runtime serves only `champion.pt`
- `candidate` and `working candidate` never leak into live serving
- serving supports:
  - `pure`
  - `hybrid`
  - optional `mcts`

### Training

Single orchestrated trainer:

1. load serving champion as base
2. generate self-play games with actors
3. collect visit-count targets into self-play replay
4. collect teacher/tactical/user positions into product replay
5. train learner from mixed replay
6. run evaluator:
   - challenger vs champion
   - challenger vs engine
   - challenger on exact/user-trap suite
   - pure/hybrid split metrics
7. only promote if product metrics improve

### Validation

Promotion should depend on:

- hybrid confirm winrate
- balanced side winrate
- decisive winrate
- exact trap recall
- user-trap recall
- pure/hybrid gap not regressing beyond threshold


## Phased Implementation Plan

## Phase 0. Stabilize Before Import

Goal:
make the current repository a safe host for AlphaZero-style training.

Tasks:

1. keep `champion-only serving`
2. keep separate dashboards for:
   - `pure`
   - `hybrid`
   - `exact/user-trap`
3. treat current `train_service_ws.py` as the host orchestrator
4. keep current `user_game_corpus.py` and exact packs as product truth

Exit criteria:

- live serving remains stable
- no self-play work can accidentally degrade runtime without promotion


## Phase 1. Introduce a Real AlphaZero Self-Play Subsystem

Goal:
replace the current lightweight self-play loop with a stronger subsystem modeled after upstream actor/replay discipline.

New modules to introduce:

- `trainer-lab/src/trainer_lab/self_play/actor_loop.py`
- `trainer-lab/src/trainer_lab/self_play/learner_bridge.py`
- `trainer-lab/src/trainer_lab/self_play/evaluator_bridge.py`
- `trainer-lab/src/trainer_lab/self_play/replay_state.py`

What to port conceptually from upstream:

- actor process loop
- checkpoint handoff
- root-noise self-play
- replay restore/resume
- evaluator-on-new-checkpoint

What to adapt for this repo:

- board encoding must stay compatible with our `board_to_tensor`
- `ttt5` requires `win_len = 4`
- policy targets must remain padded to `16x16` (`256`) because current model path expects that
- actor-generated positions must include `source: "self_play"`

Do not yet:

- remove teacher data
- let self-play become the only source of truth

Exit criteria:

- self-play can run independently
- replay buffer persists and resumes
- actors can generate usable visit-target positions for `ttt5`


## Phase 2. Replace the Lightweight Replay With a Real Mixed Replay System

Goal:
make replay the center of training, but with multiple sources.

Create a replay manager that combines:

1. `self_play_replay`
2. `teacher_replay`
3. `tactical_replay`
4. `user_corpus_replay`
5. `failure_bank_replay`

Recommended new module:

- `trainer-lab/src/trainer_lab/self_play/mixed_replay.py`

Required features:

- fixed-size ring buffers
- per-source quotas
- weighted sampling
- dedupe / canonicalization for small-board redundancy
- save/load state

Sampling policy for early rollout:

- 35% teacher/tactical
- 25% self-play
- 20% user/failure
- 20% anchor/historical stable positions

Later, if self-play quality proves good:

- 25% teacher/tactical
- 40% self-play
- 20% user/failure
- 15% anchor

Why not pure self-play immediately:

- `ttt5` is small and trap-heavy
- current product contour catches edge cases self-play may miss early
- teacher-guided tactical supervision is still valuable

Exit criteria:

- learner samples from one replay manager, not ad-hoc scattered pools
- replay restore works across restarts


## Phase 3. Add Checkpoint-vs-Champion Evaluator

Goal:
import the strongest evaluation idea from AlphaZero without losing product truth.

New evaluator responsibilities:

1. challenger vs champion
2. challenger vs engine
3. challenger on exact solved pack
4. challenger on user-trap pack
5. challenger `pure` and `hybrid` splits

Recommended new module:

- `trainer-lab/src/trainer_lab/evaluation/challenger_eval.py`

Metrics to add:

- `eloVsChampion`
- `arenaWinrateVsChampion`
- `arenaWinrateVsEngine`
- `userTrapRecallPure`
- `userTrapRecallHybrid`
- `exactSolvedRecallPure`
- `exactSolvedRecallHybrid`
- `pureHybridGap`
- `sideBalance`
- `decisiveWinRate`

Important:

Do not use upstream-style `previous checkpoint` as the only baseline.
Use `champion` as the baseline.

Exit criteria:

- no checkpoint can be promoted without beating champion on product metrics


## Phase 4. Refactor `train_variant()` Into an Orchestrator

Goal:
turn `train_variant()` into a top-level scheduler, not a giant mixed-mode trainer.

Current host:

- `apps/api/src/gomoku_api/ws/train_service_ws.py`

Target responsibility split:

- `train_service_ws.py`
  - orchestration
  - UI callbacks
  - run logging
  - promotion decision
- `trainer-lab/...`
  - self-play generation
  - replay
  - learner update
  - evaluator bridge

Refactor target:

1. bootstrap teacher phase
2. optional self-play generation phase
3. learner train phase from mixed replay
4. quick eval
5. exact/user-trap eval
6. confirm eval
7. checkpoint selection
8. candidate commit / promotion

This is the densest integration step and should be done after replay and evaluator exist.

Exit criteria:

- the trainer is easier to reason about
- self-play is no longer bolted on as a late optional block


## Phase 5. Add Product-Specific Solved Packs

Goal:
fix the biggest weakness of upstream AlphaZero for this project: blind spots on small-board tactical traps.

Maintain dedicated validation corpora:

1. `exact solved pack`
2. `edge/open-end pack`
3. `user-found trap pack`
4. `conversion pack`
5. `weak-side pack`

These should live in a deterministic testable source, not only in logs.

Recommended files:

- `trainer-lab/src/trainer_lab/data/solved_ttt5.py`
- `trainer-lab/src/trainer_lab/data/user_traps_ttt5.py`

These packs must be:

- used in validation
- used in replay boosting
- used in promotion gates

Exit criteria:

- "I can still beat the bot on screen" becomes reproducible as a stored validation case


## Phase 6. Introduce AlphaZero-Style Ratings Without Dropping Product Metrics

Goal:
borrow the useful idea of Elo/rating from upstream, but apply it correctly.

Add a rating layer for:

- self-play checkpoints
- challenger vs champion

Do not replace product metrics with Elo.
Use Elo as an additional stability signal.

Recommended usage:

- track `eloVsChampion`
- use it for trend monitoring
- do not allow Elo alone to promote a checkpoint

Exit criteria:

- we can tell whether self-play contour is truly strengthening over time


## What To Port Almost Verbatim

These upstream ideas are worth close adaptation:

1. actor/learner/evaluator role separation
2. checkpoint synchronization between actor and learner
3. replay save/load for resume
4. visit-count targets from MCTS
5. deterministic evaluator without root noise
6. larger first-layer padding for Gomoku edge handling


## What To Adapt Heavily

These should not be copied blindly:

1. environment API
   - our project has product services, variant handling, and serving lifecycle
2. evaluator logic
   - upstream compares mainly to previous checkpoint
   - we must compare to champion and product packs
3. replay
   - upstream uses one uniform replay
   - we need mixed replay with teacher/tactical/user/self-play sources
4. promotion
   - upstream rating loop is not enough for product promotion
5. network output size assumptions
   - our padded 16x16 policy handling must remain consistent


## What Not To Import

Avoid these mistakes:

1. do not replace hybrid product runtime with pure AlphaZero MCTS serving
2. do not remove teacher/tactical supervision
3. do not make self-play the default winner before proving it with metrics
4. do not promote on self-play Elo only
5. do not discard user-game relabel corpus


## File-Level Implementation Map

### Existing files to keep and extend

- `apps/api/src/gomoku_api/ws/train_service_ws.py`
  - keep as orchestration host
- `apps/api/src/gomoku_api/ws/model_registry.py`
  - keep as lifecycle authority
- `apps/api/src/gomoku_api/ws/predict_service.py`
  - keep runtime modes and tactical serving logic
- `apps/api/src/gomoku_api/ws/user_game_corpus.py`
  - keep product relabel and extend replay export
- `trainer-lab/src/trainer_lab/self_play/player.py`
  - evolve toward stronger actor backend
- `trainer-lab/src/trainer_lab/self_play/replay_buffer.py`
  - replace/expand into source-aware replay

### New files recommended

- `trainer-lab/src/trainer_lab/self_play/actor_loop.py`
- `trainer-lab/src/trainer_lab/self_play/mixed_replay.py`
- `trainer-lab/src/trainer_lab/self_play/replay_state.py`
- `trainer-lab/src/trainer_lab/evaluation/challenger_eval.py`
- `trainer-lab/src/trainer_lab/data/solved_ttt5.py`
- `trainer-lab/src/trainer_lab/data/user_traps_ttt5.py`
- `docs/ALPHA_ZERO_MIGRATION_TRACKER.md`


## Concrete Rollout Order

### Milestone 1

- keep current serving
- add stronger self-play replay and actor separation
- do not change promotion rules yet

Success means:

- self-play becomes reliable and resumable

### Milestone 2

- mixed replay replaces ad-hoc self-play sampling
- exact/user-trap packs feed both validation and replay

Success means:

- training reacts to real trap failures, not only frozen suites

### Milestone 3

- challenger-vs-champion evaluator lands
- promotion uses champion comparison + product metrics

Success means:

- no more regressions leaking into live serving

### Milestone 4

- self-play becomes a normal branch inside the main trainer
- dashboard shows:
  - self-play replay size
  - self-play contribution ratio
  - challenger vs champion
  - exact/user-trap pass rate

Success means:

- self-play is a measured contributor, not a black box


## Acceptance Criteria

The integration is successful only if all of the following improve together:

1. `hybrid confirm winrate`
2. `balancedSideWinrate`
3. `userTrapRecall`
4. `exactSolvedRecall`
5. `pureGapRate` not exploding
6. `challenger vs champion` positive
7. serving stability preserved

If self-play raises Elo but product metrics stay weak, the integration is not complete.


## Risk Analysis

### Risk 1. Self-play destabilizes small-board tactics

Mitigation:

- keep teacher/tactical replay in the mix
- use exact/user-trap packs as hard gates

### Risk 2. Replay becomes too noisy on 5x5

Mitigation:

- canonical dedupe
- source quotas
- weighted sampling

### Risk 3. Product runtime becomes slower or inconsistent

Mitigation:

- do not change serving path during training integration
- keep MCTS runtime optional

### Risk 4. Evaluation looks better than live play

Mitigation:

- store user-found failures as solved validation cases immediately


## Recommended Immediate Next Step

The best first implementation step is not "turn on self-play everywhere".

It is:

1. add a **mixed replay manager**
2. feed it from:
   - current teacher/tactical pools
   - current user corpus
   - current self-play positions
3. keep current promotion gates
4. start measuring self-play as one source among several

This gets the maximum value from `michaelnny/alpha_zero` with the lowest product risk.


## Bottom Line

The right strategy is:

- import AlphaZero as a **training backbone**
- keep this repository's **product intelligence**
- make self-play a first-class data source
- keep promotion tied to real product outcomes

That is the path that takes the **maximum** from `michaelnny/alpha_zero` without throwing away the strongest parts already built here.
