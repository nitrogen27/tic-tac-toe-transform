# Training Baselines

Reference metrics for comparing future training runs and variants. Numbers
here are frozen from the last successful runs — new pipelines or model
profiles should beat these before being considered progress.

> **Baseline captured:** 2026-04-18 (from runs on 2026-04-11).

## ttt5 (5×5, win = 4 in a row)

- **Variant ID:** `ttt5`
- **Board:** `board_size = 5`, `win_length = 4`
- **Model profile:** `standard` — ResNet, `resFilters=96`, `resBlocks=8`,
  `valueFc=160`, **~1.5M parameters** (1 508 455)
- **Training branch:** `codex/alpha-zero-integration-plan`

### Last promoted checkpoint (baseline)

| Field | Value |
|---|---|
| Timestamp | 2026-04-11 11:50:07 |
| Generation | 9 |
| Reason | `passed_all_gates` |
| `winrateVsAlgorithm` | **0.75** |
| `blockAccuracy` | **92.0%** |
| `winAccuracy` | **88.5%** |
| `balancedSideWinrate` | 0.5 |
| `winrateAsP2` | 0.5 |
| Code HEAD | `3b0c09c` — *feat: enforce champion gate and p2-balanced ttt5 training* |

### Last completed training run

| Field | Value |
|---|---|
| Run ID | `20260411T173959_ttt5` |
| Start → End | 2026-04-11 17:39:59 → 18:24:50 |
| Wall-clock | 2691.3 s (~45 min) |
| Final `winrateVsChampion` | 0.5 (no promotion) |
| Code HEAD | `fe6c601` — *feat: detach training worker and improve live monitoring* |
| Log | `saved/training_logs/ttt5/20260411T173959_ttt5.jsonl` |
| Artifacts | `saved/ttt5_resnet/{candidate,candidate_working,champion,model}.pt`, `saved/ttt5_resnet/manifest.json` |

### Training configuration (from `train.start` + `train.progress` payload)

| Field | Value |
|---|---|
| `trainingMode` | `cyclic_engine_exam` (iterations: 9, cycles: 9) |
| `epochs` (declared) | 25 |
| `totalSteps` per epoch | 120 |
| `batchSize` (declared in train.start) | 1024 |
| `batchSize` (actual mini-batch in progress) | 256 |
| `learningRate` | 0.002 |
| `mixedPrecision` | true (AMP fp16) |
| `tf32` | true |
| `channelsLast` | true |
| `device` | CUDA, NVIDIA GeForce RTX 3060 Laptop GPU (6 GB) |
| `teacherBackend` / `confirmBackend` | `builtin` / `builtin` |
| `userCorpusEnabled` | false |
| Training dataset size | 3190 positions (`effectivePositions`) |
| Throughput at steady state | ~250 samples/s, GPU 98–99 % |
| Peak reserved VRAM | ~500 MB |

### Full-run per-epoch metrics (from `background_train.done`)

Trend from `train.done.metricsHistory` — 25 of ~29 recorded epochs:

| Epoch | Loss | Acc % | MAE |
|---|---|---|---|
| 1  | 2.770 | 15.9 | 0.504 |
| 5  | 2.199 | 32.7 | 0.347 |
| 10 | 1.847 | 43.9 | 0.490 |
| 15 | 1.395 | 57.9 | 0.428 |
| 20 | 1.284 | 63.0 | 0.395 |
| 23 | 1.109 | 66.7 | 0.341 |
| 25 | 1.243 | 64.3 | 0.366 |
| 29 | 1.236 | 63.? | 0.??? |

### Pipeline defaults (`trainer-lab/src/trainer_lab/config.py`)

These are the defaults the self-play pipeline reads when no override is
supplied — record overrides when future runs deviate.

| Field | Default |
|---|---|
| `TrainConfig.batch_size` | 256 |
| `TrainConfig.epochs` | 30 |
| `SelfPlayConfig.games` (per generation) | 200 |
| `SelfPlayConfig.replay_buffer_max` | 20 000 |
| `SelfPlayConfig.evaluation_games` | 20 |
| `SelfPlayConfig.min_replay_samples` | 1024 |
| MCTS `num_simulations` (self_play/player.py) | 400 |
| Per-learner replay sample size | `batch_size * 4` = 1024 positions |

### Known pipeline characteristics at baseline

Structural properties of the code that produced the numbers above. Call out
changes to any of them when reporting new baselines — speed-ups here need to
be measured against the 2691 s wall-clock.

- Single sequential loop `self-play → replay → learner → evaluator` — see
  `trainer-lab/src/trainer_lab/self_play/pipeline.py`.
- MCTS and board state in pure Python (`list[list[int]]`, full board copy
  per move, O(N²) legal-move scan).
- Replay buffer and positions dataset persisted as JSON.
- `augment=True` expands each position to 8 D4 symmetries.
- DataLoader without `num_workers` (synchronous main-thread loading).
- GPU idle during self-play / evaluation phases (CPU-bound pure-Python MCTS).

### How to reproduce

```bash
cd C:/gitlab/ml/tic-tac-toe-transform
git checkout fe6c601            # or 3b0c09c for the last promoted run
npm run start:legacy-ui-api
# trigger ttt5 training from the UI
```

## Recording new baselines

When adding a new variant (e.g., `gomoku9`, `gomoku15`) or changing the
pipeline (NumPy/bitboard, process-parallel self-play, binary replay, etc.):

1. Run **ttt5** with the new pipeline and record its numbers here under a
   new dated entry — regression gate against the tables above.
2. Record the new variant's first `passed_all_gates` metrics (winrate vs
   algorithm, block/win accuracy, balanced side winrate).
3. Record wall-clock duration per generation, per epoch, and samples/s.
4. Record code commit SHA at the time of the baseline run.
5. Note which pipeline defaults/characteristics changed and which stayed the
   same (bullet lists above).
