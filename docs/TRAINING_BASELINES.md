# Training Baselines

Reference metrics for comparing future training runs and variants. Numbers
here are frozen from the last successful runs — new pipelines or model
profiles should beat these before being considered progress.

## ttt5 (5×5, win = 4 in a row)

- **Variant ID:** `ttt5`
- **Model profile:** `standard` — ResNet, `resFilters=96`, `resBlocks=8`,
  `valueFc=160`
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
| Duration | 2691.3s (~45 min) |
| Epochs | 25 |
| Final `winrateVsChampion` | 0.5 (no promotion) |
| Code HEAD | `fe6c601` — *feat: detach training worker and improve live monitoring* |
| Log | `saved/training_logs/ttt5/20260411T173959_ttt5.jsonl` |
| Artifacts | `saved/ttt5_resnet/{candidate,candidate_working,champion,model}.pt`, `saved/ttt5_resnet/manifest.json` |

### Known pipeline characteristics at baseline

These are the pipeline properties that produced the numbers above. Call out
changes to any of them when reporting new baselines.

- Single sequential loop `self-play → replay → learner → evaluator` — see
  `trainer-lab/src/trainer_lab/self_play/pipeline.py`
- MCTS and board state in pure Python (`list[list[int]]`, full board copy
  per move, O(N²) legal-move scan)
- Learner per generation samples `batch_size * 4` from replay (1024 positions
  at `batch_size=256`)
- Replay buffer and positions dataset persisted as JSON
- `augment=True` expands each position to 8 D4 symmetries
- DataLoader without `num_workers` (synchronous main-thread loading)
- GPU: RTX 3060 Laptop, 6 GB

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
   new dated entry — regression gate against the table above.
2. Record the new variant's first `passed_all_gates` metrics (winrate vs
   algorithm, block/win accuracy, balanced side winrate).
3. Record wall-clock duration per generation and per epoch.
4. Record code commit SHA at the time of the baseline run.
5. Note which pipeline characteristics changed (bullet list above) and which
   stayed the same.
