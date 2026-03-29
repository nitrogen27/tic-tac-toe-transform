# Investigation: Why Bot Played Poorly Despite High Training Accuracy

## Date: 2026-03-29
## Branch: research/bot-quality-investigation

## Summary

**Root cause: `torch.compile()` in `predict_service.py` crashed on Windows (no Triton) causing every prediction to fall back to random moves.**

The bot appeared to not learn anything despite training accuracy reaching 92%+. The model was trained correctly, loaded correctly, and produced correct predictions when tested via CLI. But the live predict path through WebSocket crashed silently and returned random moves.

## Timeline of Discovery

1. Training showed 92% accuracy on supervised + self-play data
2. Bot tested in browser — didn't block obvious 3-in-a-row threats
3. Hypothesis: maybe model architecture mismatch between training and predict
4. Verified: both use `(96, 8, 160)` for ttt5 — match
5. Hypothesis: maybe board encoding is wrong
6. Verified: encoder planes correct, current_player mapping correct
7. **Found**: `predict_service._maybe_compile_model()` calls `torch.compile()` without Triton guard
8. `torch.compile()` returns a compiled wrapper (no immediate error)
9. First `model(tensor)` call → `TritonMissing` exception
10. `except Exception` catches it → returns `{isRandom: True, fallback: True, move: random}`
11. Model is cached as compiled wrapper → **every subsequent predict also crashes**
12. Bot plays random every game

## The Bug

```python
# predict_service.py line 161-168 (BEFORE fix)
def _maybe_compile_model(model: Any) -> Any:
    if not torch.cuda.is_available() or not hasattr(torch, "compile"):
        return model
    try:
        return torch.compile(model, mode="reduce-overhead", fullgraph=False)
    except Exception as exc:
        return model  # torch.compile() succeeds — error comes LATER
```

`torch.compile()` does NOT throw on Windows without Triton. It returns a wrapper that crashes at first forward pass:

```
torch._inductor.exc.TritonMissing: Cannot find a working triton installation
```

This is caught by the outer try/except in `_model_predict()` (line 300):
```python
except Exception as exc:
    logger.error("Model predict error: %s", exc)  # logged once, then silent
    return {"move": random.choice(legal), "isRandom": True, "fallback": True}
```

## The Fix

```python
# predict_service.py (AFTER fix)
def _maybe_compile_model(model: Any) -> Any:
    if not torch.cuda.is_available() or not hasattr(torch, "compile"):
        return model
    try:
        import triton  # Guard: skip compile if Triton unavailable
    except ImportError:
        return model
    try:
        return torch.compile(model, mode="reduce-overhead", fullgraph=False)
    except Exception as exc:
        return model
```

Same guard was already in `train_service_ws.py` and `mini_bench.py` but missing in `predict_service.py`.

## Verification Results

### CLI Test (all pass)
| Test | Result |
|------|--------|
| Model loads | OK (1,508,455 params, cuda:0) |
| Not compiled | OK (Triton guard works) |
| Vertical block | OK (3,0) 44.5% |
| Horizontal block | OK (2,3) 93.1% |
| Diagonal \ block | OK (3,3) 86.7% |
| Diagonal / block | OK (3,1) 98.9% |
| Block from top | OK (0,2) 100% |

### Live Browser Test (via WebSocket intercept)
| Step | Board | Bot Move | Correct? |
|------|-------|----------|----------|
| 1. X center | [12]=1 | move=13 (2,3) | OK (adjacent) |
| 2. X builds column | [7,12]=1 | move=23 (4,3) | OK (no threat yet) |
| 3. X 3-in-column | [7,12,17]=1 | **move=2 (0,2)** conf=99.4% | **BLOCKS!** |

### Encoder Verification
- As X (current=1): plane0[0,0]=1 (self), plane1[0,0]=0 (opp) ✓
- As O (current=2): plane0[0,0]=0 (opp), plane1[0,0]=1 (self) ✓
- `current_player: 1` hardcoded correctly (board already remapped)

## Why Previous Tests Showed Bad Play

Every test before the Triton fix (`a7b6a75`) was against a random-move bot:
- `beec666` (revert) — random
- `9489000` (single-phase) — random
- `33f7624` (supervised_pool fix) — random
- `7759a2c` (quota fixes) — random
- All the way back to when `torch.compile` was first added

The training was always correct. The model was always good. The predict was always broken on Windows.

## Lessons Learned

1. **`torch.compile()` errors are deferred** — they don't throw at compile time, only at first forward pass
2. **Silent fallback to random is dangerous** — `isRandom: True` was logged but not shown to the user
3. **Same guard needed everywhere** — Triton guard was in train_service but missing in predict_service
4. **CLI tests ≠ server tests** — CLI doesn't go through `_maybe_compile_model`, server does
5. **Always check `fallback` field** in predict results during testing
