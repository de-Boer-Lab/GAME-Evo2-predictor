# Evo2 — Modified Scoring

This is a modified copy of [arcinstitute/evo2](https://github.com/ArcInstitute/evo2) (v0.5.3) with one addition: **per-base pair scoring** for the GAME `track` readout. The upstream Evo2 package only exposes a per-sequence (scalar) score via `score_sequences`. This fork adds `score_sequences_track`, which returns one likelihood per base instead of reducing to a single number. All other Evo2 functionality is left unchanged. 

---

## What was added

| File | Addition |
|---|---|
| `scoring.py` | New functions: `logits_to_logprobs_track`, `_score_sequences_track`, `score_sequences_track`. |
| `models.py` | New method on the `Evo2` class: `Evo2.score_sequences_track(seqs, scale, ...)`. |
| `__init__.py` | No change — `Evo2` is still the only public export. |

The new functions mirror the structure of the upstream `score_sequences`: a public batched scorer (`score_sequences_track`) wraps an internal per-batch helper (`_score_sequences_track`), which calls a logits-to-likelihoods conversion (`logits_to_logprobs_track`).

---

## The scoring idea for track requests

Evo2 is an autoregressive DNA language model: at each position it outputs a distribution over the vocabulary predicting the *next* token. For a sequence of length `L`, the model produces logits of shape `(L, vocab)`.

The original `score_sequences` reduces these to a single scalar per sequence:

1. Softmax → log-probabilities, shape `(L, vocab)`.
2. Gather along the vocab dimension using the observed token IDs → shape `(L,)`. This pulls out, for each position, the log-probability the model assigned to the actual base that appears next.
3. Reduce (`mean` or `sum`) across positions → one scalar per sequence

`score_sequences_track` keeps steps 1–2 and **skips the reduction**, returning the per-position array directly.

```
score_sequences        →  mean/sum over positions  →  scalar per sequence
score_sequences_track  →  no reduction            →  (L,) array per sequence
```

---

## `scale` parameter

`score_sequences_track` takes a required `scale` argument:

| `scale` | Computation | Output range |
|---|---|---|
| `"log"` | `torch.log_softmax(logits, dim=-1)` | `(-∞, 0]` (natural log) |
| `"linear"` | `torch.softmax(logits, dim=-1)` | `[0, 1]` |

The two are related by `linear = exp(log)`. 

---

## `prepend_bos` — the Predictor always uses `True`

`predict_evo2` passes `prepend_bos=True` on every call, so all three paths (plain point, point+range, track) are BOS-aligned and consistent with one another.

Why it matters: in `logits_to_logprobs`, the last logit is dropped (it predicts a token past the end of the sequence) and `input_ids` is shifted by 1 (the model predicts the *next* token at each position). Without a BOS token, the first base has nothing to condition on and gets no logprob — position 0 is dropped and every position shifts by one, so base *i* no longer sits at index *i*.

Prepending BOS gives position 0 a defined logprob, makes the returned array length match the input length, and puts **base *i* at index *i***. That index alignment is what makes range slicing correct: the Predictor slices tracks to `[range_start:range_end+1]` for both `track` outputs and point+range means, and without BOS every one of those slices would be off by one and silently target the wrong bases.

---

## Output shape

Each returned element is a NumPy array of shape `(L,)` — one likelihood per input base. The returned list has the same length and order as the input `seqs` list.

---

## How it's used by the Predictor

`evo2_utils.predict_evo2` uses `score_sequences_track` for **two** paths — every `track` request, and `point` requests that carry `prediction_ranges`.

**`track` readout.** The full per-base array is returned to the Evaluator:

```python
predictions = evo2_model.score_sequences_track(seqs, scale=scale_actual, prepend_bos=True)
```

**`point` readout with ranges.** A point+range score is exactly a track scored over the range, then averaged — so the same function is reused, scored in `log` (converted to `linear` afterward if requested), sliced to the range, and meaned:

```python
track = evo2_model.score_sequences_track(seqs, scale="log", prepend_bos=True)
score = np.mean(track_i[start:end + 1])   # per sequence
```

(A `point` request *without* ranges instead uses the upstream `score_sequences`, called with `prepend_bos=True` so its scores stay consistent with the track-based paths.)

**Range handling.** When `prediction_ranges` is present, the input sequence is first cropped to `[0:range_end+1]` before scoring. Full upstream context is preserved because Evo2 is autoregressive and only conditions on upstream tokens. After scoring, `track` outputs are cropped to `[range_start:range_end+1]` before being returned to the Evaluator.

---


## Compatibility

These additions are purely additive. Code that imports `score_sequences`, `score_sequences_rc`, or the `Evo2` class as before will behave identically to upstream Evo2 v0.5.3.
