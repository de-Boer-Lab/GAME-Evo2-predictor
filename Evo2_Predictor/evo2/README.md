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

The two are related by `linear = exp(log)`. `log` is the default (and is the most natural output for a language model); `linear` is provided so an evaluator can request raw probabilities without doing the conversion itself.

---

## `prepend_bos` — why it differs from `score_sequences`

| Function | Default `prepend_bos` |
|---|---|
| `score_sequences` | `False` |
| `score_sequences_track` | `True` |


In `logits_to_logprobs`, the last logit is dropped (it predicts a token past the end of the sequence) and `input_ids` is shifted by 1 (the model predicts the *next* token at each position). Without a BOS token, the first base of the sequence has nothing to condition on and gets no logprob — the returned track is implicitly missing position 0.

For `score_sequences` this barely matters: one missing position out of thousands gets averaged away. For `score_sequences_track`, position 0 would be silently dropped from every track, breaking alignment with the input. Prepending BOS gives the model a starting context so position 0 of the sequence has a defined logprob, and the returned array length matches the input sequence length.

---

## Output shape

Each returned element is a NumPy array of shape `(L,)` — one likelihood per input base. The returned list has the same length and order as the input `seqs` list.

---

## How it's used by the predictor

`evo2_utils.predict_evo2` calls `score_sequences_track` for any `readout == "track"` request:

```python
predictions = evo2_model.score_sequences_track(seqs, scale=scale_actual)
```

If `prediction_ranges` is present in the request, the input sequence is first cropped to `[0:end+1]` (full upstream context is preserved because Evo2 is autoregressive and only uses upstream tokens), then the returned track is cropped to `[start:end+1]` before being returned to the evaluator.


## Compatibility

These additions are purely additive. Code that imports `score_sequences`, `score_sequences_rc`, or the `Evo2` class as before will behave identically to upstream Evo2 v0.5.3.
