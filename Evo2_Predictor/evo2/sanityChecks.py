import numpy as np
import torch
from evo2 import Evo2

evo2_model = Evo2('evo2_7b')

# --- Test 1: Length check ---
# Track predictions should return seq_length values (one per position)
print("=== Test 1: Length check ===")
sequence = ['ACGTATGCAT']
track = evo2_model.score_sequences_track(sequence, scale='log')
assert len(track[0]) == len(sequence[0]), f"Expected {len(sequence[0])}, got {len(track[0])}"
print(f"PASSED: track length {len(track[0])} == sequence length {len(sequence[0])}")

# --- Test 2: Log vs linear scale ---
#np.exp(log_scores) should equal linear_scores
print("\n=== Test 2: Log vs linear scale ===")
sequence = ['ACGTATGCAT']
track_log = evo2_model.score_sequences_track(sequence, scale='log')
track_linear = evo2_model.score_sequences_track(sequence, scale='linear')

diff = np.abs(np.exp(track_log[0]) - track_linear[0])
print(f"Max diff: {diff.max()}")
print(f"Mean diff: {diff.mean()}")

print(np.exp(track_log[0]))
print(track_linear[0])
assert np.allclose(np.exp(track_log[0]), track_linear[0], atol=1e-2), "Log and linear scales don't match!"
print(f"PASSED: exp(log_scores) matches linear_scores")

# --- Test 3: Linear scale values should be in [0, 1] ---
# Since they are probabilities
print("\n=== Test 3: Linear scale in [0, 1] ===")
sequence = ['ACGTATGCAT']
track_linear = evo2_model.score_sequences_track(sequence, scale='linear')
assert np.all(track_linear[0] >= 0) and np.all(track_linear[0] <= 1), "Linear scores out of [0, 1] range!"
print(f"PASSED: all linear scores in [0, 1]: {track_linear[0]}")

# --- Test 4: Log scale values should be <= 0 ---
# Log probabilities are always negative
print("\n=== Test 4: Log scale values <= 0 ===")
sequence = ['ACGTATGCAT']
track_log = evo2_model.score_sequences_track(sequence, scale='log')
assert np.all(track_log[0] <= 0), "Log scores should be <= 0!"
print(f"PASSED: all log scores <= 0: {track_log[0]}")

# --- Test 5: Repetitive sequence should have high probability after first repeat ---
# A highly repetitive sequence like AAAA... should have increasing confidence
# after the model sees the pattern
print("\n=== Test 5: Repetitive sequence ===")
sequence = ['AAAAAAAAAAAAAAAAAAA']
track_linear = evo2_model.score_sequences_track(sequence, scale='linear')
# Later positions should generally have higher probability than early ones
early_mean = np.mean(track_linear[0][:3])
late_mean = np.mean(track_linear[0][-3:])
print(f"Early position mean prob: {early_mean:.4f}")
print(f"Late position mean prob:  {late_mean:.4f}")
print(f"Model is {'more' if late_mean > early_mean else 'less'} confident at later positions")

# --- Test 6: High vs low complexity sequence ---
# A simple repetitive sequence should score higher than a random one
print("\n=== Test 6: High vs low complexity ===")
simple_seq = ['ATATATATATATATATATATAT']
complex_seq = ['ACGTTGCAATCGGATCGAATCG']
point_simple = evo2_model.score_sequences(simple_seq)
point_complex = evo2_model.score_sequences(complex_seq)
print(f"Simple (repetitive) mean log prob: {point_simple[0]:.4f}")
print(f"Complex (random)    mean log prob: {point_complex[0]:.4f}")
print(f"Simple scores {'higher' if point_simple[0] > point_complex[0] else 'lower'} than complex")

# --- Test 7: Batch consistency ---
# Scoring sequences individually vs in a batch should give same results
print("\n=== Test 7: Batch consistency ===")
seq1 = ['ACGTATGCAT']
seq2 = ['TGCATACGAT']
track_individual_1 = evo2_model.score_sequences_track(seq1, scale='log')
track_individual_2 = evo2_model.score_sequences_track(seq2, scale='log')
track_batch = evo2_model.score_sequences_track(seq1 + seq2, scale='log')
assert np.allclose(track_individual_1[0], track_batch[0], atol=1e-3), "Batch result differs for seq1!"
assert np.allclose(track_individual_2[0], track_batch[1], atol=1e-3), "Batch result differs for seq2!"
print("PASSED: individual and batch scoring are consistent")

# --- Test 8: First base has no logprob (without BOS) ---
# Two sequences identical except for the first base.
# Without BOS, the first base is never predicted so track outputs should
# differ only at position 0 — but since position 0 is dropped by the shift,
# the remaining positions will differ because the first base affects context.
# This test just confirms position 0 is not scored (length is seq_length - 1 without BOS).
print("\n=== Test 8: First base has no logprob without BOS ===")
seq_a = ['ACGTATGCAT']
seq_b = ['TCGTATGCAT']  # Only first base differs (A -> T)

track_a = evo2_model.score_sequences_track(seq_a, scale='log', prepend_bos=False)
track_b = evo2_model.score_sequences_track(seq_b, scale='log', prepend_bos=False)

# Without BOS we should get seq_length - 1 values (first base never predicted)
assert len(track_a[0]) == len(seq_a[0]) - 1, f"Expected {len(seq_a[0])-1}, got {len(track_a[0])}"
assert len(track_b[0]) == len(seq_b[0]) - 1, f"Expected {len(seq_b[0])-1}, got {len(track_b[0])}"
print(f"PASSED: without BOS, track length is seq_length - 1 ({len(track_a[0])})")
print(f"Seq A track (no BOS): {track_a[0]}")
print(f"Seq B track (no BOS): {track_b[0]}")
print(f"Scores differ because first base still affects context for downstream positions")

# With BOS, first base IS predicted so we get full seq_length values
track_a_bos = evo2_model.score_sequences_track(seq_a, scale='log', prepend_bos=True)
track_b_bos = evo2_model.score_sequences_track(seq_b, scale='log', prepend_bos=True)
assert len(track_a_bos[0]) == len(seq_a[0]), f"Expected {len(seq_a[0])}, got {len(track_a_bos[0])}"
print(f"PASSED: with BOS, track length is full seq_length ({len(track_a_bos[0])})")
print(f"Seq A position 0 logprob (with BOS): {track_a_bos[0][0]:.4f}")
print(f"Seq B position 0 logprob (with BOS): {track_b_bos[0][0]:.4f}")
print(f"Position 0 logprobs differ because A != T")