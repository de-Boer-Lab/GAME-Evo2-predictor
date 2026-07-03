import torch
from evo2 import Evo2

evo2_model = Evo2('evo2_7b')

sequence = 'ACGT'
input_ids = torch.tensor(
evo2_model.tokenizer.tokenize(sequence),
dtype=torch.int,
).unsqueeze(0).to('cuda:0')

outputs, _ = evo2_model(input_ids)
logits = outputs[0]

print('Logits: ', logits)
print('Shape (batch, length, vocab): ', logits.shape)

sequence = ['ACGTATGCATATGCTGCATCG', "ATGGCTAGCTCG"]
print(len(sequence))
#These are log scale by default
var_scores_point = evo2_model.score_sequences(sequence)
print(var_scores_point)
#Need to add code to unlog for linear scale requests

var_scores_track = evo2_model.score_sequences_track(sequence, scale = "log")
print(var_scores_track)
print(len(var_scores_track[0]))
print(len(var_scores_track[1]))

var_scores_track = evo2_model.score_sequences_track(sequence, scale = "linear")
print(var_scores_track)
print(len(var_scores_track[0]))
print(len(var_scores_track[1]))