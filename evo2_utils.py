import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/scratch/iluthra/Evo2_Predictor/evo2')
import torch
from evo2 import Evo2

def predict_evo2(sequences: dict, readout: str, scale_requested, prediction_ranges):
    #If no specific scale is request then default to linear
    if scale_requested is None:
        scale_actual = "linear"
    else:
        scale_actual = scale_requested

    prediction_dict = {}
    evo2_model = Evo2('evo2_7b')

    # NOTE: Do not modify the original sequences dictionary directly,
    # as it may be needed for multiple prediction tasks.
    # Instead, create a dictionary copy to work with.
    seqs_dict = {k: v for k, v in sequences.items()}

    # Crop input sequences to [0:range_end+1] for full upstream context 
    #Evo2 is only using the context upstream to make the predictions
    if prediction_ranges:
        for seq_id, pr in prediction_ranges.items():
            if pr and seq_id in seqs_dict:
                range_start, range_end = pr
                print(f"DEBUG: Original length of '{seq_id}': {len(seqs_dict[seq_id])} bases.")
                seqs_dict[seq_id] = seqs_dict[seq_id][:range_end+1]
                print(f"DEBUG: Cropped input to [0:{range_end+1}], length: {len(seqs_dict[seq_id])} bases.")

    seqs = list(seqs_dict.values())
    seq_ids = list(seqs_dict.keys())
    print(seqs)

    if readout == "point":
        print("Making point predictions")
        #Can send all sequences to the model as a list
        predictions = evo2_model.score_sequences(seqs)
        print(predictions)
        #The predictions from score_sequences are always in log scale
        #If linear scale is requested, convert
        if scale_actual == "linear":
            predictions = np.exp(np.array(predictions))

    if readout == "track":
        print("Making track predictions")
        predictions = evo2_model.score_sequences_track(seqs, scale = scale_actual)
        print(predictions)
       # Crop predictions to [range_start:range_end+1]
        if prediction_ranges:
            cropped = []
            for seq_id, pred in zip(seq_ids, predictions):
                pr = prediction_ranges.get(seq_id)
                if pr:
                    range_start, range_end = pr
                    print(f"DEBUG: Cropping '{seq_id}' predictions to [{range_start}:{range_end+1}]")
                    cropped.append(pred[range_start:range_end+1])
                else:
                    cropped.append(pred)
            predictions = cropped

    #Re-create the seq_id: predictions dictionary to send back to predictor_RestAPI.py
    prediction_dict = {
        seq_id: float(pred) if np.isscalar(pred) else pred.tolist()
        for seq_id, pred in zip(seq_ids, predictions)
    }    
    print(prediction_dict)
    return prediction_dict, scale_actual

