import os
import sys
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'evo2'))

from evo2 import Evo2

# Dynamically read the baked weights path from the container environment
_WEIGHTS = os.environ.get("EVO2_WEIGHTS", "/opt/evo2-predictor/weights/evo2_7b.pt")

print(f"Loading Evo 2 7B into memory from {_WEIGHTS}...")
evo2_model = Evo2('evo2_7b', local_path=_WEIGHTS)

def predict_evo2(sequences: dict, readout: str, scale_requested, prediction_ranges):
    #If no specific scale is request then default to linear
    if scale_requested is None:
        scale_actual = "linear"
    else:
        scale_actual = scale_requested

    prediction_dict = {}

    # NOTE: Do not modify the original sequences dictionary directly,
    # as it may be needed for multiple prediction tasks.
    # Instead, create a dictionary copy to work with.
    seqs_dict = {k: v for k, v in sequences.items()}

    # Crop input sequences to [0:range_end] for full upstream context 
    #Evo2 is only using the context upstream to make the predictions
    #Prediction ranges are assuming post-flank added coordinates
    if prediction_ranges:
        for seq_id, pr in prediction_ranges.items():
            if pr and seq_id in seqs_dict:
                range_start, range_end = pr
                print(f"DEBUG: Original length of '{seq_id}': {len(seqs_dict[seq_id])} bases.")
                seqs_dict[seq_id] = seqs_dict[seq_id][:range_end]
                print(f"DEBUG: Cropped input to [0:{range_end}], length: {len(seqs_dict[seq_id])} bases.")

    seqs = list(seqs_dict.values())
    seq_ids = list(seqs_dict.keys())
    
    if readout == "point":
        if not prediction_ranges:
            print("Making point predictions with no prediction ranges requested")
            #Can send all sequences to the model as a list
            #prepend_bos=True to stay consistent with the track/ranges paths (base i -> index i)

            predictions = evo2_model.score_sequences(seqs, prepend_bos=True)
            #The predictions from score_sequences are always in log scale
            #If linear scale is requested, convert
            if scale_actual == "linear":
                predictions = np.exp(np.array(predictions))
        elif prediction_ranges:
            print("Making point predictions with prediction ranges requested")
            # Point-with-ranges == track-with-ranges then mean over the range only.
            # Track path uses prepend_bos=True, so base i sits at index i and the
            # [start:end] slice lands exactly on the requested range.
            track = evo2_model.score_sequences_track(seqs, scale="log", prepend_bos=True)

            predictions = []
            for seq_id, pred in zip(seq_ids, track):
                pr = prediction_ranges.get(seq_id)
                if pr:
                    range_start, range_end = pr
                    pred = pred[range_start:range_end]
                predictions.append(np.mean(pred))
            predictions = np.array(predictions)
            #track scale="log" always returns log; convert if linear requested
            if scale_actual == "linear":
                predictions = np.exp(predictions)


    if readout == "track":
        print("Making track predictions")
        predictions = evo2_model.score_sequences_track(seqs, scale = scale_actual, prepend_bos=True)

       # Crop predictions to [range_start:range_end]
        if prediction_ranges:
            cropped = []
            for seq_id, pred in zip(seq_ids, predictions):
                pr = prediction_ranges.get(seq_id)
                if pr:
                    range_start, range_end = pr
                    print(f"DEBUG: Cropping '{seq_id}' predictions to [{range_start}:{range_end}]")
                    cropped.append(pred[range_start:range_end])
                else:
                    cropped.append(pred)
            predictions = cropped

    assert len(predictions) == len(seq_ids), \
        f"got {len(predictions)} predictions for {len(seq_ids)} sequences"

    #Re-create the seq_id: predictions dictionary to send back to predictor_RestAPI.py
    prediction_dict = {
        seq_id: float(pred) if np.isscalar(pred) else pred.tolist()
        for seq_id, pred in zip(seq_ids, predictions)
    }    
    torch.cuda.empty_cache()
    return prediction_dict, scale_actual

