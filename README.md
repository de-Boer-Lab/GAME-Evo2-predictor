# Evo2 Predictor

A [GAME](https://genomic-api-for-model-evaluation-documentation.readthedocs.io/)-compatible predictor that wraps the Evo2 7B DNA language model for use against any GAME Evaluator. Given a set of input sequences and prediction tasks, the Predictor returns either per-sequence or per-position likelihood scores from the model.

**Underlying model:** Brixi, G., Durrant, M.G., Ku, J., *et al.* 2026. Evo2 7B (1M-token context). See [arcinstitute/evo2](https://github.com/ArcInstitute/evo2).

---

## Features

- **`point` readout** — one scalar score per sequence (mean log-likelihood).
- **`track` readout** — one score per base, returned as an array per sequence.
- **`log` and `linear` scales** — request either log-likelihoods or raw probabilities.
- **`prediction_ranges`** — optionally restrict predictions to a sub-region of each input sequence. Evo2 conditions only on upstream context, so the sequence is cropped to `[0:end+1]` before scoring, and `track` outputs are cropped to `[start:end+1]` afterward.
- **Optional `upstream_seq` / `downstream_seq` flanks** — appended to every input sequence before scoring.
- **JSON and MessagePack** wire formats, negotiated via `Content-Type` and `Accept` headers.
- **Auto-versioned predictor name** — the Apptainer build date is appended to the Predictor name on container startup (e.g. `Evo2_7b_Predictor_20251128-180629_PST`) so every prediction is traceable to a specific container build.

---

## Build the container (optional)

```bash
apptainer build evo2_predictor.sif predictor.def
```

The container is built from `python:3.13-slim` and installs `numpy`, `tqdm`, `pandas`, `msgpack`, `scipy`, `flask`, and `waitress`. The full Evo2 source (including modified scoring code) is copied in at build time from `../Evo2_Predictor`.

> **GPU required.** Evo2 7B does not run on CPU. The Predictor requires at least one CUDA-capable GPU (one H100 is sufficient for the 7B model).

---

## Run the predictor

```bash
apptainer run --nv evo2_predictor.sif <predictor_ip> <predictor_port>
```

| Argument | Description |
|---|---|
| `predictor_ip` | IP address or hostname to bind to (e.g. `0.0.0.0`) |
| `predictor_port` | Port to listen on |

The predictor exposes a REST API on `http://<predictor_ip>:<predictor_port>`.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/help` | GET | Model metadata (name, version, publication, authors, bin size). |
| `/formats` | GET | Lists supported request and response MIME types. |
| `/predict` | POST | Main prediction endpoint. |

---

## Request structure (`POST /predict`)

```json
{
  "readout": "track",
  "sequences": {
    "seq1": "ACGTACGT...",
    "seq2": "TTGCCAAT..."
  },
  "prediction_ranges": {
    "seq1": [100, 300],
    "seq2": [50, 250]
  },
  "prediction_tasks": [
    {
      "name": "K562_accessibility",
      "type": "accessibility",
      "cell_type": "K562",
      "species": "human",
      "scale": "log"
    }
  ]
}
```

### Top-level fields
 
| Field | Required? | Description |
|---|---|---|
| `readout` | yes | `"point"` or `"track"`. `"interaction_matrix"` is not supported by Evo2 and will be rejected. |
| `sequences` | yes | Dict of `{seq_id: sequence}`. Valid bases: `A`, `T`, `C`, `G`, `N`. |
| `prediction_tasks` | yes | List of prediction-task objects (fields described below). |
| `prediction_ranges` | optional | Dict of `{seq_id: [start, end]}`, inclusive and 0-indexed. Keys must match `sequences`. |
| `upstream_seq` | optional | String prepended to every sequence before scoring. |
| `downstream_seq` | optional | String appended to every sequence before scoring. |
 
### Fields inside each `prediction_tasks` entry
 
| Field | Required? | Description |
|---|---|---|
| `name` | yes | Label for the task; echoed back in the response. |
| `type` | yes | The kind of signal to predict. Accepted values: `"accessibility"`, `"expression"`, or any string prefixed with `binding_`, or `expression_` (for example, `binding_CTCF`, `expression_K562`). |
| `cell_type` | yes | Cell type or tissue (e.g. `"K562"`). |
| `species` | yes | Species (e.g. `"human"`). |
| `scale` | optional | `"log"` or `"linear"`. Defaults to `"linear"` when omitted. |
 
---

## Response structure

```json
{
  "predictor_name": "Evo2_7b_Predictor_20251128-180629_PST",
  "bin_size": 1,
  "prediction_tasks": [
    {
      "name": "K562_accessibility",
      "type_requested": "accessibility",
      "type_actual": "accessibility",
      "cell_type_requested": "K562",
      "cell_type_actual": "K562",
      "species_requested": "human",
      "species_actual": "human",
      "scale_prediction_requested": "log",
      "scale_prediction_actual": "log",
      "predictions": {
        "seq1": [-1.21, -0.87, ...],
        "seq2": [-0.95, -1.13, ...]
      }
    }
  ]
}
```

`bin_size` is only included for `track` responses (Evo2 returns one value per base, so `bin_size = 1`). For `point` responses, `predictions` values are scalars rather than arrays.

Because Evo2 is a generative DNA language model, the `_actual` fields always match the corresponding `_requested` fields — the model can produce predictions for any cell type, species, or assay type requested.

---

## Repository layout

```
Evo2_Predictor/
├── predictor.def                    # Apptainer build recipe
├── predictor_RestAPI.py             # Flask app and /predict, /help, /formats endpoints
├── evo2_utils.py                    # predict_evo2 — wraps the Evo2 model
├── config.py                        # Auto-versions the predictor name
├── schema_validation.py             # Request validation and preprocessing
├── error_checking_functions.py      # APIError classes and validators
├── predictor_content_handler.py     # JSON / MessagePack encode + decode
├── predictor_help_message.json      # Static metadata for /help
└── evo2/                            # Modified Evo2 source — see evo2/README.md
    ├── models.py                    # Adds Evo2.score_sequences_track method
    ├── scoring.py                   # Adds score_sequences_track and helpers
    ├── utils.py                     # Model name maps
    ├── __init__.py
    └── version.py
```

The `evo2/` subdirectory is a modified copy of [arcinstitute/evo2](https://github.com/ArcInstitute/evo2) with added per base pair scoring. See [`evo2/README.md`](evo2/README.md) for details on the track prediction implementation.


## Citation

If you use this Predictor, please cite both the GAME framework and the underlying Evo2 model:

> Brixi, G., Durrant, M.G., Ku, J., *et al.* 2026.
