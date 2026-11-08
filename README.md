# Evo2 Predictor

A [GAME](https://genomic-api-for-model-evaluation-documentation.readthedocs.io/)-compatible Predictor that wraps the Evo2 7B DNA language model for use against any GAME Evaluator. Given a set of input sequences and prediction tasks, the Predictor returns either per-sequence or per-position likelihood scores from the model.

**Underlying model:** Brixi, G., Durrant, M.G., Ku, J., *et al.* 2026. Evo2 7B (1M-token context). See [arcinstitute/evo2](https://github.com/ArcInstitute/evo2).

## Important Links

- To learn more about the GAME Framework ([Main GAME Repository](https://github.com/de-Boer-Lab/Genomic-API-for-Model-Evaluation), [preprint](https://www.biorxiv.org/content/10.1101/2025.07.04.663250v1.full))
- GAME Documentation: [ReadTheDocs](https://genomic-api-for-model-evaluation-documentation.readthedocs.io)
- Pre-built Evo2-7B container image: [Hugging Face](https://huggingface.co/datasets/deBoerLab/Evo2_Predictor_GAME)
- To learn more about Evo2: [Evo2 GitHub Repository](https://github.com/ArcInstitute/evo2)
- List of all [GAME Modules](https://github.com/de-Boer-Lab/GAME_modules)

---

## Features

- **`point` readout** — one scalar score per sequence (mean log-likelihood).
- **`track` readout** — one score per base, returned as an array per sequence.
- **`log` and `linear` scales** — request either log-likelihoods or raw probabilities. Note that for `point` requests the `linear` value is `exp(mean(log p))`, i.e. the geometric mean of the per-base probabilities, not the arithmetic mean.
- **`prediction_ranges`** — optionally restrict predictions to a sub-region of each input sequence. Ranges are 0-based and half-open, `[start, end)`. Evo2 conditions only on upstream context, so the sequence is cropped to `[0:end]` before scoring, and `track` outputs are cropped to `[start:end]` afterward. For `point` requests, the score is the mean log-likelihood over `[start:end]` only.
- **Optional `upstream_seq` / `downstream_seq` flanks** — appended to every input sequence before scoring.
- **JSON and MessagePack** wire formats, negotiated via `Content-Type` and `Accept` headers.
- **Auto-versioned Predictor name** — the Apptainer build date is appended to the Predictor name on container startup (e.g. `Evo2_7b_20261030-005544_EDT`) so every prediction is traceable to a specific container build.
- **Maximum sequence length** — sequences longer than 1,048,575 bases (after any `upstream_seq`/`downstream_seq` flanks are added) are rejected with a `prediction_request_failed` (422) error. This is one less than the model's 1,048,576-token context, because a BOS token is prepended to every sequence.

---

## Build the container (optional)

Build from inside `Evo2_Predictor/`. Two inputs are required that are not tracked in git:

1. **Base image** `evo2/evo2.sif`: the prebuilt Evo2 Apptainer image (~12.3 GB) from [victornemeth/evo2apptainer](https://github.com/victornemeth/evo2apptainer), which already includes CUDA, PyTorch, Vortex and flash-attn:
   ```bash
   wget -O evo2/evo2.sif "https://zenodo.org/records/15194473/files/evo2.sif?download=1"
   ```
2. **Weights** `weights/evo2_7b.pt`: the Evo2 7B checkpoint (~13 GB) from Arc Institute's Hugging Face repo [arcinstitute/evo2_7b](https://huggingface.co/arcinstitute/evo2_7b). Download `evo2_7b.pt` directly, or copy it out of the Hugging Face cache after running `Evo2('evo2_7b')` once.

```bash
cd Evo2_Predictor
apptainer build evo2_predictor.sif evo2_predictor.def
```

The build layers the GAME Predictor scripts and the modified `evo2/` package on top of the base image:

- `flask` and `msgpack` are installed for the REST API.
- The modified `evo2` package (v0.5.3, adds `score_sequences_track`) is installed with `pip install -e ... --no-deps --force-reinstall`, replacing the base image's evo2 0.1.0 without touching its PyTorch/Vortex install.
- The weights are baked into the image and loaded directly with `Evo2('evo2_7b', local_path=...)`, so there is no Hugging Face download at runtime.

> **H100 GPU required.** Evo2 7B does not run on CPU. The Predictor requires at least one CUDA-capable GPU (one H100 is sufficient for the 7B model).
---

## Run the Predictor

```bash
apptainer run --nv evo2_predictor.sif <predictor_ip> <predictor_port>
```

| Argument | Description |
|---|---|
| `predictor_ip` | IP address or hostname to bind to (e.g. `0.0.0.0`) |
| `predictor_port` | Port to listen on |

The Predictor exposes a REST API on `http://<predictor_ip>:<predictor_port>`.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/help` | GET | Predictor metadata: model, GAME schema version, publication, container authors, description, bin size. |
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
      "species": "homo_sapiens",
      "scale": "log"
    }
  ]
}
```

### Top-level fields
 
| Field | Required? | Description |
|---|---|---|
| `readout` | yes | `"point"` or `"track"`. `"interaction_matrix"` is a valid GAME readout but Evo2 cannot produce it, so it is rejected with a 422 `prediction_request_failed`. Any other value is rejected with a 400 `bad_prediction_request`. |
| `sequences` | yes | Dict of `{seq_id: sequence}`. Valid bases: `A`, `T`, `C`, `G`, `N`. |
| `prediction_tasks` | yes | List of prediction-task objects (fields described below). |
| `prediction_ranges` | optional | Dict of `{seq_id: [start, end]}`, 0-based and half-open `[start, end)`, so `start < end <= len(sequence)`. Keys must match `sequences`. Interpreted in post-flank coordinates when `upstream_seq`/`downstream_seq` are supplied. |
| `upstream_seq` | optional | String prepended to every sequence before scoring. |
| `downstream_seq` | optional | String appended to every sequence before scoring. |
 
### Fields inside each `prediction_tasks` entry
 
| Field | Required? | Description |
|---|---|---|
| `name` | yes | Label for the task; echoed back in the response. |
| `type` | yes | The kind of signal to predict. Accepted values: `"accessibility"`, `"expression"`, or any string prefixed with `binding_` or `expression_` (for example, `binding_CTCF`, `expression_K562`). `conformation_*` types are valid GAME types but Evo2 cannot predict them, so they are rejected with a 422 `prediction_request_failed`. |
| `cell_type` | yes | Cell type or tissue (e.g. `"K562"`). |
| `species` | yes | Species (e.g. `"homo_sapiens"`). |
| `scale` | optional | `"log"` or `"linear"`. Defaults to `"linear"` when omitted. |
 
---

## Response structure

```json
{
  "predictor_name": "Evo2_7b_20261030-005544_EDT",
  "bin_size": 1,
  "prediction_tasks": [
    {
      "name": "K562_accessibility",
      "type_requested": "accessibility",
      "type_actual": ["sequence_likelihood"],
      "cell_type_requested": "K562",
      "cell_type_actual": null,
      "species_requested": "homo_sapiens",
      "species_actual": null,
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

Evo2 is a DNA language model: it scores how likely each sequence is and does not condition on assay type, cell type or species. The response reflects this. `type_actual` is always `["sequence_likelihood"]`, and `cell_type_actual` and `species_actual` are `null`, since the predictor ignores these fields.

---

## Errors

- **400** `bad_prediction_request`: the request itself is malformed. This covers missing mandatory keys, unrecognized `readout`/`type`/`scale` values, non-string values, `prediction_ranges` keys that don't match `sequences`, and malformed ranges (not a list, negative indices, or `start >= end`).
- **422** `prediction_request_failed`: the request is valid GAME but Evo2 can't fulfil it. This covers an `interaction_matrix` readout, `conformation_*` types, empty sequences or invalid bases (anything other than `A/C/G/T/N`), sequences longer than 1,048,575 bases after flanking, and ranges out of bounds for their sequence.

---

## Repository layout

```
Evo2_Predictor/
├── evo2_predictor.def               # Apptainer build recipe
├── predictor_RestAPI.py             # Flask app and /predict, /help, /formats endpoints
├── evo2_utils.py                    # predict_evo2 — wraps the Evo2 model
├── config.py                        # Auto-versions the Predictor name
├── schema_validation.py             # Request validation and preprocessing
├── error_checking_functions.py      # APIError classes and validators
├── predictor_content_handler.py     # JSON / MessagePack encode + decode
├── predictor_help_message.json      # Static metadata for /help
├── weights/
│   └── evo2_7b.pt                   # Evo2 7B checkpoint (not in git; see Build)
└── evo2/                            # Modified Evo2 source — see evo2/README.md
    ├── evo2.sif                     # Base image (not in git; see Build)
    ├── pyproject.toml
    └── evo2/
        ├── models.py                # Adds Evo2.score_sequences_track method
        ├── scoring.py               # Adds score_sequences_track and helpers
        ├── utils.py                 # Model name maps
        ├── __init__.py
        └── version.py
```

The `evo2/` subdirectory is a modified copy of [arcinstitute/evo2](https://github.com/ArcInstitute/evo2) with added per base pair scoring. See [`evo2/README.md`](Evo2_Predictor/evo2/README.md) for details on the track prediction implementation.


## Citation

If you use this Predictor, please cite both the GAME framework and the underlying Evo2 model:

> Brixi, G., Durrant, M.G., Ku, J., *et al.* 2026.
