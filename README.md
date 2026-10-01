# D.A.R.T. — Decision Already Reached, Thanks

A small, fast decision model. Give it a query and a list of options; it returns **every option with its probability, ranked**, from **one forward pass**. There is no text generation, no decoding loop and no language-model head: a fine-tuned Qwen3-0.6B reads the prompt once, and a tiny decision head turns the final hidden state into option scores.

This repository contains the whole project: the data pipeline, the training and evaluation code, the packaged model API, a terminal app (**Mark-1**) and a web app (**Mark-2**: React + FastAPI + MongoDB).

| | |
|---|---|
| Accuracy (held-out public test split) | 51.2% zero-shot → **72.5%** fine-tuned, better in all six decision types |
| Latency (RTX 3050 Laptop GPU, 4 GB) | **about 17 ms** mean end to end, 33 ms worst case (256-token prompt) |
| Shared-context questions | about **2.1x faster** than answering each question separately, identical decisions |
| Memory | 1.4 GB of GPU memory at inference; trains in 4 GB |

> **Honest note on the numbers.** Accuracy is measured on held-out test splits of the *same public sources* used for training, not on a hand-verified set of real user data, so treat 72.5% as optimistic. A real held-out set is still to be built (see [Status](#status-and-limitations)).

---

## How it works

```
query (+ optional shared context) + list of options
        |
fixed plain-text prompt        Context: ...   Query: ...   Options: A) ... B) ...   Answer:
        |
tokenizer (left padding, at most 256 tokens)
        |
Qwen3-0.6B backbone (no vocabulary head), ONE forward pass, no KV cache needed
        |
hidden state of the LAST token
        |
decision head: Linear(1024 -> 12 slots)
        |
mask the slots beyond the number of options -> softmax
        |
sort descending -> [(option, probability), ...]; the top option is the decision
```

Why the last token: Qwen3 is a causal model, so only the final position has seen the whole prompt. Prompts are **left**-padded so the last position is always a real token, and position ids are computed from the attention mask so padding never shifts them.

---

## Model architecture

### Backbone: Qwen3-0.6B (`Qwen/Qwen3-0.6B`, loaded with `AutoModel`, no LM head)

| Property | Value |
|---|---|
| Type | Decoder-only transformer, causal attention, all 28 layers full attention |
| Layers | 28 |
| Hidden size | 1,024 |
| Attention | 16 query heads, 8 key/value heads (grouped-query), head dimension 128 |
| MLP | SwiGLU (SiLU gate), intermediate size 3,072 |
| Normalisation / position | RMSNorm, rotary position embeddings (theta 1,000,000) |
| Vocabulary | 151,936 tokens, tied embeddings (no separate output matrix is loaded) |
| Parameters | about 0.6 billion |
| Inference precision | float16 (weights merged); training used bfloat16 |

### LoRA adapters (training)

| Property | Value |
|---|---|
| Rank / alpha / dropout | 64 / 128 / 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` in all 28 layers |
| Trainable parameters | about 40.4 million (the backbone itself stays frozen) |
| At export | merged into the backbone, so inference needs no `peft` library and pays no adapter cost |

### Decision head

| Property | Value |
|---|---|
| Layer | one `Linear(1024 -> 12)`, kept in float32 (12,300 parameters) |
| Option slots | 12, so a question can have 2 to 12 options |
| Masking | slots beyond the option count get a large negative value before softmax, so they receive probability 0 |
| Output | softmax over the valid slots, sorted descending; the top option is the decision |
| Confidence | the raw softmax probability, unmodified and uncalibrated |

### Decision formats (one formulation for all)
Every example is a query plus an ordered option list; the label is the index of the correct option.
Types trained on: categorical (2-8 options), yes/no, low/medium/high, scale 1-5, scale 0-5 and scale 0-10. Ordered types use neighbour-smoothed targets, so a neighbouring value counts as "less wrong".

### Training recipe

| Item | Setting |
|---|---|
| Loss | cross-entropy with label smoothing 0.05; extra 0.10 neighbour mass for ordered types |
| Sample weights | rarer decision types and sources count more (inverse square root), hard slice 1.5x, rare scale values up-weighted |
| Optimiser | AdamW, LoRA learning rate 1e-4, head learning rate 1e-3, cosine schedule, 3% warm-up, gradient clipping 1.0 |
| Batch | 8 per step x 4 accumulation = 32 |
| Memory savers | gradient checkpointing, bf16 backbone, fp32 head |
| Length | 1 epoch (about 2,000 steps), roughly 1.9 hours on an RTX 3050 Laptop GPU (4 GB) |
| After training | the decision head was re-fitted on frozen backbone features (removes never-predicted scale values), then the adapter was merged and exported |

### Training data
71,590 training rows (2,210 validation) from 13 public datasets plus 620 cleaned rows of an earlier synthetic set: CLINC150, AG News, GoEmotions, MMLU (auxiliary train), ARC, CommonsenseQA, BoolQ, PubMedQA, RACE, SST-5, Yelp, STS-B and HelpSteer2. 15% of rows carry synthetic typos and slang (the "noisy" slice), 8,294 rows belong to shared-context groups, and every prompt fits in 256 tokens. The original 20k synthetic file was rejected after an audit (about 74% of its rows were templates with random labels).

### Results (4,366 held-out public test rows)

| Decision type | Zero-shot | Fine-tuned |
|---|---|---|
| binary | 71.3% | 82.6% |
| categorical | 64.4% | 83.2% |
| low / medium / high | 48.3% | 71.1% |
| scale 0-10 | 11.4% | 33.3% |
| scale 0-5 | 16.1% | 55.6% |
| scale 1-5 | 23.5% | 55.1% |
| **overall** | **51.2%** | **72.5%** |

The 0-10 scale is the weakest: its labels are rounded similarity and quality ratings and are noisy. Top-2 accuracy is 89.7%.

### Speed

| Path | Single-decision latency |
|---|---|
| Standard PyTorch (CPU-launch-bound) | about 70 ms |
| **CUDA-graph replay** with fp16 (used by Mark-1 and Mark-2) | **about 15 ms** typical, 33 ms at 256 tokens; 17 ms mean including tokenizing |
| CPU only (this laptop, 1 thread, int8 dynamic quantisation, accuracy not verified) | about 450 ms |

Prompts are left-padded to a length bucket (64, 96, 128, 192, 256) and one CUDA graph is kept per bucket. The fast path is not bit-identical to standard execution (decisions matched on 397 of 400 rows at bf16; the differences are near-ties).

### Parallel decisions over a shared context
Several questions about one passage read the passage once: its attention cache is computed once, repeated along the batch, and the short per-question parts run together. Each question is read from its own last real token, so its answer equals the standalone answer (maximum probability difference 9e-6 in fp32; 459 of 459 identical decisions on the real model). See `dart/parallel.py`.

---

## Repository layout

```
dart/            model, training, evaluation, packaging, fast inference, parallel decisions
  model.py         DecisionModel: backbone + last-token pooling + masked 12-slot head
  data.py          dataset, collator (left padding), length bucketing
  loss.py          neighbour-smoothed targets, weighted loss, sample weights
  metrics.py       accuracy, top-2, MAE, within-one-step, per-group summaries
  train.py         LoRA training (supports --init-from to continue a run)
  baseline.py      zero-shot baseline
  evaluate.py      evaluates a checkpoint and compares it with the baseline
  head_refit.py    re-fits only the decision head on cached backbone features
  fast_inference.py  CUDA-graph predictor
  parallel.py      shared-prefix KV-cache decisions
  api.py           public API: DART.from_pretrained, decide, decide_many
  export.py        exports a checkpoint as a self-contained artifact folder
  verify_artifact.py acceptance check for an exported artifact
data_prep/       dataset pipeline (salvage, public-dataset converters, noise, build, report)
Mark-1/          terminal app
Mark-2/          web app
  backend/dartweb/   FastAPI service (options parser, store, rate limiter, app)
  backend/tests/     backend tests
  frontend/          React + TypeScript (Vite)
tests/           core tests (data, model, loss, parallel, fast inference, API, determinism)
requirements-lock.txt   exact package versions used
```

Not in the repository (too large, see `.gitignore`): the exported model (`artifacts/`), training runs (`runs/`) and all datasets (`assets/`).

---

## Setup

Tested on Windows 11, Python 3.13, Node 22 and MongoDB 8, with an NVIDIA RTX 3050 Laptop GPU (4 GB). Commands are PowerShell, run from the repository root.

```powershell
python -m venv .venv
# PyTorch with CUDA 12.6 (use the CPU build if you have no NVIDIA GPU)
.\.venv\Scripts\python.exe -m pip install "torch==2.8.0+cu126" --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
.\.venv\Scripts\python.exe -m pip install transformers==5.5.0 peft==0.19.1 pandas pyarrow scikit-learn numpy pytest huggingface_hub accelerate
.\.venv\Scripts\python.exe -m pip install -r Mark-2/backend/requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
`requirements-lock.txt` lists the exact versions used.

---

## Reproducing the model

```powershell
# 1. Tests (core)
.\.venv\Scripts\python.exe -m pytest tests -q

# 2. Download the public datasets into assets/public/<folder> (see the table below)
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download as d; [d(repo_id=r, repo_type='dataset', local_dir='assets/public/'+f, allow_patterns=p) for f, r, p in [('clinc150','clinc/clinc_oos',None),('agnews','fancyzhx/ag_news',None),('goemotions','google-research-datasets/go_emotions',['simplified/*']),('mmlu','cais/mmlu',['auxiliary_train/*']),('arc','allenai/ai2_arc',None),('commonsenseqa','tau/commonsense_qa',None),('boolq','google/boolq',None),('pubmedqa','qiaojin/PubMedQA',['pqa_labeled/*']),('race','ehovy/race',['all/*']),('sst5','SetFit/sst5',None),('yelp','Yelp/yelp_review_full',None),('stsb','nyu-mll/glue',['stsb/*']),('helpsteer2','nvidia/HelpSteer2',['train.jsonl.gz','validation.jsonl.gz'])]]"

# 3. Build train / val / proxy_test files in assets/processed (run data_prep.salvage first if you also have the original synthetic file)
.\.venv\Scripts\python.exe -m data_prep.build_dataset

# 4. Zero-shot baseline, then LoRA training (about 1.9 h for one epoch on a 4 GB GPU)
.\.venv\Scripts\python.exe -m dart.baseline --batch-size 32
.\.venv\Scripts\python.exe -m dart.train --epochs 1 --out runs/lora_run1

# 5. Evaluate, optionally re-fit the head, export, verify
.\.venv\Scripts\python.exe -m dart.evaluate --run runs/lora_run1
.\.venv\Scripts\python.exe -m dart.head_refit --run runs/lora_run1 --batch-size 16
.\.venv\Scripts\python.exe -m dart.export --run runs/lora_run1_refit
.\.venv\Scripts\python.exe -m dart.verify_artifact
```
The download command above fetches exactly the 13 datasets `build_dataset.py` reads (folder names matter): `clinc150`, `agnews`, `goemotions`, `mmlu`, `arc`, `commonsenseqa`, `boolq`, `pubmedqa`, `race`, `sst5`, `yelp`, `stsb` and `helpsteer2`. `build_dataset` also reads `assets/processed/salvaged.jsonl`, which `python -m data_prep.salvage` creates from the original synthetic file (`assets/datasets/dart_train_20k.jsonl`, not included). If you don't have that file, create an empty `assets/processed/salvaged.jsonl` and the build will use the public datasets only. Check each dataset's licence before using it commercially.

Using the packaged model in code:

```python
from dart.api import DART

dart = DART.from_pretrained("artifacts/dart_v1.0")
result = dart.decide("My card was stolen, what should I do?", ["report lost card", "check weather", "transfer money"])
print(result["decisions"][0]["decision"], result["decisions"][0]["confidence"])

many = dart.decide_many("Tony lives in England and is 16.",
                        [{"id": "where", "query": "Where does Tony live?", "options": ["China", "England"]},
                         {"id": "age", "query": "How old is Tony?", "options": ["12", "16"]}])
```
Output format (per decision): `id`, `ranked` (every option with its probability, descending), `decision`, `confidence`.

---

## Mark-1 — terminal app

A small interactive program for trying the trained model. Specification:

| | |
|---|---|
| File | `Mark-1/mark1.py` |
| Model | the exported artifact `artifacts/dart_v1.0` (override with `--artifact`) |
| Inputs | a question, an optional context, and options (comma-separated, or use `\|` between options that contain commas) |
| Output | every option with its probability and a bar, the decision, the confidence, and the **latency in milliseconds** |
| Options | 2 to 12 per question; the whole prompt must fit in 256 tokens |
| Flags | `--device cpu` to run without a GPU, `--no-fast` to skip the CUDA-graph path |
| Speed | about 11 ms per decision on a GPU for short prompts (the model load is excluded) |

```powershell
.\.venv\Scripts\python.exe Mark-1\mark1.py
```
Type `quit` at any prompt to exit.

---

## Mark-2 — web app (React + FastAPI + MongoDB)

The model as a standalone application. One query box (context and question together), one options box, and a result showing the decision, confidence, ranked probabilities, model latency and round-trip time.

### Specification

| Part | Details |
|---|---|
| Frontend | Vite + React + TypeScript, plain CSS, DART branding; the dev server proxies `/api` to the backend |
| Backend | FastAPI; loads the model once at startup (GPU if available, else CPU); inference runs behind a lock |
| Database | MongoDB collection `dart.decisions`: stores every query, options, the model's ranked output, confidence, latency, model version and optional user feedback (for future training) |
| Input | `query`: the whole text, context and question together (1 to 2,000 characters). `options`: a list or a range. |
| Output | the same structure as Mark-1, as JSON: ranked options with probabilities, decision, confidence, `latency_ms` |
| Safety | per-client rate limit, CORS allow-list, input length caps, clear JSON errors, no IP addresses stored, database outage never blocks an answer (30-second circuit breaker) |

**Options syntax**

| You type | The model gets |
|---|---|
| `yes, no, maybe` | those three options (use `\|` if an option contains a comma) |
| `0-5` (also `0 to 5`, `0..5`) | 0, 1, 2, 3, 4, 5 |
| `1-10` | 1 to 10 |
| `0.0-1.0` | 0.0, 0.1, ... 1.0 (decimal because an end point is decimal) |
| `0-5 step 0.5` | 0.0, 0.5, ... 5.0 |
| `-1 to 1` | -1, 0, 1 (use `to` for negative numbers) |

A range may have at most 12 values; otherwise the API explains which `step` to add. For best results name the scale in the query ("Rate this review from 0 to 5: ...") and put the question last. The model was trained on integer scales (0-5, 1-5, 0-10); decimal ranges work but are outside its training.

### API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | `{status, model_loaded, device, model_version}` |
| `POST /api/decide` | body `{query, options}`; returns `id`, `model`, `decisions[0] {id, ranked[{option, probability}], decision, confidence}`, `latency_ms`, `options` (resolved), `options_kind`, `stored` |
| `POST /api/decisions/{id}/feedback` | body `{correct_option}` (must be one of the decision's options); saves the label for training |

Interactive API docs are served at `/docs` while the backend runs.

### Run it

1. **MongoDB** (optional but needed to store inputs): start the Windows service from an Administrator PowerShell with `Start-Service MongoDB`. Without it the app still answers, with `"stored": false`.
2. **Backend** (window 1):
   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn dartweb.app:app --app-dir Mark-2/backend --port 8000
   ```
   Wait about 20 seconds for `Application startup complete`; check http://localhost:8000/api/health.
3. **Frontend** (window 2):
   ```powershell
   cd Mark-2\frontend
   npm install
   npm run dev
   ```
   Open http://localhost:5173.

Settings (environment variables, all optional):

| Variable | Default | Meaning |
|---|---|---|
| `DART_MODEL_DIR` | `artifacts/dart_v1.0` | exported model folder |
| `DART_DEVICE` | auto (`cuda` if available) | `cuda` or `cpu` |
| `MONGODB_URI` / `MONGODB_DB` | `mongodb://localhost:27017` / `dart` | database |
| `CORS_ORIGINS` | `http://localhost:5173` | comma-separated allowed origins |
| `RATE_LIMIT_PER_MINUTE` | `60` | per client; `0` turns it off |
| `VITE_API_URL` (frontend) | empty (use the dev proxy) | backend address for a built/deployed frontend |

### Turning user inputs into training data
Label decisions with the feedback endpoint, then export them in the training format:
```powershell
.\.venv\Scripts\python.exe Mark-2\backend\export_training_data.py
```
This writes `assets/processed/mark2_labeled.jsonl` (rows with feedback) and `mark2_unlabeled.jsonl` (inputs still to be labelled).

### Tests
```powershell
.\.venv\Scripts\python.exe -m pytest tests Mark-2/backend/tests -q     # Python
cd Mark-2\frontend; npm test; npm run build                            # front end
```

---

## Status and limitations

- **Built and verified:** data pipeline, training, evaluation, fast inference, parallel decisions, packaged API, Mark-1, Mark-2 (run locally).
- **Not done yet:**
  - a hand-verified real held-out set, needed for formal sign-off and to judge any further improvement honestly;
  - hard-slice emphasis training and the reinforcement-learning stage;
  - deployment of Mark-2 (it runs locally; on a CPU-only host latency is hundreds of milliseconds, not about 15 ms).
- **Known limits:** at most 12 options; prompts over 256 tokens are rejected; training data is general public text, so niche domains may be weaker; the 0-10 scale is the least accurate type; probabilities are raw model output, not calibrated; the GPU fast path can be slower (70 to 150 ms) for the first request after a few idle seconds, because laptop GPUs drop to a low clock when idle.
