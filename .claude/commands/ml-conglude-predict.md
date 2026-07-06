---
name: ml-conglude-predict
description: >
  Score unlabeled ligands against proteins using ConGLUDe. Invoke ONLY when
  the user explicitly mentions "ConGLUDe" or "conglude" together with prospective
  prediction intent — e.g. "run predict", "score ligands", "virtual screening",
  "predict affinity", "run predict.py", "screen a library", "embed a library",
  "encode SMILES", "run embed_ligands_scalable", "embed Enamine", "embed ligands
  at scale", "sharded embedding", "compute ligand embeddings for a large file".
  Do NOT trigger on retrospective evaluation or benchmarking requests with labeled
  ground truth — use the eval skill for those.
version: 2.0.0
---

# ConGLUDe — Prospective Prediction

Score ligands against proteins with no labeled data. Two pathways depending on library size:

- **Small libraries** (up to ~1M SMILES): run `predict.py` directly (handles embedding internally).
- **Large libraries** (millions+): first encode with `embed_ligands_scalable.py`, then screen with `screen_large_library.py`.

---

## Inputs to collect

Ask for each item not already provided:

1. **Dataset directory** — full path to the dataset folder that will hold the input data and receive outputs. Can be anywhere on the filesystem (e.g. `/projects/my_screen`).
2. **Proteins** — one of:
   - A list of PDB IDs (4-character codes, e.g. `5O1I 4AGO 2VUK`)
   - A path to a local `protein_ids.txt` file (one PDB ID per line)
   - A path to a directory of `.pdb` files
3. **Ligands / SMILES** — one of:
   - A path to a `smiles.txt` file (one SMILES string per line — plain text, no header, no names)
   - A path to a `.csv`, `.tsv`, or `.sdf` file that needs to be converted
   - Raw SMILES strings pasted directly into the chat
4. **Library size** — if not obvious from context, ask the user roughly how many ligands they have. This determines which pathway to use.

---

## Pathway selection

| Condition | Pathway |
|-----------|---------|
| ≤ ~1M SMILES, or user says "small" / "predict" | **Small** — `predict.py` |
| > ~1M SMILES, or user says "large library" / "Enamine" / "embed at scale" | **Large** — `embed_ligands_scalable.py` → `screen_large_library.py` |
| Pre-computed shard embeddings already exist | **Large** — skip embedding, go straight to `screen_large_library.py` |

---

## Data preparation (both pathways)

**IMPORTANT: Underscores in protein/PDB names are not allowed.** The code uses `_` as a delimiter internally (e.g. `filename.split('_')[0]` to extract protein IDs from graph filenames). If a source PDB file contains underscores (e.g. `my_protein.pdb`), replace all `_` with `-` when setting up the ConGLUDe data directory. Apply this renaming to:
- The PDB filename in `raw/pdb_files/` (e.g. `my-protein.pdb`)
- The entry in `info/protein_ids.txt` (e.g. `my-protein`)

For external datasets, never modify the original source data — only rename within the created `ConGLUDe` directory.

**If proteins were given as PDB IDs**, write them one-per-line to `<dataset_dir>/info/protein_ids.txt`.
PDB files will be auto-downloaded if they are not already present.

**If a local PDB directory was given**, pass it via `--pdb_dir`. Still write the PDB IDs (filenames without `.pdb`) to `protein_ids.txt`.

**If SMILES were given as a CSV**, extract the SMILES column:
```python
import pandas as pd
df = pd.read_csv("input.csv")
smiles_col = [c for c in df.columns if 'smiles' in c.lower()][0]
df[smiles_col].dropna().to_csv("<dataset_dir>/info/smiles.txt", index=False, header=False)
```

**If SMILES were given as an SDF**, extract them with RDKit:
```python
from rdkit import Chem
writer = open("<dataset_dir>/info/smiles.txt", "w")
for mol in Chem.SDMolSupplier("input.sdf"):
    if mol is not None:
        writer.write(Chem.MolToSmiles(mol) + "\n")
writer.close()
```

**If SMILES were pasted directly**, write them to the file one per line.

---

## Conda environment (required before any Python command)

Activate the `conglude` conda environment before running any Python command in this skill:

```bash
conda activate conglude
```

All `python` commands below assume this environment is active. In the Bash tool, activate via:

```bash
bash -c 'source /SW/python/miniconda3/x86_64/etc/profile.d/conda.sh && conda activate conglude && python -u ...'
```

Do NOT use `conda run -n conglude` — it buffers all stdout/stderr until the process exits, which prevents real-time logging to tee/log files.

---

## GPU setup (required before running)

Both ESM embedding computation (during data processing) and model inference use GPU. Before running, you **must** select a free GPU:

1. Check GPU utilization:
```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

2. Pick a GPU with low memory usage and low utilization, then export it:
```bash
export CUDA_VISIBLE_DEVICES=<free_gpu_id>
```

3. Pass `--device cuda:0` (the default) — this refers to the first GPU visible after the CUDA_VISIBLE_DEVICES filter.

---

## Logging

All runs must be logged to a file so output is preserved for debugging and review. Before running any command, create the logs directory and set up a timestamped log file:

```bash
mkdir -p logs
LOG_FILE="logs/predict_$(date +%Y-%m-%d_%H-%M-%S).log"
```

Append `2>&1 | tee "$LOG_FILE"` to every `python predict.py`, `python embed_ligands_scalable.py`, or `python screen_large_library.py` command so output goes to both the terminal and the log file.

After the run completes (success or failure), tell the user where the log file is and ask whether to keep or remove it using `AskUserQuestion`.

---

# Small Library Pathway

## Directory layout

**For external datasets**, set `dataset_dir = <path>` (the root directory). The code creates:
```
<path>/ConGLUDe/
├── data/
│   ├── info/
│   │   ├── protein_ids.txt   ← one PDB ID per line
│   │   └── smiles.txt        ← one SMILES string per line
│   ├── raw/
│   │   └── pdb_files/        ← optional: local PDB files
│   └── processed/            ← created automatically during run
└── results/                  ← created automatically during run
```
Create the `<path>/ConGLUDe/data/info/` directory and place input files there before running.

**For repo datasets**, use the existing structure:
```
<dataset_dir>/
└── info/
    ├── protein_ids.txt   ← one PDB ID per line
    └── smiles.txt        ← one SMILES string per line
```

## Run command

```bash
export CUDA_VISIBLE_DEVICES=<free_gpu_id>
python predict.py \
  --dataset_dir <dataset_dir> 2>&1 | tee "$LOG_FILE"
```

Add optional flags as appropriate:
- `--smiles_path <path>` — if the SMILES file is not at `<dataset_dir>/info/smiles.txt`
- `--pdb_dir <path>` — if PDB files are stored locally (not to be auto-downloaded); defaults to `<dataset_dir>/raw/pdb_files`
- `--device cpu` — if no GPU is available
- `--no_save_embeddings` — to skip saving embeddings (saves disk space)
- `--protein_batch_size 32` / `--ligand_batch_size 512` — reduce if running out of memory

## Results

Output location depends on whether the dataset is inside the repo or external:

**Repo datasets** (`dataset_dir` under `./data/`):
```
results/<dataset_name>/<YYYY-MM-DD_HH-MM>/
├── predictions/
│   ├── vs_predictions.npy   ← ligand–protein similarity matrix (ligands × proteins)
│   ├── pr_predictions.npy   ← ligand–pocket similarity matrix  (ligands × pockets)
│   └── pp_predictions.csv   ← predicted pocket positions and confidence scores
└── embeddings/              ← only if --save_embeddings True (default)
    ├── protein_embeddings.npy
    ├── protein_names.txt
    ├── pocket_embeddings.npy
    └── pocket_names.txt
```

**External datasets** (`dataset_dir` = `<path>`):
```
<path>/ConGLUDe/results/<YYYY-MM-DD_HH-MM>/
├── predictions/
└── embeddings/              ← only if --save_embeddings True
```

Also mention:
- `<dataset_dir>/processed/ligand_embeddings/index2smiles.json` maps column indices in the prediction matrices to SMILES strings.
- `protein_names.txt` / `pocket_names.txt` map row indices.
- Higher values in the similarity matrices = higher predicted affinity.

---

# Large Library Pathway

For libraries with millions+ of SMILES. Two phases: encode ligands into sharded embeddings, then screen against proteins.

## Step 1 — Encode ligands (`embed_ligands_scalable.py`)

Skip this step if pre-computed shard embeddings already exist.

### Additional inputs to collect

- **File format details** (only if input is a delimited file, not plain SMILES-per-line):
  - SMILES column index (0-based, default: 0)
  - Delimiter (default: tab)
  - Whether the file has a header line (default: yes)
- **Shard size** — number of SMILES per shard (default: 1,000,000)
- **VS-only?** — whether to save only the virtual-screening half of embeddings, discarding the pocket-ranking half (default: false). Halves output size.
- **Save as float16?** — halves disk usage at minor precision cost (default: false)

### Pipeline phases

1. **Phase 0 — Splitting** (optional): splits a raw input file into numbered shard files (`smiles_0.txt`, `smiles_1.txt`, ...) in the smiles directory. Skipped if shards already exist.
2. **Phase 1 — Feature extraction** (CPU-parallel): computes ECFP4 fingerprints and molecular descriptors for each shard using RDKit. Produces compressed `.dat.gz` files. Multiple shards are processed in parallel.
3. **Phase 2 — GPU encoding** (sequential): loads features shard-by-shard, encodes through the trained MLP ligand encoder, and saves `.npz` embedding files.

Each phase is resumable — already-completed shards are skipped unless `--overwrite` is passed.

### Run command

```bash
export CUDA_VISIBLE_DEVICES=<free_gpu_id>
python embed_ligands_scalable.py \
  --input_file <path_to_smiles_file> \
  --smiles_dir <output_dir>/info \
  --output_dir <output_dir> \
  --shard_size 1000000 2>&1 | tee "$LOG_FILE"
```

From pre-split shard files:
```bash
python embed_ligands_scalable.py \
  --smiles_dir <dir_containing_smiles_N_txt> \
  --output_dir <output_dir> 2>&1 | tee "$LOG_FILE"
```

### Common flags

| Flag | Description |
|------|-------------|
| `--input_file <path>` | Raw input file to split into shards |
| `--smiles_column <int>` | Column index of SMILES (default: 0) |
| `--delimiter <str>` | Field delimiter (default: tab) |
| `--no_header` | Input file has no header to skip |
| `--shard_size <int>` | SMILES per shard (default: 1,000,000) |
| `--vs_only` | Only save VS (protein-space) embedding half |
| `--save_f16` | Save embeddings as float16 |
| `--batch_size <int>` | GPU batch size (default: 4096) |
| `--num_workers <int>` | CPU workers per shard (default: 32) |
| `--n_parallel <int>` | Shards processed in parallel (default: 16) |
| `--memory_limit <float>` | Min available RAM in GB before launching new shards (default: 200) |
| `--shard_start <int>` | First shard index to process |
| `--shard_end <int>` | Last shard index (exclusive) |
| `--shard_indices <int...>` | Explicit shard indices (overrides start/end) |
| `--features_only` | Only compute features, skip GPU encoding |
| `--encode_only` | Only run GPU encoding (features must exist) |
| `--overwrite` | Recompute even if outputs exist |
| `--no_descriptors` | Disable molecular descriptor computation |
| `--scaler_dir <path>` | Pre-fitted RobustScaler directory (default: `data/common/scalers`) |

### Embedding output layout

```
<output_dir>/
└── processed/
    └── ligand_embeddings/
        ├── ecfp4_2048_0.dat.gz              ← fingerprints shard 0
        ├── descriptors_0.dat.gz             ← descriptors shard 0
        ├── metadata_ecfp4_2048_0.json       ← feature metadata
        ├── metadata_embeddings_0.json       ← embedding metadata
        ├── index2smiles_0.json              ← index-to-SMILES mapping per shard
        └── embeddings_0.npz                 ← encoded embeddings
```

### Memory and parallelism tuning

- **`--memory_limit`**: Pauses new shard jobs when RAM drops below threshold. Set to ~50% of total RAM on shared machines.
- **`--n_parallel`**: Concurrent shard feature-extraction processes. Total CPU threads = `n_parallel * num_workers`.
- **`--batch_size`**: GPU batch size for the MLP encoder. The MLP is lightweight — 4096+ usually fits easily.

### Resuming interrupted runs

The pipeline is fully resumable:
- Phase 0 skips splitting if shard files already exist in `--smiles_dir`
- Phase 1 skips shards that already have a `metadata_*.json` file
- Phase 2 skips shards that already have an `embeddings_N.npz` file

To reprocess specific failed shards, use `--shard_indices`:
```bash
python embed_ligands_scalable.py \
  --smiles_dir <dir> --output_dir <dir> \
  --shard_indices 5 12 37 \
  --overwrite 2>&1 | tee "$LOG_FILE"
```

---

## Step 2 — Screen against proteins (`screen_large_library.py`)

Once shard embeddings exist, score them against proteins:

```bash
export CUDA_VISIBLE_DEVICES=<free_gpu_id>
python screen_large_library.py \
  --protein_dataset_dir <protein_dataset_dir> \
  --ligand_emb_dir <output_dir> \
  --smiles_dir <output_dir>/info \
  --output_dir <screening_output_dir> 2>&1 | tee "$LOG_FILE"
```

### Additional flags

| Flag | Description |
|------|-------------|
| `--protein_results_dir <path>` | Load existing protein embeddings instead of re-computing |
| `--pdb_dir <path>` | Directory containing raw PDB files |
| `--input_file <path>` | Original input file (for extracting compound IDs) |
| `--id_column <int>` | Column index of compound IDs in the original input file |
| `--smiles_column <int>` | Column index of SMILES in the original input file (for verification) |
| `--input_delimiter <str>` | Field delimiter in the original input file (default: tab) |
| `--no_header` | Original input file has no header line |
| `--top_k <int>` | Save top-k scoring ligands per protein (default: 10,000; 0 to disable) |
| `--shard_start <int>` | First shard index to process |
| `--shard_end <int>` | Last shard index (exclusive) |
| `--no_compress` | Write plain CSV instead of bz2-compressed |
| `--batch_size <int>` | Batch size for protein embedding computation |
| `--device <str>` | Device for protein embedding inference |

### Screening output

```
<screening_output_dir>/
├── vs_scores.csv.bz2                ← full score matrix (id; smiles; score_proteinA; score_proteinB; ...)
├── vs_top10000_<proteinA>.csv       ← top-k hits for protein A
└── vs_top10000_<proteinB>.csv       ← top-k hits for protein B
```

### Post-run verification

```python
import numpy as np
import os

emb_dir = "<output_dir>/processed/ligand_embeddings"
shard_files = sorted(f for f in os.listdir(emb_dir) if f.startswith("embeddings_") and f.endswith(".npz"))
total = 0
for f in shard_files:
    data = np.load(os.path.join(emb_dir, f))
    total += data["embeddings"].shape[0]
    print(f"{f}: {data['embeddings'].shape}, dtype={data['embeddings'].dtype}")
print(f"Total ligands embedded: {total:,}")
```

---

## Cleanup

After reporting results, ask the user which created folders to keep and which to delete. For external datasets, processing creates subdirectories under `<dataset_dir>/ConGLUDe/`. Present the following options:

- **`data/`** — All processing artifacts: downloaded PDB files, protein graphs (expensive to regenerate — requires ESM on GPU), ligand fingerprints/descriptors, and metadata. Delete to save space; keep to avoid re-running the full processing pipeline on subsequent predictions.
- **`results/<timestamp>/`** — Prediction results: similarity matrices and predicted pocket positions.
- **`results/<timestamp>/embeddings/`** — ConGLUDe protein/ligand embeddings (only present if `--save_embeddings True` was used). Can be large. Useful for downstream analysis or running new predictions against additional ligands without re-embedding proteins.
- **Shard embeddings** (large pathway only) — `<output_dir>/processed/ligand_embeddings/`. Reusable for screening against additional proteins without re-encoding.

Ask which folders to keep using AskUserQuestion with multiSelect, then delete the ones the user doesn't want.
