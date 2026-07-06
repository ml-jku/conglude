---
name: ml-conglude-eval
description: >
  Measure ConGLUDe model performance on labeled benchmark data. Invoke ONLY when
  the user explicitly mentions "ConGLUDe" or "conglude" together with intent to
  assess model quality on data with known ground truth — e.g. "evaluate",
  "benchmark", "run eval.py", "compute metrics", "test set performance", "AUC",
  "BEDROC", "enrichment factor", "retrospective". Do NOT trigger on prospective
  screening, scoring, or ranking requests without labeled ground truth — use the
  predict skill for those.
version: 1.0.0
---

# ConGLUDe — Retrospective Evaluation

Evaluate ConGLUDe on labeled benchmark data (known actives/inactives) to measure model performance, optionally followed by visualization of results.

---

## Inputs to collect

Ask for each item not already provided:

1. **Dataset directory** — path to the labeled dataset (e.g. `./data/datasets/test_datasets/litpcba` for repo datasets, or an external path)
2. **Save embeddings?** — whether to write embedding files to disk (default: false)
3. **Visualize results?** — whether to generate plots after evaluation (default: yes if `save_predictions` is enabled)
4. **PyMOL visualizations?** — whether to generate PyMOL `.pse` scene files showing predicted pockets on the protein structure (default: false). If yes, pass `datamodule.test_datasets.default_vs.save_cleaned_pdbs=true` during evaluation so the cleaned PDB files needed for PyMOL scene generation are saved.

---

## Required data layout

The directory structure depends on whether the dataset is inside the repo or external:

### Repo datasets (path starts with `./data/` or `data/`)

`dataset_dir` is used directly as the data root:

```
<dataset_dir>/
├── info/
│   └── protein_ids.txt                      ← one PDB ID per line (always required)
└── raw/
    └── smiles_files/
        ├── <pdb_id>/
        │   ├── actives.txt                  ← one SMILES per line
        │   └── inactives.txt                ← one SMILES per line
        └── <another_pdb_id>/
            ├── actives.txt
            └── inactives.txt
```

### External datasets (any other absolute path)

The code appends `ConGLUDe/data/` to `dataset_dir` as the data root. This keeps ConGLUDe processing artifacts separate from the original source data. When preparing an external dataset, place files under this subdirectory:

```
<dataset_dir>/
└── ConGLUDe/
    └── data/
        ├── info/
        │   └── protein_ids.txt              ← one PDB ID per line (always required)
        └── raw/
            ├── pdb_files/                   ← optional; PDB files placed here skip download
            │   └── <pdb_id>.pdb
            └── smiles_files/
                ├── <pdb_id>/
                │   ├── actives.txt          ← one SMILES per line
                │   └── inactives.txt        ← one SMILES per line
                └── <another_pdb_id>/
                    ├── actives.txt
                    └── inactives.txt
```

The `pdb_dir` parameter defaults to `<data_root>/raw/pdb_files` (i.e. `<dataset_dir>/ConGLUDe/data/raw/pdb_files/` for external datasets). Place PDB files there to avoid automatic download from RCSB.

Each PDB ID listed in `protein_ids.txt` gets its own subdirectory under `raw/smiles_files/` containing `actives.txt` and `inactives.txt`. When the same ligand set applies to multiple PDB structures (e.g. different crystal structures of the same protein), duplicate the actives/inactives files into each PDB's subdirectory — each PDB is evaluated independently.

**IMPORTANT: Never use `multi_pdb_targets: true`** unless it is already explicitly set in an existing dataset-specific config. For new evaluations with multiple PDB structures for the same protein target, always set up each PDB as a separate entry in `protein_ids.txt` with its own smiles_files subdirectory. The model evaluates each PDB independently and results are reported per-PDB.

**IMPORTANT: Underscores in protein/PDB names are not allowed.** The code uses `_` as a delimiter internally (e.g. `filename.split('_')[0]` to extract protein IDs from graph filenames). If a source PDB file contains underscores (e.g. `my_protein.pdb`), replace all `_` with `-` when setting up the ConGLUDe data directory. Apply this renaming to:
- The PDB filename in `raw/pdb_files/` (e.g. `my-protein.pdb`)
- The entry in `info/protein_ids.txt` (e.g. `my-protein`)
- The subdirectory name under `raw/smiles_files/` (e.g. `my-protein/`)

For external datasets, never modify the original source data — only rename within the created `ConGLUDe` directory.

---

## PDB file validation (required before running)

**The processing pipeline is all-or-nothing**: it indexes SMILES only for proteins whose PDB file parsed successfully. Any PDB failure means those proteins' ligands are also excluded from the shared feature files (fingerprints, descriptors). Fixing PDB files after the fact requires a full rerun of processing. Therefore, **always validate and prepare all PDB files before launching `eval.py`**.

After writing `protein_ids.txt` and the SMILES files:

1. **Pre-populate `raw/pdb_files/`** — do NOT rely on the pipeline's auto-download (it only handles standard 4-char PDB IDs available in `.pdb` format on RCSB).

2. **Non-standard protein IDs** (anything not a plain 4-char PDB code) cannot be auto-downloaded. Extract the base PDB ID, download it, and copy with the suffixed name for each entry.

3. **CIF-only entries** — newer PDB depositions may lack `.pdb` format. Download the `.cif` and convert via BioPython (`MMCIFParser` → `PDBIO.save()`). Note: BioPython's PDB writer can fail on large structures with residue numbers exceeding PDB format limits — if conversion fails, try `gemmi` or find an alternative PDB ID.

4. **Validate all files** — loop over `protein_ids.txt` and confirm every `<id>.pdb` exists and is parseable by `PDBParser`. Report failures to the user and fix (alternative PDB ID, AlphaFold structure, or remove the entry) before proceeding.

**Only proceed to `eval.py` when all entries pass validation.**

---

## Dataset config (YAML)

Evaluation uses Hydra. For new/custom datasets, use the `default_vs` config and override `dataset_dir`. Only use dataset-specific configs when re-evaluating an existing benchmark dataset.

### Existing benchmark datasets

These have their own configs with dataset-specific settings: `asd`, `coach420`, `dude`, `holo4k`, `kinobeads`, `litpcba`, `pdbbind_refined`, `pdbbind_time`, `posebusters`.

To evaluate one of these, uncomment its line in `configs/datamodule/test_datasets/test_datasets.yaml` (e.g. `- litpcba: litpcba`).

### New/custom datasets

Uncomment `- default_vs: default_vs` in `test_datasets.yaml` and override `dataset_dir` on the command line (see Run evaluation below).

The `default_vs` config uses sensible defaults (`task: vs`, `structure_based: false`, `labeled_smiles: binary`). Override additional options via Hydra if needed (e.g. adjust `batch_size` if running out of memory).


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

Both ESM embedding computation (during data processing) and model inference use GPU. Before running eval.py, you **must** select a free GPU:

1. Check GPU utilization:
```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

2. Pick a GPU with low memory usage and low utilization, then export it:
```bash
export CUDA_VISIBLE_DEVICES=<free_gpu_id>
```

3. Always pass `trainer.devices=1` to avoid DDP multi-GPU mode. With `devices=auto` (the default), PyTorch Lightning spawns one process per visible GPU, which corrupts prediction CSV files due to concurrent writes. Single-device evaluation is correct and sufficient for test-time inference.

---

## Logging

All runs must be logged to a file so output is preserved for debugging and review. Before running any command, create the logs directory and set up a timestamped log file:

```bash
mkdir -p logs
LOG_FILE="logs/eval_$(date +%Y-%m-%d_%H-%M-%S).log"
```

Append `2>&1 | tee "$LOG_FILE"` to every `python eval.py` command so output goes to both the terminal and the log file.

After the run completes (success or failure), tell the user where the log file is and ask whether to keep or remove it using `AskUserQuestion`.

---

## Run evaluation

### Existing benchmark dataset

First set the dataset in `configs/datamodule/test_datasets/test_datasets.yaml`, then:
```bash
export CUDA_VISIBLE_DEVICES=<free_gpu_id>
python eval.py trainer.devices=1 model.save_predictions=false model.save_embeddings=false 2>&1 | tee "$LOG_FILE"
```

By default, predictions and embeddings are NOT saved for predefined benchmark datasets (they are slow to write and usually not needed — metrics are sufficient). Only add `model.save_predictions=true` or `model.save_embeddings=true` if the user explicitly requests saving predictions/embeddings.

### Custom dataset with default_vs

First set `default_vs` in `test_datasets.yaml`, then override `dataset_dir`:
```bash
export CUDA_VISIBLE_DEVICES=<free_gpu_id>
python eval.py datamodule.test_datasets.default_vs.dataset_dir=<path_to_dataset> trainer.devices=1 model.save_predictions=false model.save_embeddings=false 2>&1 | tee "$LOG_FILE"
```

### Common Hydra overrides

```bash
# Enable saving predictions/embeddings (only when user explicitly requests)
python eval.py model.save_predictions=true model.save_metrics=true model.save_embeddings=true trainer.devices=1 2>&1 | tee "$LOG_FILE"

# Dataset-specific options (e.g. for custom datasets)
python eval.py datamodule.test_datasets.default_vs.batch_size=8 trainer.devices=1 2>&1 | tee "$LOG_FILE"
```

---

## Results folder layout

Each dataset gets its **own results directory**:
- **Repo datasets** (under `./data/`): `./results/<dataset_name>/<YYYY-MM-DD_HH-MM>/`
- **External datasets** (`dataset_dir` = `<path>`): `<path>/ConGLUDe/results/<YYYY-MM-DD_HH-MM>/`

Subfolders per dataset: `predictions/`, `embeddings/`, `metrics/`, `plots/`

When evaluating multiple datasets simultaneously, each dataset writes to its own directory (e.g. `results/litpcba/2026-05-25_18-37/`, `results/coach420/2026-05-25_18-37/`).

When `save_metrics=true`, each dataset saves:
- **`metrics/summary.csv`** — aggregate dataset-wide metrics (all task types). Columns: `metric, value`.
- **VS**: additionally `metrics/metrics.csv` with per-protein columns: `protein, auc, bedroc, ef_0.005, ef_0.01, ef_0.05`
- **TF**: additionally `metrics/metrics.csv` with per-molecule columns: `molecule, auprc, delta_auprc, auc, ef_0.05, ef_0.01, ef_0.005`

---

## Visualization (optional)

After evaluation completes, generate plots if the user requested visualization. Requirements:
- All task types: `save_metrics=true` (produces `summary.csv`)
- VS datasets: additionally `save_predictions=true` (for score distributions, ROC curves, enrichment curves)
- TF datasets: additionally `save_predictions=true` (for per-molecule ROC curves, enrichment curves)

### Run visualization

`visualize.py` accepts **multiple run directories** (one per dataset). Pass any combination of dataset directories:

```bash
python visualize.py --run-dir results/litpcba/<timestamp> results/kinobeads/<timestamp> results/coach420/<timestamp> --no-pymol
```

The script processes each directory sequentially and generates task-appropriate plots:
- **All datasets**: `summary.png` — bar chart of aggregate dataset-wide metrics (from `summary.csv`)
- **VS datasets**: score distributions, per-protein ROC curves, per-protein enrichment curves, per-target metric bar charts (from `vs_predictions.csv` and `metrics.csv`)
- **TF datasets** (e.g. kinobeads): per-molecule ROC curves with mean overlay, per-molecule enrichment curves with mean overlay (from `vs_predictions.csv`). VS-style plots are automatically skipped.
- **PP/PR datasets**: only the summary bar chart (no per-target breakdown)
- Directories without the expected files are skipped gracefully

If the user requested PyMOL visualizations, omit `--no-pymol`. This requires that `save_cleaned_pdbs: true` was set during evaluation. If the cleaned PDBs are at a non-standard location, point to them with `--dataset-root <path>`.

### Output

After generating the plots, report:
1. The paths to all saved figure files
2. A brief summary of results per task type
3. Any notable outliers — targets with particularly good or poor performance

---

## Cleanup

After reporting results, ask the user which created folders to keep and which to delete. For external datasets, processing creates subdirectories under `<dataset_dir>/ConGLUDe/`. Present the following options:

- **`data/`** — All processing artifacts: downloaded PDB files, protein graphs (expensive to regenerate — requires ESM on GPU), ligand fingerprints/descriptors (can be large), and metadata. Delete to save space; keep to avoid re-running the full processing pipeline on subsequent evaluations.
- **`results/<timestamp>/`** — Evaluation results: per-protein metrics (CSV), per-ligand prediction scores, and visualization PNGs (ROC curves, enrichment curves, score distributions, metric comparisons).
- **`results/<timestamp>/embeddings/`** — ConGLUDe protein/ligand embeddings (only present if `save_embeddings=true` was used). Can be large. Useful for downstream analysis or transfer learning.

Ask which folders to keep using AskUserQuestion with multiSelect, then delete the ones the user doesn't want.
