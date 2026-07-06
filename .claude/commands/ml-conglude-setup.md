---
name: ml-conglude-setup
description: >
  Install or troubleshoot the ConGLUDe conda environment and repository.
  Invoke when the user mentions "ConGLUDe" or "conglude" together with
  environment setup intent — e.g. "install conglude", "set up the environment",
  "create the conda env", "fix import errors", "setup_env.sh", "install
  dependencies", "CUDA mismatch", "torch not found", "environment broken".
  Do NOT trigger for running predictions, evaluation, or embedding tasks.
version: 1.0.0
---

# ConGLUDe — Environment Setup

Install the ConGLUDe conda environment, automatically adapting to the target hardware.

---

## Target environment name

This skill installs and verifies the **`conglude`** conda environment. All other ConGLUDe skills (train, eval, predict, embed) require this environment to be active before running Python commands:

```bash
conda activate conglude
```

---

## Overview

`setup_env.sh` accepts an optional CUDA wheel tag as its first argument. The tag controls which PyTorch and torch-scatter wheels are installed. If omitted, it defaults to `cu128`.

```bash
bash setup_env.sh          # uses cu128 (default)
bash setup_env.sh cu121    # uses cu121 (for older drivers, e.g. CUDA 12.1–12.7)
```

The `cu121` variant installs PyTorch 2.1.2; all other tags install PyTorch 2.7.0.

Before running the script, **always detect the hardware first** to choose the correct tag if not explicitly specified by the user.

---

## Step 1: Detect hardware (ALWAYS do this first)

Run these commands and **explain each result to the user** so they understand what was detected:

```bash
# Check for NVIDIA GPU and driver
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null

# Maximum CUDA toolkit version supported by driver
nvidia-smi 2>/dev/null | grep "CUDA Version"
```

After running, tell the user:
- Which GPU was detected (or that none was found)
- Which CUDA version their driver supports
- Which tag you will use and why

### Decision logic

Based on the output, determine which tag to pass to `setup_env.sh`:

| Condition | Tag argument |
|-----------|--------------|
| `nvidia-smi` reports CUDA Version >= 12.8 | `cu128` (or omit — it's the default) |
| `nvidia-smi` reports CUDA Version 12.1–12.7 | `cu121` |
| `nvidia-smi` reports CUDA Version 11.8–12.0 | `cu118` |
| `nvidia-smi` fails or no GPU | `cpu` |

Inform the user which tag was selected and what it means (e.g. "Your driver supports CUDA 12.3, so I'll use `cu121` which installs PyTorch 2.1.2 with CUDA 12.1 support — this is backward-compatible with your driver.").

---

## Step 2: Check prerequisites

Tell the user what you're checking and why before each command.

### conda / mamba

Check if conda is available. Conda is required to manage the environment and install RDKit (which has no reliable pip package).

If conda is not installed, tell the user: "Conda is required for this setup. I'll download Miniforge — a lightweight conda installer — and run it."
```bash
# Linux x86_64
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

On HPC clusters, check for a module system first and explain: "HPC clusters often provide conda/CUDA via modules rather than direct install."
```bash
module avail conda 2>&1 | head -5
module avail cuda 2>&1 | head -5
```

---

## Step 3: Check if environment already exists

Before installing, check whether `conglude` already exists:

```bash
conda env list | grep conglude
```

**If the environment exists**, skip directly to the verification checks (Step 5). Only proceed to Step 4 (install) if one or more verification checks fail.

**If the environment does not exist**, proceed to Step 4.

---

## Step 4: Install environment

Explain to the user what the script will do before running it:

"I'm now running `setup_env.sh` with tag `<tag>`. This will:
1. Create a new conda environment called `conglude` with Python 3.11
2. Install PyTorch (+ torchvision, torchaudio) with the correct CUDA support
3. Install RDKit (chemistry toolkit for molecular fingerprints)
4. Install PyTorch Geometric and torch-scatter (for graph neural networks)
5. Fix a common Linux library path issue (libstdc++)
6. Install the ConGLUDe package itself in editable mode

This may take 5–10 minutes."

### Logging

Log the setup output so it can be reviewed if anything goes wrong:

```bash
mkdir -p logs
LOG_FILE="logs/setup_$(date +%Y-%m-%d_%H-%M-%S).log"
```

```bash
bash setup_env.sh cu121 2>&1 | tee "$LOG_FILE"    # for CUDA 12.1–12.7 drivers
bash setup_env.sh 2>&1 | tee "$LOG_FILE"          # default: cu128
```

After completion, tell the user: "Environment created. Activate it with `conda activate conglude`."

After the full setup is done (including verification in Step 5), tell the user where the log file is and ask whether to keep or remove it using `AskUserQuestion`.

---

## Step 5: Verify installation

Tell the user: "Let me verify that everything installed correctly by testing each core dependency."

Run these checks in sequence, and report each result clearly (pass/fail):

```bash
# 1. PyTorch and CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, version: {torch.version.cuda}')"

# 2. PyG and torch-scatter
python -c "import torch_geometric; import torch_scatter; print('PyG + torch-scatter OK')"

# 3. RDKit
python -c "from rdkit import Chem; print('RDKit OK')"

# 4. ESM
python -c "import esm; print('ESM OK')"

# 5. ConGLUDe full import
python -c "from conglude.model import ConGLUDeModel; print('ConGLUDe OK')"
```

After all checks pass, summarize: "All dependencies verified. Your ConGLUDe environment is ready."

If any check fails and the environment was pre-existing (came from Step 3), tell the user which checks failed and offer to reinstall: "The existing environment has issues. I'll re-run `setup_env.sh` to fix it." Then proceed to Step 4.

If any check fails after a fresh install (came from Step 4), explain what the failed component does and why it's needed before attempting the fix from the Troubleshooting section.

---

## Step 6: Download evaluation data (optional)

**Always ask the user** whether they want to download training and/or evaluation datasets. Explain: "ConGLUDe comes with training and benchmark datasets for reproducing paper results. These are only needed if you want to run evaluations or re-train the model. Would you like me to download them? I can get all datasets or specific ones."

If the user wants data:
```bash
python download_data.py                          # all datasets
python download_data.py --dataset_name litpcba   # single dataset
python download_data.py --dataset_name vs        # group alias (downloads dude + litpcba)
```

Individual datasets: `asd`, `coach420`, `dude`, `holo4k`, `kinobeads`, `litpcba`, `pdbbind_refined`, `pdbbind_time`, `posebusters`

Group aliases:

| Alias | Datasets |
|-------|----------|
| `test` | all 9 test datasets |
| `train` / `train_val` | `SB_train_val`, `LB_train_val` |
| `vs` | `dude`, `litpcba` |
| `tf` | `kinobeads` |
| `pp` | `coach420`, `holo4k`, `pdbbind_refined` |
| `pr` | `asd`, `pdbbind_time`, `posebusters` |

Already-downloaded datasets are automatically skipped.

### Processing SB_train_val

If `SB_train_val` is among the downloaded datasets, **ask the user** whether they would like to process it now. Explain: "SB_train_val needs to be processed before it can be used for training (extracting protein graphs, ligand features, etc.). This is done by `process_data.py` and can take a while as it computes ESM embeddings on GPU and builds protein graphs for ~23k proteins. Would you like me to run it now?"

If the user agrees, **first select a free GPU** — processing requires GPU for ESM embedding computation:

1. Check GPU utilization:
```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

2. Pick a GPU with low memory usage and low utilization, then export it:
```bash
export CUDA_VISIBLE_DEVICES=<free_gpu_id>
```

3. Run processing:
```bash
python process_data.py --dataset_name SB_train_val
```

---

## Troubleshooting

When a problem occurs, always explain to the user what went wrong in plain language before applying the fix.

### `libstdc++.so.6: version GLIBCXX_3.4.30 not found`

Tell the user: "This means your system's C++ standard library is older than what some packages expect. I'll install a compatible version inside the conda environment and make sure it's found first."

```bash
conda install -y libgcc-ng
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib
conda deactivate && conda activate conglude
```

### torch-scatter compiles from source and fails

Tell the user: "torch-scatter couldn't find a pre-built wheel matching your PyTorch + CUDA combination, so it tried to compile from source (which requires a C++ compiler and CUDA toolkit). Let me check what versions you have and find the right wheel."

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```
Then find the matching wheel at https://data.pyg.org/whl/

### ESM weight download hangs

Tell the user: "ESM model weights (~2.5 GB) are downloaded automatically on first use. If your system has no internet access, you'll need to pre-download them or point to a shared cache."

```bash
export TORCH_HOME=/path/to/shared/torch_cache
```

### PyG AttributeError after upgrade

Tell the user: "PyTorch Geometric is pinned to a specific version because newer versions change internal APIs that ConGLUDe relies on. Don't upgrade it."

### numpy version conflict

Tell the user: "You may see pip downgrade numpy during install — this is expected and correct. ConGLUDe pins numpy for reproducibility."

### `ModuleNotFoundError: No module named 'conglude'`

Tell the user: "The ConGLUDe package itself wasn't installed. I'll install it in editable mode so you can use it and modify the source code without reinstalling."

```bash
pip install -e .
```