---
name: ml-conglude-train
description: >
  Train the ConGLUDe model. Invoke ONLY when the user explicitly mentions
  "ConGLUDe" or "conglude" together with training intent — e.g. "train",
  "start training", "run train.py", "fine-tune", "retrain", "train the model",
  "debug training run". Do NOT trigger on evaluation, prediction, or embedding
  requests — use the eval or predict skills for those.
version: 1.0.0
---

# ConGLUDe — Training

Train the ConGLUDe model on structure-based (SB) and/or ligand-based (LB) data, with configurable task, test datasets, debug mode, and WandB logging.

---

## Inputs to collect

Use `AskUserQuestion` to gather these options. Items marked with a default can be skipped if the user doesn't specify them.

### 1. Task

Which task(s) to train on. This controls which training/validation data splits are loaded (e.g. `task=vs` loads `vs_train_protein_ids.txt` and `vs_val_protein_ids.txt` instead of the full `train_protein_ids.txt`). Loss functions are not affected — they are always active based on their weight being non-zero in the model config.

| Value | Description |
|-------|-------------|
| `all` | All tasks combined (default — trains on full SB + LB data) |
| `vs` | Virtual screening only |
| `tf` | Target fishing only |
| `pp` | Pocket prediction only |
| `pr` | Pocket ranking only |

Default: `vs`

### 2. Test datasets

Test datasets are selected **automatically** based on the chosen task. Do NOT ask the user — just use the mapping below:

| Task | Test datasets |
|------|---------------|
| `all` | `dude`, `litpcba`, `kinobeads`, `coach420`, `holo4k`, `pdbbind_refined`, `asd`, `pdbbind_time`, `posebusters` |
| `vs` | `dude`, `litpcba`, `kinobeads` |
| `pp` | `coach420`, `holo4k`, `pdbbind_refined` |
| `pr` | `asd`, `pdbbind_time`, `posebusters` |

After determining the test datasets, check which ones are already downloaded and automatically download any that are missing:

```bash
# Check which test datasets exist
ls data/datasets/test_datasets/
```

For each required dataset that is not present, download it:
```bash
python download_data.py --dataset_name <dataset_name>
```

Group aliases can be used for efficiency: `test` (all 9), `vs` (dude + litpcba), `tf` (kinobeads), `pp` (coach420 + holo4k + pdbbind_refined), `pr` (asd + pdbbind_time + posebusters).

### 3. Debug mode

| Option | Effect |
|--------|--------|
| Debug run | 4 epochs, 1 GPU, anomaly detection on, no callbacks, no logging. Fast sanity check. |
| Full training | Up to 500 epochs, early stopping (patience 50), model checkpointing, gradient clipping. |

Default: Full training

### 4. WandB logging

| Option | Effect |
|--------|--------|
| Enabled | Logs to WandB project "ConGLUDe". Requires `wandb login`. |
| Disabled | No logger — runs fully offline. |

Default: Enabled (for full training), Disabled (for debug)

Note: Debug mode always disables logging regardless of this setting.

---

## Pre-flight checks

Before launching training, always do these:

### Conda environment (required before any Python command)

Activate the `conglude` conda environment before running any Python command in this skill:

```bash
conda activate conglude
```

All `python` commands below assume this environment is active. In the Bash tool, activate via:

```bash
bash -c 'source /SW/python/miniconda3/x86_64/etc/profile.d/conda.sh && conda activate conglude && python -u ...'
```

Do NOT use `conda run -n conglude` — it buffers all stdout/stderr until the process exits, which prevents real-time logging to tee/log files.

### GPU setup (required before running)

Both ESM embedding computation (during data processing) and model training use GPU. Before running train.py, you **must** select a free GPU:

1. Check GPU utilization:
```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

2. Pick a GPU with low memory usage and low utilization, then export it:
```bash
export CUDA_VISIBLE_DEVICES=<free_gpu_id>
```

3. Always pass `trainer.devices=1` to avoid DDP multi-GPU mode. With `devices=auto` (the default), PyTorch Lightning spawns one process per visible GPU, which can cause issues with prediction CSV writes during post-training testing and complicates debugging. Single-device training is simpler and sufficient unless the user explicitly asks for multi-GPU.

For multi-GPU training (only if the user explicitly requests it):
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py task=all trainer.devices=4 trainer.strategy=ddp
```

### Check training data availability

After GPU is selected and `CUDA_VISIBLE_DEVICES` is exported, check that training data is ready. Processing requires GPU (ESM embeddings), so this must come after GPU setup.

**Important:** Both SB and LB training data are always required, regardless of the task parameter. ConGLUDe is a multi-task model — the `task` parameter only controls which protein ID splits are used, but all loss functions (and thus all data types) remain active. Only skip SB or LB data if the user explicitly asks to train on one data type alone.

```bash
# Check if SB_train_val directory exists at all
ls data/datasets/train_val_datasets/SB_train_val/ 2>/dev/null

# Check if it's been processed (graphs exist)
ls data/datasets/train_val_datasets/SB_train_val/processed/graphs/10_neighbors_10.0_cutoff/ 2>/dev/null | wc -l

# Check if LB_train_val exists
ls data/datasets/train_val_datasets/LB_train_val/processed/ligand_embeddings/ecfp4_2048.dat 2>/dev/null
```

**If SB_train_val directory does not exist** — download and then process it:
```bash
python download_data.py --dataset_name train_val
python process_data.py --dataset_name SB_train_val
```

**If SB_train_val directory exists but has no `processed/` subdirectory (or 0 graph files)** — it's downloaded but not yet processed. Process it:
```bash
python process_data.py --dataset_name SB_train_val
```
This takes a long time (computes ESM embeddings on GPU, builds protein graphs for ~23k proteins). Inform the user of the expected duration.

**If LB_train_val is missing** — download it:
```bash
python download_data.py --dataset_name train_val
```

---

## Configure test datasets

Before running, update `configs/datamodule/test_datasets/test_datasets.yaml` to reflect the user's selection. Comment out datasets not selected, uncomment selected ones.

The file format is:
```yaml
  _target_: conglude.datamodule.DatasetList
  defaults:
    - _self_
    # - default_vs: default_vs
    - litpcba: litpcba
    - dude: dude
    # - kinobeads: kinobeads
    # - pdbbind_time: pdbbind_time
    # - posebusters: posebusters
    # - asd: asd
    # - coach420: coach420
    # - holo4k: holo4k
    # - pdbbind_refined: pdbbind_refined
```

Uncomment the selected datasets and comment out the rest. Always show the user what you changed.

---

## Logging

All runs must be logged to a file so output is preserved for debugging and review. Before running any command, create the logs directory and set up a timestamped log file:

```bash
mkdir -p logs
LOG_FILE="logs/train_$(date +%Y-%m-%d_%H-%M-%S).log"
```

Append `2>&1 | tee "$LOG_FILE"` to every `python train.py` command so output goes to both the terminal and the log file.

After the run completes (success or failure), tell the user where the log file is and ask whether to keep or remove it using `AskUserQuestion`.

---

## Run training

### Debug run

```bash
export CUDA_VISIBLE_DEVICES=<free_gpu_id>
python train.py +debug=default task=<task> trainer.devices=1 model.save_predictions=false model.save_embeddings=false 2>&1 | tee "$LOG_FILE"
```

Debug mode overrides: 4 epochs, 1 device, anomaly detection, no callbacks, no logger.

### Full training — no WandB

```bash
export CUDA_VISIBLE_DEVICES=<free_gpu_id>
python train.py task=<task> ~logger trainer.devices=1 model.save_predictions=false model.save_embeddings=false 2>&1 | tee "$LOG_FILE"
```

The `~logger` syntax removes the logger config group entirely. Do NOT use `logger=null` — it causes a Hydra error ("Config group override must be a string or a list. Got NoneType").

### Full training — with WandB

```bash
export CUDA_VISIBLE_DEVICES=<free_gpu_id>
python train.py task=<task> trainer.devices=1 model.save_predictions=false model.save_embeddings=false 2>&1 | tee "$LOG_FILE"
```

If WandB is not logged in, run `wandb login` first or set `WANDB_API_KEY`.

### Common overrides

```bash
# Resume from checkpoint
python train.py task=all checkpoint_name=<name> trainer.devices=1

# Change seed
python train.py task=all seed=42 trainer.devices=1

# Reduce batch size (if OOM)
python train.py task=all datamodule.train_datasets.SB_train.batch_size=32 trainer.devices=1

# Change max epochs
python train.py task=all trainer.max_epochs=200 trainer.devices=1

# Adjust loss weights (format: seg/vnpos/conf/prank/prot/SBvs/LBvs)
python train.py task=all model.LB_virtual_screening_loss_weight=6.0 trainer.devices=1
```

---

## Monitor training

If running in the foreground, training progress is printed to stdout (PyTorch Lightning progress bar).

If WandB is enabled, the user can monitor at: https://wandb.ai/ (project: ConGLUDe)

Key metrics to watch:
- `avg_val/virtual_screening/bedroc` — primary validation metric (used for early stopping and checkpointing)
- `train_loss` — should decrease
- `learning_rate` — starts with warmup, then decays on plateau

---

## Output

After training completes:

1. **Best checkpoint** saved at: `checkpoints/ConGLUDe/` (as configured in `callbacks/default.yaml`)
2. **Test results** printed to stdout (evaluated using the best checkpoint)
3. **WandB run** (if enabled) — contains full training curves, model artifacts, and config

Report to the user:
- Where the best checkpoint was saved
- The final validation BEDROC
- Test set results summary
- WandB run URL (if applicable)

---

## Troubleshooting

### `AttributeError: 'ConGLUDeDataset' object has no attribute 'split'`

The `split` parameter must be stored as `self.split` in `ConGLUDeDataset.__init__`. Check that line `self.split = split` exists in `conglude/datamodule.py` after the `dataset_name` assignment.

### `CUDA out of memory`

Reduce batch size:
```bash
python train.py task=all datamodule.train_datasets.SB_train.batch_size=32 datamodule.train_datasets.LB_train.batch_size=8
```

Or use fewer GPUs with gradient accumulation:
```bash
python train.py task=all trainer.devices=1 trainer.accumulate_grad_batches=4
```

### `Config group override must be a string or a list. Got NoneType`

This happens when passing `logger=null` directly. Use `~logger` to remove the config group, or use `+debug=default` which already sets `logger: null` internally.

### WandB authentication error

```bash
wandb login
# or
export WANDB_API_KEY=<your_key>
```

### Test dataset not found / FileNotFoundError

The test dataset data hasn't been downloaded. Run:
```bash
python download_data.py --dataset_name <dataset>
```

### No training data / empty DataLoader

Check that both datasets exist and SB_train_val is processed:
```bash
ls data/datasets/train_val_datasets/SB_train_val/processed/graphs/10_neighbors_10.0_cutoff/ | wc -l
ls data/datasets/train_val_datasets/LB_train_val/info/protein_ids.txt
```
