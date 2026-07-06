import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from pytorch_lightning.loggers import WandbLogger
import torch

from conglude.utils.lightning import init_lightning_callbacks



@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def run(cfg: DictConfig):
    
    torch.set_float32_matmul_precision(cfg.precision)
    pl.seed_everything(cfg.seed, workers=True)

    # Resolve config to a plain dict for logging (e.g. WandB)
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model)

    # Build run name: [DEBUG_]<task>_<train_mode>_<loss_weight_flags>
    debug_flag = "DEBUG_" if cfg.debug else ""
    train_mode = "mixed" if len(datamodule.train_datasets) == 2 else datamodule.train_datasets[0].dataset_name.split("_")[0]
    loss_weights = f"{int(cfg.model.segmentation_loss_weight)}{int(cfg.model.vn_pos_loss_weight)}{int(cfg.model.confidence_loss_weight)}{int(cfg.model.pocket_ranking_loss_weight)}{int(cfg.model.protein_loss_weight)}{int(cfg.model.SB_virtual_screening_loss_weight)}{int(cfg.model.LB_virtual_screening_loss_weight)}"
    run_name = f"{debug_flag}{cfg.task}_{train_mode}_{loss_weights}"

    callbacks = init_lightning_callbacks(cfg)
    # `~logger` (Hydra config-group removal) drops the key entirely rather
    # than setting it to false/null, so fall back to Lightning's own default
    # logger in that case — the LearningRateMonitor callback in the
    # (non-debug) default callback set requires an active logger, so
    # `logger=False` is only viable when it (and model checkpointing) are
    # explicitly disabled, as in debug mode.
    cfg_logger = cfg.get("logger", True)
    if cfg_logger is False or cfg_logger is None:
        logger = False
    elif cfg_logger is True:
        logger = True
    else:
        logger = instantiate(cfg_logger)(config=config, name=run_name)
        if isinstance(logger, WandbLogger):
            logger.watch(model, log="all")

    trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    trainer.fit(model, datamodule=datamodule)
    # Evaluate on test set using the best checkpoint from training, or the
    # in-memory model when checkpointing is disabled (e.g. debug runs)
    ckpt_path = "best" if trainer.checkpoint_callback is not None else None
    trainer.test(datamodule=datamodule, model=model, ckpt_path=ckpt_path)
    if isinstance(logger, WandbLogger):
        wandb.finish()


if __name__ == "__main__":
    run()
