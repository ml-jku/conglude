from hydra.utils import instantiate
from omegaconf import DictConfig
import hydra
import wandb
from pytorch_lightning.loggers import WandbLogger
import torch


@hydra.main(config_path="configs", config_name="eval", version_base="1.2")
def eval(cfg: DictConfig):
    
    torch.set_float32_matmul_precision(cfg.precision)
    
    datamodule = instantiate(cfg.datamodule, _recursive_=True)
    model = instantiate(cfg.model)

    logger = None
    trainer = instantiate(cfg.trainer, logger=None, callbacks=None)
  
    trainer.test(datamodule=datamodule, model=model)  
    if isinstance(logger, WandbLogger):
        wandb.finish()


if __name__ == "__main__":
    eval()
