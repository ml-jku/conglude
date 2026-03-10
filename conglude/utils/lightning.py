from typing import List
import torch
import wandb
import os
import json

from hydra.utils import instantiate, log
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint



def init_lightning_callbacks(
    cfg: DictConfig
) -> List[Callback]:
    """
    Initialize callbacks for pytorch lightning.

    Parameters
    ----------
    cfg: DictConfig
        The configuation for the callbacks.

    Returns
    -------
        List[Callback]: The callbacks
    """

    callbacks: List[Callback] = []

    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(instantiate(cb_conf))
    
    return callbacks



class CustomModelCheckpoint(ModelCheckpoint):
    """
    Extended PyTorch Lightning ModelCheckpoint callback with W&B-aware directory
    structure and additional component-wise checkpointing.

    Parameters
    ----------
    dirpath: Optional[str]
        Base directory where checkpoints should be stored. The current W&B run ID is automatically appended.
    **kwargs:
        Additional keyword arguments forwarded to `pytorch_lightning.callbacks.ModelCheckpoint`.
    """
    
    def __init__(
        self, 
        dirpath: str = None, 
        **kwargs: dict,
    ) -> None:
        
         # Get the current Weights & Biases run ID
        run_name = wandb.run.id

        # Append run ID to checkpoint directory
        if dirpath is None:
            dirpath = f"checkpoints/{run_name}"
        else:
            dirpath = f"{dirpath}/{run_name}"
        self.dirpath = dirpath
        
        super().__init__(dirpath=dirpath, **kwargs)


    def on_save_checkpoint(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule,
        checkpoint: dict,
    ) -> None:
        """
        Called when Lightning saves a checkpoint.
        In addition to the standard checkpoint file, this method saves individual model components (if they exist) 
        and stores their hyperparameters as JSON.
        """

        # Call the parent method to save the standard checkpoint
        super().on_save_checkpoint(trainer, pl_module, checkpoint)

        # Ensure checkpoint directory exists
        os.makedirs(self.dirpath, exist_ok=True)
        
        # Save VN-EGNN module if it exists
        if hasattr(pl_module, "vnegnn"):
            path = f"{self.dirpath}/vnegnn.pth"
            torch.save(pl_module.vnegnn.state_dict(), path)

            if hasattr(pl_module.vnegnn, "hparams"):
                path = f"{self.dirpath}/vnegnn_config.json"
                with open(path, "w") as file:
                    json.dump(pl_module.vnegnn.hparams, file)

        # Save pocket encoder module (linear projection) if it exists
        if hasattr(pl_module, "pocket_encoder"):
            path = f"{self.dirpath}/pocket_encoder.pth"
            torch.save(pl_module.pocket_encoder.state_dict(), path)

            if hasattr(pl_module.pocket_encoder, "hparams"):
                path = f"{self.dirpath}/pocket_encoder_config.json"
                with open(path, "w") as file:
                    json.dump(pl_module.pocket_encoder.hparams, file)

        # Save protein encoder module (linear projection) if it exists
        if hasattr(pl_module, "protein_encoder"):
            path = f"{self.dirpath}/protein_encoder.pth"
            torch.save(pl_module.protein_encoder.state_dict(), path)

            if hasattr(pl_module.protein_encoder, "hparams"):
                path = f"{self.dirpath}/protein_encoder_config.json"
                with open(path, "w") as file:
                    json.dump(pl_module.protein_encoder.hparams, file)

        # Save ligand encoder module if it exists
        if hasattr(pl_module, "ligand_encoder"):
            path = f"{self.dirpath}/ligand_encoder.pth"
            torch.save(pl_module.ligand_encoder.state_dict(), path)

            if hasattr(pl_module.ligand_encoder, "hparams"):
                path = f"{self.dirpath}/ligand_encoder_config.json"
                with open(path, "w") as file:
                    json.dump(pl_module.ligand_encoder.hparams, file)