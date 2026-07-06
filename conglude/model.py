import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from itertools import groupby

from datetime import datetime
from typing import Any, Dict, Tuple
from torch_geometric.data import Batch
import yaml

from conglude.datamodule import MixedDataset
from conglude.utils.losses import VNPositionHuberLoss, DiceLoss, ConfidenceLoss, InfoNCELoss, BCELoss
from conglude.utils.metrics import VirtualScreeningMetrics, TargetFishingMetrics, PocketPredictionMetrics, PocketRankingMetrics
from conglude.utils.lr_schedulers import PlateauWithWarmup, CosineWithWarmup
from conglude.utils.common import write_list_to_txt
from conglude.utils.visualization import build_pymol_scene_script, create_pymol_scene
from conglude.modules.vnegnn import VNEGNN
from conglude.modules.mlp import MLPEncoder
from conglude.modules.cluster import DBSCANCluster



class ConGLUDeModel(pl.LightningModule):
    """
    This model jointly learns representations for proteins, binding pockets, and ligands in order to perform multiple structure-based and ligand-based
    drug discovery tasks (pocket prediction, pocket ranking, virtual screening, target fishing) within a single framework.

    Parameters
    ----------
    vnegnn: torch.nn.Module
        Geometric protein encoder that predicts pocket locations and residue segmentation using an equivariant GNN with virtual nodes (VN-EGNN).
    pocket_encoder: torch.nn.Module
        Projection network mapping pocket features to the contrastive embedding space.
    protein_encoder: torch.nn.Module
        Projection network mapping protein features to the contrastive embedding space.
    ligand_encoder: torch.nn.Module
        Neural network encoding ligand fingerprints and/or descriptors into embedding vectors.
    cluster: torch.nn.Module, optional
        Optional clustering module used to merge predicted pockets.
    optimizer: torch.optim.Optimizer
        Optimizer class used for training.
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler configuration.
    segmentation_loss: torch.nn.Module
        Loss for residue-level pocket segmentation.
    vn_pos_loss: torch.nn.Module
        Loss for predicting the positions of virtual nodes representing pockets.
    confidence_loss: torch.nn.Module
        Loss for estimating confidence scores of predicted pockets.
    pocket_ranking_loss: torch.nn.Module
        Contrastive loss used for ranking pockets relative to ligands.
    protein_loss: torch.nn.Module
        Contrastive loss aligning ligand and protein representations.
    SB_virtual_screening_loss: torch.nn.Module
        Structure-based virtual screening loss.
    LB_virtual_screening_loss: torch.nn.Module
        Ligand-based virtual screening loss.
    checkpoint_name: str, optional
        Name of a checkpoint from which pretrained weights should be loaded.
    checkpoint_path: str
        Directory containing pretrained checkpoints.
    num_pocket_nodes: int
        Number of virtual pocket nodes initialized per protein.
    protein_node: bool
        Whether protein-level representations are included in the contrastive model.
    save_predictions: bool
        If True, prediction outputs are saved during testing.
    save_embeddings: bool
        If True, learned embeddings are stored during testing.
    save_metrics: bool
        If True, virtual screening metrics per protein are stored during testing.
    """
    
    def __init__(
        self,

        vnegnn: torch.nn.Module,
        pocket_encoder: torch.nn.Module,
        protein_encoder: torch.nn.Module,
        ligand_encoder: torch.nn.Module,
        cluster: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = torch.optim.AdamW,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR = None,

        segmentation_loss: torch.nn.Module = DiceLoss(),
        segmentation_loss_weight: float = 1.0,
        vn_pos_loss: torch.nn.Module = VNPositionHuberLoss(),
        vn_pos_loss_weight: float = 1.0,
        confidence_loss: torch.nn.Module = ConfidenceLoss(gamma=4),
        confidence_loss_weight: float = 1.0,
        pocket_ranking_loss: torch.nn.Module = InfoNCELoss(temperature=0.0625),
        pocket_ranking_loss_weight: float = 1.0,
        protein_loss: torch.nn.Module = InfoNCELoss(temperature=0.0625),
        protein_loss_weight: float = 1.0,
        SB_virtual_screening_loss: torch.nn.Module = InfoNCELoss(temperature=0.0625),
        SB_virtual_screening_loss_weight: float = 1.0,
        LB_virtual_screening_loss: torch.nn.Module = BCELoss(scaling=1.0),
        LB_virtual_screening_loss_weight: float = 1.0,
        checkpoint_name: str = None,
        checkpoint_path: str = "checkpoints",
        
        num_pocket_nodes: int = 8,
        protein_node: bool = True,
        save_predictions: bool = False,
        save_embeddings: bool = False,
        save_metrics: bool = False,
        save_pymol_visualizations: bool = False,
    ):

        self.save_hyperparameters(
            logger=False,
            ignore=[
                "vnegnn",
                "pocket_encoder",
                "protein_encoder",
                "ligand_encoder",
                "cluster",
                "segmentation_loss",
                "vn_pos_loss",
                "confidence_loss",
                "pocket_ranking_loss",
                "protein_loss",
                "SB_virtual_screening_loss",
                "LB_virtual_screening_loss"
            ],
        )

        super(ConGLUDeModel, self).__init__()
       
        self.vnegnn = vnegnn
        if checkpoint_name is not None:
            try:
                vnegnn_state_dict = torch.load(f'{checkpoint_path}/{checkpoint_name}/vnegnn.pth', weights_only=True, map_location=self.device)
                self.vnegnn.load_state_dict(vnegnn_state_dict)
            except:
                print("Unable to load VN-EGNN weights.")
        
        self.ligand_encoder = ligand_encoder
        if checkpoint_name is not None:
            try:
                ligand_encoder_state_dict = torch.load(f'{checkpoint_path}/{checkpoint_name}/ligand_encoder.pth', weights_only=True, map_location=self.device)
                self.ligand_encoder.load_state_dict(ligand_encoder_state_dict)
            except:
                print("Unable to load ligand encoder weights.")

        self.pocket_encoder = pocket_encoder
        if checkpoint_name is not None:
            try:
                pocket_encoder_state_dict = torch.load(f'{checkpoint_path}/{checkpoint_name}/pocket_encoder.pth', weights_only=True, map_location=self.device)
                self.pocket_encoder.load_state_dict(pocket_encoder_state_dict)
            except:
                print("Unable to load pocket encoder weights.")

        self.protein_encoder = protein_encoder
        if checkpoint_name is not None:
            try:
                protein_encoder_state_dict = torch.load(f'{checkpoint_path}/{checkpoint_name}/protein_encoder.pth', weights_only=True, map_location=self.device)
                self.protein_encoder.load_state_dict(protein_encoder_state_dict)
            except:
                print("Unable to load protein encoder weights.")

        if cluster is not None:
            self.cluster = cluster

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.segmentation_loss = segmentation_loss
        self.segmentation_loss_weight = segmentation_loss_weight
        self.vn_pos_loss = vn_pos_loss
        self.vn_pos_loss_weight = vn_pos_loss_weight
        self.confidence_loss = confidence_loss
        self.confidence_loss_weight = confidence_loss_weight
        self.pocket_ranking_loss = pocket_ranking_loss
        self.pocket_ranking_loss_weight = pocket_ranking_loss_weight
        self.protein_loss = protein_loss
        self.protein_loss_weight = protein_loss_weight
        self.SB_virtual_screening_loss = SB_virtual_screening_loss
        self.SB_virtual_screening_loss_weight = SB_virtual_screening_loss_weight
        self.LB_virtual_screening_loss = LB_virtual_screening_loss
        self.LB_virtual_screening_loss_weight = LB_virtual_screening_loss_weight

        self.num_pocket_nodes = num_pocket_nodes
        self.protein_node = protein_node
        
        self.save_predictions = save_predictions
        self.save_embeddings = save_embeddings
        self.save_metrics = save_metrics
        self.save_pymol_visualizations = save_pymol_visualizations

        # # Setup pocket counters and metrics for evaluation
        # self.setup()

        
    def setup(self, stage):
        """
        Prepare pocket counters and metrics for evaluation.
        """

        self.pocket_counters = self.get_pocket_counters()
        self.initialize_metrics()


    def get_pocket_counters(self):
        """
        Collect pocket counters from all train, validation, and test dataloaders.

        Returns
        -------
        pocket_counters: dict
            Dictionary mapping dataloader names (str) to their associated `pocket_counter` dictionaries, across train, validation, and test splits.
        """

        dm = self.trainer.datamodule
        pocket_counters = {}

        if hasattr(dm, "train_dataloader") and dm.train_dataloader() is not None:
            dataset = dm.train_dataloader().dataset
            if dataset.dataset_name == "mixed_train":
                pocket_counters["SB_train"] = dataset.pocket_counter["SB_train"]
                pocket_counters["LB_train"] = dataset.pocket_counter["LB_train"]
            else:
                pocket_counters[dataset.dataset_name] = dataset.pocket_counter
        if hasattr(dm, "val_dataloader"):
            for dataloader in dm.val_dataloader():
                dataset = dataloader.dataset
                pocket_counters[dataset.dataset_name] = dataset.pocket_counter

        if hasattr(dm, "test_dataloader"):
            for dataloader in dm.test_dataloader():
                dataset = dataloader.dataset
                pocket_counters[dataset.dataset_name] = dataset.pocket_counter

        return pocket_counters
    
    
    def initialize_metrics(self):
        """
        Initialize evaluation metrics for different splits and tasks.
        """

        dm = self.trainer.datamodule

        # Initialize training dataset metrics
        if dm.train_dataloader() is None:
            self.metrics = {}

        # Split for structure- and ligand-based batches when using a mixed dataset
        elif isinstance(dm.train_dataloader().dataset, MixedDataset):
            self.metrics = {
                "SB_train": {
                    "virtual_screening": VirtualScreeningMetrics(ef_fractions=[0.05]),
                    "pocket_prediction": PocketPredictionMetrics(calc_iou=False),
                    "pocket_ranking": PocketRankingMetrics(),
                },

                "LB_train": {
                    "virtual_screening": VirtualScreeningMetrics(),
                },
            }

        # Otherwise, initialize metrics based on whether the training dataset is structure- or ligand-based
        elif dm.train_dataloader().dataset.structure_based:
            self.metrics = {
                dm.train_dataloader().dataset.dataset_name: {
                    "virtual_screening": VirtualScreeningMetrics(ef_fractions=[0.05]),
                    "pocket_prediction": PocketPredictionMetrics(calc_iou=False),
                    "pocket_ranking": PocketRankingMetrics(),
                }
            }
        else:
            self.metrics = {
                dm.train_dataloader().dataset.dataset_name: {
                    "virtual_screening": VirtualScreeningMetrics(),
                }
            }

        # Add metrics for each validation dataset based on whether it's structure- or ligand-based
        for dataloader in dm.val_dataloader():
            name = dataloader.dataset.dataset_name
            
            if dataloader.dataset.structure_based:
                self.metrics[name] = {
                    "virtual_screening": VirtualScreeningMetrics(ef_fractions=[0.05]),
                    "pocket_prediction": PocketPredictionMetrics(calc_iou=False),
                    "pocket_ranking": PocketRankingMetrics(),
                }
            else:
                self.metrics[name] = {
                    "virtual_screening": VirtualScreeningMetrics(),
                }

        # Add metrics for each test dataset based on its task
        for dataloader in dm.test_dataloader():
            name = dataloader.dataset.dataset_name
            task = dataloader.dataset.task

            save_per_target_csv = self.save_metrics

            if task == "vs":
                self.metrics[name] = {"virtual_screening": VirtualScreeningMetrics(save_per_target_csv=save_per_target_csv)}
            elif task == "tf":
                self.metrics[name] = {"target_fishing": TargetFishingMetrics(save_per_target_csv=save_per_target_csv)}
            elif task == "pp":
                self.metrics[name] = {"pocket_prediction": PocketPredictionMetrics(calc_iou=False, save_per_target_csv=save_per_target_csv)}
            elif task == "pr":
                self.metrics[name] = {"pocket_ranking": PocketRankingMetrics(save_per_target_csv=save_per_target_csv)}
            elif task == "all":
                self.metrics[name] = {
                    "virtual_screening": VirtualScreeningMetrics(save_per_target_csv=save_per_target_csv),
                    "target_fishing": TargetFishingMetrics(save_per_target_csv=save_per_target_csv),
                    "pocket_prediction": PocketPredictionMetrics(calc_iou=False, save_per_target_csv=save_per_target_csv),
                    "pocket_ranking": PocketRankingMetrics(save_per_target_csv=save_per_target_csv),
                }


    def on_train_epoch_start(
        self
    ) -> None:
        """
        Move training metrics to the correct device at the start of each epoch.
        Handles both MixedDataset (SB/LB metrics) and single-loader datasets.
        """

        # Retrieve the training dataloader
        train_dataloader = self.trainer.datamodule.train_dataloader()

        # Mixed dataset (multiple training sources)
        if isinstance(train_dataloader.dataset, MixedDataset):
            # Move structure-based metrics
            for task in self.metrics["SB_train"]:
                self.metrics["SB_train"][task].to(self.device)
            # Move ligand-based metrics
            for task in self.metrics["LB_train"]:
                self.metrics["LB_train"][task].to(self.device)

        # Single dataset loader        
        else:
            loader_name = train_dataloader.dataset.dataset_name

            for task in ["virtual_screening", "target_fishing", "pocket_prediction", "pocket_ranking"]:
                if task in self.metrics[loader_name]:
                    self.metrics[loader_name][task].to(self.device)


    def on_validation_epoch_start(
        self
    ) -> None:
        """
        Move validation metrics to the correct device at the start of each epoch.
        """

        # Retrieve all validation dataloaders
        val_dataloaders = self.trainer.datamodule.val_dataloader()

        # Iterate over each validation dataloader
        for val_dataloader in val_dataloaders:
            loader_name = val_dataloader.dataset.dataset_name

            # Move task-specific metrics to the correct device if they exist
            for task in [
                "virtual_screening",
                "target_fishing",
                "pocket_prediction",
                "pocket_ranking",
            ]:
                if task in self.metrics[loader_name]:
                    self.metrics[loader_name][task].to(self.device)


    def on_test_epoch_start(
        self
    ) -> None:
        """
        Move test metrics to the correct device and initialize
        storage tensors if predictions or embeddings must be saved.
        """

        # Retrieve all test dataloaders
        test_dataloaders = self.trainer.datamodule.test_dataloader()

        # Move task-specific metrics to the correct device
        for test_dataloader in test_dataloaders:
            loader_name = test_dataloader.dataset.dataset_name

            for task in [
                "virtual_screening",
                "target_fishing",
                "pocket_prediction",
                "pocket_ranking",
            ]:
                if task in self.metrics[loader_name]:
                    self.metrics[loader_name][task].to(self.device)

        # Initialize tensors for saving predictions/embeddings if required
        if self.save_predictions or self.save_embeddings:
            self.initialize_save_tensors()


    def training_step(
        self, 
        batch: Any,
        batch_idx: int
    ) -> torch.Tensor:
        """
        Perform a single training step.
        Handles both MixedDataset (LB + SB mixed batches) and standard single-dataset training.

        Parameters
        ----------
        batch: Any
            The batch returned by the dataloader.
            - If using MixedDataset: Tuple[data_batch, source]
            - Otherwise: data_batch
        batch_idx: int
            Index of the current batch within the epoch.

        Returns
        -------
        torch.Tensor
            The computed training loss for this batch.
        """

        train_dataloader = self.trainer.datamodule.train_dataloader()

        # Dataset specifics for mixed dataset with structure- and ligand-based batches
        if isinstance(train_dataloader.dataset, MixedDataset):

            batch, structure_based = batch
            dataset_specs = {
                "name": "SB_train" if structure_based else "LB_train",
                "task": "train",
                "structure_based": structure_based,
                "multi_pdb_targets": False
            }

        # Dataset specifics for standard single-dataset training
        else:
            dataset_specs = {
                "name": train_dataloader.dataset.dataset_name,
                "task": "train",
                "structure_based": train_dataloader.dataset.structure_based,
                "multi_pdb_targets": train_dataloader.dataset.multi_pdb_targets
            }

        # Process batch and compute loss
        loss = self.process_step(batch, dataset_specs)

        return loss


    def validation_step(
        self, 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ):
        """
        Perform a single validation step.
        Supports multiple validation dataloaders. 
        The `dataloader_idx` is automatically provided by PyTorch Lightning when multiple validation dataloaders are returned from the DataModule.

        Parameters
        ----------
        batch: Any
            The validation batch returned by the corresponding dataloader.
        batch_idx: int
            Index of the batch within the current validation epoch.
        dataloader_idx: int, optional
            Index of the active validation dataloader.
        """

        val_dataloader = self.trainer.datamodule.val_dataloader()[dataloader_idx]

        # Dataset specifics
        dataset_specs = {
            "name": val_dataloader.dataset.dataset_name,
            "task": "val",
            "structure_based": val_dataloader.dataset.structure_based,
            "multi_pdb_targets": val_dataloader.dataset.multi_pdb_targets
        }
        
        # Process batch
        self.process_step(batch, dataset_specs)


    def test_step(
        self, 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ):
        """
        Perform a single test step.
        Supports multiple test dataloaders. 
        The `dataloader_idx` is automatically provided by PyTorch Lightning when multiple test dataloaders are returned from the DataModule.

        Parameters
        ----------
        batch: Any
            The test batch returned by the corresponding dataloader.
        batch_idx: int
            Index of the batch within the current test epoch.
        dataloader_idx: int, optional
            Index of the active test dataloader.
        """

        test_dataloader = self.trainer.datamodule.test_dataloader()[dataloader_idx]

        # Dataset specifics
        dataset_specs = {
            "name": test_dataloader.dataset.dataset_name,
            "task": test_dataloader.dataset.task,
            "structure_based": test_dataloader.dataset.structure_based,
            "multi_pdb_targets": test_dataloader.dataset.multi_pdb_targets
        }
        
        # Process batch
        self.process_step(batch, dataset_specs)
    

    def on_train_epoch_end(
        self
    ) -> None:
        """
        Compute and log the accumulated training metrics.
        """

        train_dataloader = self.trainer.datamodule.train_dataloader()

        # Compute metrics separately for structure- and ligand-based training with MixedDataset
        if isinstance(train_dataloader.dataset, MixedDataset):
            self.compute_and_log_metrics("SB_train")
            self.compute_and_log_metrics("LB_train")
        
        # Compute metrics for standard single-dataset training
        else:
            loader_name = train_dataloader.dataset.dataset_name
            self.compute_and_log_metrics(loader_name)        


    def on_validation_epoch_end(
        self
    ) -> None:
        """
        Compute and log metrics individually per validation dataset 
        and aggregate metrics across all validation datasets, 
        logging their averages as "avg_val/<metric_name>"
        """

        val_dataloaders = self.trainer.datamodule.val_dataloader()

        all_val_metrics = {}
        # Compute metrics per validation dataloader
        for val_dataloader in val_dataloaders:
            loader_name = val_dataloader.dataset.dataset_name
            val_metrics = self.compute_and_log_metrics(loader_name)

            # Aggregate metrics across loaders
            for val_metric_name, val_metric_value in val_metrics.items():
                if val_metric_name in all_val_metrics:
                    all_val_metrics[val_metric_name].append(val_metric_value)
                else:
                    all_val_metrics[val_metric_name] = [val_metric_value]
        
        # Compute averaged validation metrics across all loaders and log them
        for metric_name, metric_values in all_val_metrics.items():
            avg_metric_value = sum(metric_values) / len(metric_values)
            self.log(f"avg_val/{metric_name}", avg_metric_value, sync_dist=True, add_dataloader_idx=False)

    
    def on_test_epoch_end(self) -> None:
        """
        Compute and log test metrics.
        Optionally saves predictions or embeddings per dataset.
        """

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        test_dataloaders = self.trainer.datamodule.test_dataloader()

        for test_dataloader in test_dataloaders:
            ds_name = test_dataloader.dataset.dataset_name
            ds_dir = test_dataloader.dataset.dataset_dir
            results_dir = self.get_results_dir(timestamp, ds_name, ds_dir)

            if self.save_metrics:
                metrics_path = os.path.join(results_dir, "metrics")
                os.makedirs(metrics_path, exist_ok=True)
            else:
                metrics_path = None

            self.compute_and_log_metrics(ds_name, metrics_path=metrics_path)

            if (self.save_predictions or self.save_embeddings) and ds_name in self._ds:
                self.save_results(ds_name, results_dir=results_dir)

            if self.save_pymol_visualizations and ds_name in self._ds:
                self._write_pymol_scenes(ds_name, results_dir)


    def get_results_dir(self, timestamp: str, dataset_name: str, dataset_dir: str) -> str:
        """
        Resolve the timestamped run directory for a single dataset.

        For repo datasets (dataset_dir under ./data/): results/<dataset_name>/<timestamp>/
        For external datasets: <dataset_dir>/ConGLUDe/results/<timestamp>/
        """
        if dataset_dir.startswith("./data/") or dataset_dir.startswith("data/"):
            results_dir = os.path.join("results", dataset_name, timestamp)
        else:
            results_dir = os.path.join(dataset_dir, "ConGLUDe", "results", timestamp)

        os.makedirs(results_dir, exist_ok=True)
        return results_dir


    def forward(
        self,
        proteins: Batch,
        ligands: torch.Tensor,
        ligand_batch_idx: torch.Tensor,
        ligand_idx: torch.Tensor,
        dataset_specs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Forward pass of the ConGLUDe model.

        Parameters
        ----------
        proteins: torch_geometric.data.Batch
            Batched protein input dictionary. Must contain:
        ligands: torch.Tensor
            Ligand input tensor of shape (N_ligands, ligand_feature_dim).
        ligand_batch_idx: torch.Tensor
            Tensor of shape (N_ligands,) mapping each ligand to itscorresponding protein index within the batch.
        ligand_idx: torch.Tensor
            Tensor containing ligand indices.
        dataset_specs : Dict[str, Any]
            Dictionary specifying dataset configuration. Expected keys: "task", "structure_based", "multi_pdb_targets"

        Returns
        -------
        Dict[str, Any]
            Output dictionary containing predictions, labels, indices and optionally embeddings.
        """

        output = {}

        # Extract pocket and protein features via VN-EGNN
        if dataset_specs["structure_based"]:
            pocket_feats, pocket_pos, protein_feats, _, _, residue_segm, confidence, _, _, _ = self.vnegnn(proteins)
        else: 
            # If not structure-based, don't backprop through structure module
            with torch.no_grad():
                pocket_feats, pocket_pos, protein_feats, _, _, residue_segm, confidence, _, _, _ = self.vnegnn(proteins)

        # Optional pocket clustering
        if hasattr(self, "cluster"):
            pocket_pos_rearranged = pocket_pos.view(len(proteins), self.num_pocket_nodes, -1)
            pocket_feats_rearranged = pocket_feats.view(len(proteins), self.num_pocket_nodes, -1)
            confidence_rearranged = confidence.view(len(proteins), self.num_pocket_nodes)

            pocket_pos_clustered, pocket_feats_clustered, confidence_clustered, pocket_batch_idx = self.cluster(pocket_pos_rearranged, pocket_feats_rearranged, confidence_rearranged)
            pocket_batch_idx = pocket_batch_idx.to(self.device)

        else:
            pocket_pos_clustered = pocket_pos
            pocket_feats_clustered = pocket_feats
            confidence_clustered = confidence

            pocket_batch_idx = torch.arange(len(proteins), device=self.device).repeat_interleave(self.num_pocket_nodes)
                    
        # Store raw and clustered pocket predictions
        output["predictions"] = {
            "pocket_pos": pocket_pos,
            "confidence": confidence,
            "pocket_pos_clustered": pocket_pos_clustered,
            "confidence_clustered": confidence_clustered,
            "residue_segm": residue_segm,
        }

        output["index"] = {"pocket_batch_idx": pocket_batch_idx}
        output["labels"] = {"pocket_centers": proteins["pocket_center"]}
       
        # Project protein and pocket embeddings to contrastive space and normalize
        encoded_pockets = self.pocket_encoder(pocket_feats_clustered)
        encoded_pockets = torch.nn.functional.normalize(encoded_pockets, dim=1)

        if self.save_embeddings:
            output["embeddings"] = {"encoded_pockets": encoded_pockets}
        
        if self.protein_node:
            encoded_proteins = self.protein_encoder(protein_feats)
            encoded_proteins = torch.nn.functional.normalize(encoded_proteins, dim=1)

            if self.save_embeddings:
                output["embeddings"]["encoded_proteins"] = encoded_proteins
        
            encoded_proteins_pockets = torch.cat((encoded_proteins[pocket_batch_idx], encoded_pockets), dim=1)

        # Encode ligands if not None
        if not ligands is None:
            encoded_ligands = self.ligand_encoder(ligands)

            # Split into protein-specific and binding-specific parts
            if self.protein_node:
                encoded_ligands_p = torch.nn.functional.normalize(encoded_ligands[:,:(encoded_ligands.shape[1]//2)], dim=1)
                encoded_ligands_b = torch.nn.functional.normalize(encoded_ligands[:,(encoded_ligands.shape[1]//2):], dim=1)
                encoded_ligands = torch.cat((encoded_ligands_p, encoded_ligands_b), dim=1)

            else:
                encoded_ligands = torch.nn.functional.normalize(encoded_ligands, dim=1)

            if self.save_embeddings:
                output["embeddings"]["encoded_ligands"] = encoded_ligands

        # Find closest pocket to ground-truth pocket center
        if dataset_specs["structure_based"]:
            
            # Map pocket centers to protein indices
            pocket_center_batch_idx = torch.tensor([i for i in range(len(proteins)) for _ in range(proteins[i]["pocket_center"].shape[0])], device=self.device)

            # Compute pairwise distances
            diffs = proteins["pocket_center"][:, None, :] - pocket_pos_clustered[None, :, :]
            dists = torch.norm(diffs, dim=-1)  # (N_pocket_centers, N_pockets)

            # Mask pockets belonging to different proteins
            mask = (pocket_batch_idx[None, :] != pocket_center_batch_idx[:, None]).to(self.device)
            dists = dists.masked_fill(mask, float("inf"))

            # Select closest valid pocket per center
            closest_pocket_idx_batch = torch.argmin(dists, dim=1)  # (N_pocket_centers,)
            closest_pocket_idx = (torch.arange(len(pocket_batch_idx), device=self.device) - torch.cat([torch.tensor([0], device=self.device), torch.bincount(pocket_batch_idx).cumsum(0)[:-1]])[pocket_batch_idx])[closest_pocket_idx_batch]

            output["labels"]["closest_pocket"] = closest_pocket_idx
            output["index"]["pocket_center_batch_idx"] = pocket_center_batch_idx

        # Pocket ranking predictions
        if (dataset_specs["task"] in ["train", "val"] and dataset_specs["structure_based"]) or dataset_specs["task"] in ["vs", "pr", "all"]:

            # Compute ligand–pocket similarity scores
            if self.protein_node:
                all_pocket_preds = encoded_pockets@(encoded_ligands_b.T)
            else:
                all_pocket_preds = encoded_pockets@(encoded_ligands.T)

            # For each ligand, select pockets from the same protein
            pocket_preds_per_ligand = [all_pocket_preds[pocket_batch_idx == ligand_batch_idx[i], i] for i in range(len(ligands))]

            # Number of pockets per ligand
            num_pockets = torch.tensor([len(b) for b in pocket_preds_per_ligand], device=self.device)
            max_pockets = num_pockets.max()

            # Initialize padded output
            pocket_preds = -100 * torch.ones((len(ligands), max_pockets), dtype=all_pocket_preds.dtype, device=self.device)

            # Fill tensor
            for ligand_idx, preds in enumerate(pocket_preds_per_ligand):
                pocket_preds[ligand_idx, :len(preds)] = preds

            output["predictions"]["pocket_preds"] = pocket_preds


        if dataset_specs["task"] in ["train", "val"] and dataset_specs["structure_based"] and self.protein_node:
            protein_preds = encoded_proteins@(encoded_ligands_p.T)

            output["predictions"]["protein_preds"] = protein_preds

        # Virtual screening predictions
        if dataset_specs["task"] in ["train", "val", "vs", "tf", "all"]:
            
            # Use concatenated protein+pocket representation
            if self.protein_node:
                all_vs_preds = encoded_proteins_pockets @ encoded_ligands.T
            # Use only pocket representation
            else:   
                all_vs_preds = encoded_pockets @ encoded_ligands.T

            # Use closest pocket per protein
            if dataset_specs["task"] == "train" and dataset_specs["structure_based"]:
                vs_preds = all_vs_preds[closest_pocket_idx_batch]
            
            # Max-pooling over pockets per protein
            elif (dataset_specs["task"] == "val" and dataset_specs["structure_based"]) or not self.protein_node:
                pocket_batch_idx_exp = pocket_batch_idx[:, None].expand(-1, all_vs_preds.size(1))
                vs_preds, _ = scatter_max(all_vs_preds, pocket_batch_idx_exp, dim=0)

            # Protein-only similarity
            else:
                vs_preds = encoded_proteins@(encoded_ligands_p.T)

            # Average predictions across multiple PDBs if requested
            if dataset_specs["multi_pdb_targets"]:
                vs_preds = torch.mean(vs_preds, dim=0)

            # Non-structure-based case: regroup per ligand
            elif not dataset_specs["structure_based"]:
                vs_preds = torch.cat([vs_preds[i, ligand_batch_idx==i] for i in range(len(proteins))])

            output["predictions"]["vs_preds"] = vs_preds

        return output
    

    def process_step(
        self,
        batch: Tuple[Batch, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        dataset_specs: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Execute one full training/validation/test step.

        Parameters
        ----------
        batch: Tuple
            Batch data consisting of:
            - proteins: batched protein graph object
            - ligands: batched ligand tensor
            - labels: ground-truth VS labels (LB case)
            - ligand_batch_idx: maps each ligand to its protein index
            - ligand_idx: global ligand indices
        dataset_specs : Dict[str, Any]
            Dictionary specifying dataset configuration. Expected keys: "name", "task", "structure_based", "multi_pdb_targets"

        Returns
        -------
        torch.Tensor
            Total loss (0 for test mode)
        """

        # Unpack batch and run forward pass
        proteins, ligands, labels, ligand_batch_idx, ligand_idx = batch
        output = self.forward(proteins, ligands, ligand_batch_idx, ligand_idx, dataset_specs)

        loss_dict = {}
        loss = 0

        # Attach additional data if predictions are to be saved (test mode)
        if dataset_specs["task"] not in ["train", "val"] and self.save_predictions:
            if dataset_specs["task"] in ["vs", "tf", "pr"]:
                output["index"]["ligand_batch_idx"] = ligand_batch_idx
                output["index"]["ligand_idx"] = ligand_idx
            if dataset_specs["task"] in ["vs", "tf"]:
                output["labels"]["vs_labels"] = labels

        if dataset_specs["task"] not in ["train", "val"] and (self.save_predictions or self.save_embeddings):
            output["protein_names"] = proteins.name
                          
        if dataset_specs["structure_based"]:
        
            # Segmentation loss
            if dataset_specs["task"] in ["train", "val"] and self.segmentation_loss_weight != 0:
                segmentation_loss = self.segmentation_loss(output["predictions"]["residue_segm"].squeeze(), proteins["residue"].y)
                loss_dict["segmentation_loss"] = segmentation_loss

                loss += self.segmentation_loss_weight * segmentation_loss                
            
            # Virtual node position loss
            if dataset_specs["task"] in ["train", "val"] and self.vn_pos_loss_weight != 0:
                (vn_pos_loss, _,) = self.vn_pos_loss(true_positions=proteins["pocket_center"],
                                                    pred_vn_positions=output["predictions"]["pocket_pos"],
                                                    vn_batch_index=proteins["pocket"].batch)
                
                loss_dict["vn_pos_loss"] = vn_pos_loss

                loss += self.vn_pos_loss_weight * vn_pos_loss
            
            # Confidence loss
            if dataset_specs["task"] in ["train", "val"] and self.confidence_loss_weight != 0:
                pocket_pos_rearranged = output["predictions"]["pocket_pos"].view(len(proteins), self.num_pocket_nodes, -1)
                
                pocket_dists = torch.norm(proteins["pocket_center"].unsqueeze(1) - pocket_pos_rearranged, dim=-1)

                confidence_loss = self.confidence_loss(pocket_dists.view(-1), output["predictions"]["confidence"])
                loss_dict["confidence_loss"] = confidence_loss
                
                loss += self.confidence_loss_weight * confidence_loss

            # Update pocket prediction metrics
            if "pocket_prediction" in self.metrics[dataset_specs["name"]]:
                
                if len(proteins["pocket_center"]) > len(proteins):
                    pocket_counts = output["index"]["pocket_center_batch_idx"]

                else:
                    pocket_counter = self.pocket_counters[dataset_specs["name"]]
                    pocket_counts = torch.tensor([pocket_counter[name.split("_")[0]] for name in proteins.name])
                    
                self.metrics[dataset_specs["name"]]["pocket_prediction"].update(
                    pocket_pos_clustered = output["predictions"]["pocket_pos_clustered"],
                    confidence_clustered = output["predictions"]["confidence_clustered"],
                    pocket_batch_idx = output["index"]["pocket_batch_idx"],
                    pocket_centers = proteins["pocket_center"],
                    pocket_center_batch_idx = output["index"]["pocket_center_batch_idx"],
                    pocket_counts = pocket_counts,
                    ligand_coords=proteins["ligand"].ligand_coordinates,
                    ligand_batch_index=proteins["ligand"].batch,
                    ligand_inds = proteins["ligand"].indices,
                    pred_segm = torch.sigmoid(output["predictions"]["residue_segm"]).squeeze(),
                    y_segm  = proteins["residue"].y.int(),
                    protein_names = proteins.name,
                )                    
            
            # Derive pocket ranking labels
            if (dataset_specs["task"] in ["train", "val"] and self.pocket_ranking_loss_weight != 0) or "pocket_ranking" in self.metrics[dataset_specs["name"]]:
                
                pocket_labels = torch.zeros_like(output["predictions"]["pocket_preds"], device=self.device)
                pocket_labels[torch.arange(len(ligands)), output["labels"]["closest_pocket"]] = 1
                pocket_labels[output["predictions"]["pocket_preds"]==-100] = -100

            # Pocket ranking loss
            if dataset_specs["task"] in ["train", "val"] and self.pocket_ranking_loss_weight != 0:
                pocket_ranking_loss = self.pocket_ranking_loss(output["predictions"]["pocket_preds"], pocket_labels) 
                loss_dict["pocket_ranking_loss"] = pocket_ranking_loss

                loss += self.pocket_ranking_loss_weight * pocket_ranking_loss
            
            # Update pocket ranking metrics
            if "pocket_ranking" in self.metrics[dataset_specs["name"]]:
                
                self.metrics[dataset_specs["name"]]["pocket_ranking"].update(
                    output["predictions"]["pocket_pos_clustered"],
                    output["predictions"]["confidence_clustered"],
                    output["index"]["pocket_batch_idx"],
                    output["predictions"]["pocket_preds"],
                    ligand_batch_idx,
                    ligand_idx,
                    proteins["pocket_center"],
                    output["index"]["pocket_center_batch_idx"],
                    protein_names=proteins.name,
                )
                    
            # Calculate molecule-to-protein loss
            if (dataset_specs["task"] in ["train", "val"] and self.protein_loss_weight != 0) or "target_fishing" in self.metrics[dataset_specs["name"]]: 

                protein_labels = torch.eye(output["predictions"]["protein_preds"].shape[0], device = self.device)

                if self.protein_loss_weight != 0:
                    protein_loss = self.protein_loss(output["predictions"]["protein_preds"], protein_labels)
                    loss_dict["protein_loss"] = protein_loss
                    
                    loss += self.protein_loss_weight * protein_loss
                
                # Update target fishing metrics
                if "target_fishing" in self.metrics[dataset_specs["name"]]:
                    self.metrics[dataset_specs["name"]]["target_fishing"].update(output["predictions"]["protein_preds"], protein_labels, ligand_idx.unsqueeze(0).repeat(len(ligand_idx), 1))


        # Virtual screening
        if dataset_specs["task"] in ["train", "val"] or "virtual_screening" in self.metrics[dataset_specs["name"]] or "target_fishing" in self.metrics[dataset_specs["name"]]:

            vs_preds = output["predictions"]["vs_preds"]

            # Calculate virtual screening loss
            if dataset_specs["structure_based"]:
                assert vs_preds.shape[0] == vs_preds.shape[1]
                labels = torch.eye(vs_preds.shape[0], device = self.device).long()
                ligand_batch_idx = torch.arange(labels.shape[0]).unsqueeze(1).expand(-1, labels.shape[1])

                # Protein+pocket to molecule loss for structure-based training
                if dataset_specs["task"] in ["train", "val"] and self.SB_virtual_screening_loss_weight != 0:
                    SB_virtual_screening_loss = self.SB_virtual_screening_loss(vs_preds, labels)
                    loss_dict["SB_virtual_screening_loss"] = SB_virtual_screening_loss
                    
                    loss += self.SB_virtual_screening_loss_weight * SB_virtual_screening_loss

            else:
                # Ligand-based virtual screening loss
                if dataset_specs["task"] in ["train", "val"] and self.LB_virtual_screening_loss_weight != 0:
                    LB_virtual_screening_loss = self.LB_virtual_screening_loss(vs_preds, labels) # , ligand_batch_idx)
                    loss_dict["LB_virtual_screening_loss"] = LB_virtual_screening_loss
                    
                    loss += self.LB_virtual_screening_loss_weight * LB_virtual_screening_loss

            # Update virtual screening metrics
            if "virtual_screening" in self.metrics[dataset_specs["name"]]:
                if dataset_specs["multi_pdb_targets"]:
                    self.metrics[dataset_specs["name"]]["virtual_screening"].update(vs_preds, labels, ligand_batch_idx, [proteins.name[0]])
                else:
                    self.metrics[dataset_specs["name"]]["virtual_screening"].update(vs_preds, labels, ligand_batch_idx, proteins.name)

        # Update target fishing metrics in ligand-based setting
        if dataset_specs["structure_based"] == False and "target_fishing" in self.metrics[dataset_specs["name"]]:

            self.metrics[dataset_specs["name"]]["target_fishing"].update(vs_preds, labels, ligand_idx)
                
        # Log losses for training and validation
        if dataset_specs["task"] in ["train", "val"]:
                         
            loss_dict["total_loss"] = loss

            batch_size = len(proteins) if not dataset_specs["multi_pdb_targets"] else 1
            self.log_losses(loss_dict, dataset_specs, batch_size)

        # Optionally save test predictions or embeddings
        else:
            if self.save_predictions or self.save_embeddings:
                self.update_save_lists(output, dataset_specs["task"], dataset_specs["name"])
        
        return loss


    def initialize_save_tensors(self) -> None:
        """
        Initialize per-dataset containers for storing predictions, embeddings, and metadata during testing.
        """
        self._ds = {}
        self._ds_offsets = {}
        self._ds_tasks = {}

    def _init_dataset_containers(self, dataset_name: str, task: str) -> None:
        """Lazily initialize containers for a dataset on first batch encounter."""
        d = {"protein_names": [], "pocket_names": []}

        if self.save_predictions:
            if task in ("pp", "pr"):
                d.update(pocket_pos=[], confidence=[], pocket_batch_idx=[])
            if task == "pr":
                d.update(pocket_centers=[], pocket_center_batch_idx=[],
                         pocket_preds=[], pr_ligand_idx=[], pr_ligand_batch_idx=[])
            if task in ("vs", "tf"):
                d.update(vs_preds=[], vs_labels=[], vs_protein_names=[], vs_ligand_idx=[])

        if self.save_embeddings:
            d["pocket_embeddings"] = []
            if self.protein_node:
                d.update(protein_embeddings=[], ligand_embeddings_p=[])
            d["ligand_embeddings_b"] = []

        self._ds[dataset_name] = d
        self._ds_offsets[dataset_name] = 0
        self._ds_tasks[dataset_name] = task


    def update_save_lists(
        self,
        output: dict,
        task: str,
        dataset_name: str,
    ) -> None:
        """
        Accumulate batch-wise outputs into per-dataset containers.

        Parameters
        ----------
        output: dict
            Dictionary returned by `forward(...)` and `process_step(...)`.
        task: str
            Dataset task type ("pp", "pr", "vs", "tf").
        dataset_name: str
            Name of the dataset this batch belongs to.
        """

        if dataset_name not in self._ds:
            self._init_dataset_containers(dataset_name, task)

        ds = self._ds[dataset_name]

        def append_if_exists(container, dictionary, key, detach=True):
            if key not in dictionary:
                return
            x = dictionary[key]
            if detach and torch.is_tensor(x):
                x = x.detach()
            container.append(x)

        offset = self._ds_offsets[dataset_name]

        if task in ("pp", "pr") or self.save_embeddings:
            ds["protein_names"].extend(output["protein_names"])
            ds["pocket_names"].extend([f"{output['protein_names'][j]}_pocket_{k}" for j, protein in groupby(output["index"]["pocket_batch_idx"].cpu().tolist()) for k, _ in enumerate(protein, start=1)])

        if self.save_predictions:

            if task in ("pp", "pr"):
                append_if_exists(ds["pocket_pos"], output["predictions"], "pocket_pos_clustered")
                append_if_exists(ds["confidence"], output["predictions"], "confidence_clustered")

                if "pocket_batch_idx" in output["index"]:
                    ds["pocket_batch_idx"].append(output["index"]["pocket_batch_idx"].detach() + offset)

            if task == "pr":
                append_if_exists(ds["pocket_centers"], output["labels"], "pocket_centers")
                if "pocket_center_batch_idx" in output["index"]:
                    ds["pocket_center_batch_idx"].append(output["index"]["pocket_center_batch_idx"].detach() + offset)
                if "pocket_preds" in output["predictions"]:
                    preds = output["predictions"]["pocket_preds"].detach()
                    preds = F.pad(preds, (0, self.num_pocket_nodes - preds.shape[1]), value=-100)
                    ds["pocket_preds"].append(preds)

                append_if_exists(ds["pr_ligand_idx"], output["index"], "ligand_idx")

                if "ligand_batch_idx" in output["index"]:
                    ds["pr_ligand_batch_idx"].append(output["index"]["ligand_batch_idx"].detach() + offset)

            if task in ("vs", "tf"):
                append_if_exists(ds["vs_preds"], output["predictions"], "vs_preds")
                append_if_exists(ds["vs_labels"], output["labels"], "vs_labels")
                append_if_exists(ds["vs_ligand_idx"], output["index"], "ligand_idx")

                if "ligand_batch_idx" in output["index"]:
                    ds["vs_protein_names"].extend([output["protein_names"][idx] for idx in output["index"]["ligand_batch_idx"].cpu().tolist()])

            if task in ("pp", "pr"):
                if "pocket_center_batch_idx" in output["index"]:
                    self._ds_offsets[dataset_name] += output["index"]["pocket_center_batch_idx"].max().item() + 1
                else:
                    self._ds_offsets[dataset_name] += len(output["protein_names"])

        if self.save_embeddings:
            append_if_exists(ds["pocket_embeddings"], output["embeddings"], "encoded_pockets")

            if self.protein_node:
                append_if_exists(ds["protein_embeddings"], output["embeddings"], "encoded_proteins")
                if "encoded_ligands" in output["embeddings"]:
                    encoded_ligands = output["embeddings"]["encoded_ligands"].detach()
                    ds["ligand_embeddings_p"].append(encoded_ligands[:, :encoded_ligands.shape[1]//2])
                    ds["ligand_embeddings_b"].append(encoded_ligands[:, encoded_ligands.shape[1]//2:])
            else:
                append_if_exists(ds["ligand_embeddings_b"], output["embeddings"], "encoded_ligands")


    def save_results(
        self,
        dataset_name: str,
        results_dir: str,
    ) -> None:
        """
        Save accumulated predictions and/or embeddings for a single dataset.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset whose accumulated data should be saved.
        results_dir : str
            Output directory for this dataset's results.
        """

        ds = self._ds[dataset_name]

        if self.save_predictions:

            predictions_path = os.path.join(results_dir, "predictions")
            os.makedirs(predictions_path, exist_ok=True)

            # pp predictions
            if ds.get("pocket_pos"):
                pocket_pos = torch.cat(ds["pocket_pos"], dim=0).cpu().numpy()
                confidence = torch.cat(ds["confidence"], dim=0).cpu().numpy()

                pp_df = pd.DataFrame(pocket_pos, columns=["pred_x", "pred_y", "pred_z"])
                pp_df["confidence"] = confidence
                pp_df["pocket_name"] = ds["pocket_names"]
                pp_df["protein_name"] = pp_df["pocket_name"].str.split("_").str[0]

                pp_df = pp_df[["protein_name", "pocket_name", "pred_x", "pred_y", "pred_z", "confidence"]]
                pp_df.to_csv(os.path.join(predictions_path, "pp_predictions.csv"), index=False)

            # pr predictions
            if ds.get("pocket_preds"):
                pocket_preds = torch.cat(ds["pocket_preds"], dim=0).cpu().numpy()
                pocket_batch_idx = torch.cat(ds["pocket_batch_idx"], dim=0).cpu().numpy()
                pocket_pos = torch.cat(ds["pocket_pos"], dim=0).cpu().numpy()
                confidence = torch.cat(ds["confidence"], dim=0).cpu().numpy()
                pocket_names = np.array(ds["pocket_names"])
                protein_names = ds["protein_names"]

                pr_ligand_idx = torch.cat(ds["pr_ligand_idx"], dim=0).cpu().numpy() if ds["pr_ligand_idx"] else None
                pr_ligand_batch_idx = torch.cat(ds["pr_ligand_batch_idx"], dim=0).cpu().numpy() if ds["pr_ligand_batch_idx"] else None

                has_labels = bool(ds.get("pocket_centers")) and bool(ds.get("pocket_center_batch_idx"))

                if has_labels:
                    pocket_centers = torch.cat(ds["pocket_centers"], dim=0).cpu().numpy()
                    pocket_center_batch_idx = torch.cat(ds["pocket_center_batch_idx"], dim=0).cpu().numpy()

                rows = []

                if has_labels:
                    for i in range(pocket_centers.shape[0]):
                        protein_idx = pocket_center_batch_idx[i].item()
                        protein_name = protein_names[protein_idx]
                        target_xyz = pocket_centers[i]
                        ligand = pr_ligand_idx[i].item()
                        scores = pocket_preds[i]

                        for pred_idx, score in enumerate(scores):
                            if score == -100:
                                continue
                            protein_mask = pocket_batch_idx == protein_idx
                            pred_xyz = pocket_pos[protein_mask][pred_idx]
                            pocket_name = pocket_names[protein_mask][pred_idx]
                            conf = confidence[protein_mask][pred_idx].item()
                            dist = np.linalg.norm(pred_xyz - target_xyz).item()

                            rows.append({
                                "protein_name": protein_name,
                                "pocket_name": pocket_name,
                                "ligand_idx": ligand,
                                "pred_x": pred_xyz[0].item(),
                                "pred_y": pred_xyz[1].item(),
                                "pred_z": pred_xyz[2].item(),
                                "target_x": target_xyz[0].item(),
                                "target_y": target_xyz[1].item(),
                                "target_z": target_xyz[2].item(),
                                "confidence": conf,
                                "conglude_score": score.item(),
                                "distance": dist
                            })
                else:
                    for i in range(pocket_preds.shape[0]):
                        protein_idx = pr_ligand_batch_idx[i].item() if pr_ligand_batch_idx is not None else i
                        protein_name = protein_names[protein_idx]
                        ligand = pr_ligand_idx[i].item() if pr_ligand_idx is not None else i
                        scores = pocket_preds[i]

                        for pred_idx, score in enumerate(scores):
                            if score == -100:
                                continue
                            protein_mask = pocket_batch_idx == protein_idx
                            pred_xyz = pocket_pos[protein_mask][pred_idx]
                            pocket_name = pocket_names[protein_mask][pred_idx]
                            conf = confidence[protein_mask][pred_idx].item()

                            rows.append({
                                "protein_name": protein_name,
                                "pocket_name": pocket_name,
                                "ligand_idx": ligand,
                                "pred_x": pred_xyz[0].item(),
                                "pred_y": pred_xyz[1].item(),
                                "pred_z": pred_xyz[2].item(),
                                "confidence": conf,
                                "conglude_score": score.item(),
                            })

                pr_df = pd.DataFrame(rows)
                pr_df.to_csv(os.path.join(predictions_path, "pr_predictions.csv"), index=False)

            # vs predictions
            if ds.get("vs_preds"):
                vs_preds = torch.cat(ds["vs_preds"], dim=0).cpu().numpy()
                vs_labels = torch.cat(ds["vs_labels"], dim=0).cpu().numpy()
                vs_ligand_idx = torch.cat(ds["vs_ligand_idx"], dim=0).cpu().numpy() if ds["vs_ligand_idx"] else None

                vs_df = pd.DataFrame({
                    "protein_name": ds["vs_protein_names"],
                    "ligand_idx": vs_ligand_idx,
                    "vs_pred": vs_preds,
                    "vs_label": vs_labels,
                })
                vs_df.to_csv(os.path.join(predictions_path, "vs_predictions.csv"), index=False)

        if self.save_embeddings:

            embeddings_path = os.path.join(results_dir, "embeddings")
            os.makedirs(embeddings_path, exist_ok=True)

            write_list_to_txt(os.path.join(embeddings_path, "protein_names.txt"), ds["protein_names"])
            write_list_to_txt(os.path.join(embeddings_path, "pocket_names.txt"), ds["pocket_names"])

            pocket_embeddings = torch.cat(ds["pocket_embeddings"], dim=0)
            np.save(os.path.join(embeddings_path, "pocket_embeddings.npy"), pocket_embeddings)

            if self.protein_node:
                protein_embeddings = torch.cat(ds["protein_embeddings"], dim=0)
                np.save(os.path.join(embeddings_path, "protein_embeddings.npy"), protein_embeddings)
                ligand_embeddings_p = torch.cat(ds["ligand_embeddings_p"], dim=0)
                np.save(os.path.join(embeddings_path, "ligand_embeddings_p.npy"), ligand_embeddings_p)

            ligand_embeddings_b = torch.cat(ds["ligand_embeddings_b"], dim=0)
            np.save(os.path.join(embeddings_path, "ligand_embeddings_b.npy"), ligand_embeddings_b)


    def _write_pymol_scenes(self, dataset_name: str, results_dir: str) -> None:
        """Save PyMOL-ready scenes for pocket visualization."""

        ds = self._ds.get(dataset_name, {})
        if not self.save_predictions or not ds.get("pocket_pos"):
            return

        pocket_pos = torch.cat(ds["pocket_pos"], dim=0).cpu().numpy()
        confidence = torch.cat(ds["confidence"], dim=0).cpu().numpy()

        visualization_root = os.path.join(results_dir, "predictions", "pymol_visualizations")

        pp_df = pd.DataFrame({
            "protein_name": ds["protein_names"],
            "pocket_name": ds["pocket_names"],
            "pred_x": pocket_pos[:, 0],
            "pred_y": pocket_pos[:, 1],
            "pred_z": pocket_pos[:, 2],
            "confidence": confidence,
        })

        for protein_name, protein_df in pp_df.groupby("protein_name", sort=False):
            protein_pdb_path = self._find_protein_pdb(protein_name)
            if protein_pdb_path is None:
                continue

            create_pymol_scene(
                protein_name=protein_name,
                protein_pdb=protein_pdb_path,
                pocket_df=protein_df,
                output_dir=os.path.join(visualization_root, protein_name),
            )


    def _find_protein_pdb(self, protein_name: str) -> str:
        """Return the cleaned PDB path for a protein if it exists."""

        for dataloader in self.trainer.datamodule.test_dataloader():
            data_dir = getattr(dataloader.dataset, "data_dir", None)
            if data_dir is None:
                continue

            candidate_path = os.path.join(data_dir, "processed", "cleaned_pdbs", protein_name, "protein.pdb")
            if os.path.isfile(candidate_path):
                return candidate_path

        return None
   

    def compute_and_log_metrics(
        self,
        dataset_name: str,
        metrics_path=None
    ) -> dict:
        """
        Compute, log, and reset all metrics associated with a given dataset.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset whose metrics should be computed and logged, must be a key in `self.metrics`.
        metrics_path : str, optional
            Directory path for saving metrics CSVs. Only passed to metrics that support it.
        """

        all_metrics = {}

         # Iterate over all possible tasks
        for task in ["virtual_screening", "target_fishing", "pocket_prediction", "pocket_ranking"]:

            # Only compute metrics if this dataset supports the given task
            if task in self.metrics[dataset_name]:

                # Compute aggregated metric values (pass metrics_path to all metrics that support it)
                if metrics_path is not None:
                    metrics_dict = self.metrics[dataset_name][task].compute(metrics_path)
                else:
                    metrics_dict = self.metrics[dataset_name][task].compute()

                # Log each metric individually
                for metric, result in metrics_dict.items():
                    all_metrics[f"{task}/{metric}"] = result
                    self.log(
                        f"{dataset_name}/{task}/{metric}",
                        result,
                        sync_dist=True,
                        add_dataloader_idx=False,
                    )

                # Reset metric state after logging
                self.metrics[dataset_name][task].reset()

        # Save aggregate summary CSV
        if metrics_path is not None and all_metrics:
            import csv
            os.makedirs(metrics_path, exist_ok=True)
            summary_path = os.path.join(metrics_path, "summary.csv")
            with open(summary_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                for metric, value in all_metrics.items():
                    writer.writerow([metric, float(value)])

        return all_metrics

        
    def log_losses(
        self,
        loss_dict: dict,
        dataset_specs: dict,
        batch_size: int,
    ) -> None:
        """
        Log loss values and (optionally) learnable loss parameters.

        Parameters
        ----------
        loss_dict: dict
            Dictionary mapping loss names (str) to scalar loss tensors.
        dataset_specs : dict
            Dictionary describing the current dataset split. Expected keys: "name" and "task".
        batch_size : int
            Effective batch size used for logging normalization in Lightning.
        """
        # Log losses
        for loss_name, loss in loss_dict.items():
            if loss is not None:
                self.log(f"{dataset_specs['name']}/losses/{loss_name}", loss, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
 
        # Log learnable loss parameters
        if dataset_specs["task"] == "train":
            for loss_name in loss_dict:

                # Retrieve loss module from the LightningModule
                loss_fn = getattr(self, loss_name, None)
                if loss_fn is None:
                    continue 

                # Only inspect torch modules 
                if isinstance(loss_fn, torch.nn.Module):
                    for pname, param in loss_fn.named_parameters(recurse=True):
                        if param.requires_grad:
                            # If scalar parameter log value directly, if tensor parameter log its norm for stability
                            self.log(
                                f"parameters/{loss_name}/{pname}",
                                param.detach().cpu().item() if param.numel() == 1 else param.norm().detach().cpu().item(),
                                on_step=True,
                                on_epoch=False,
                                sync_dist=True,
                            )

            
    def configure_optimizers(
        self
    ) -> Dict[str, Any]:
        """
        Configure the optimizer and optional learning rate scheduler for PyTorch Lightning.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
                - "optimizer": torch optimizer instance
                - "lr_scheduler" (optional): dictionary specifying the scheduler and its configuration
        """

        # Instantiate the optimizer
        optimizer = self.optimizer(params=self.parameters())
        optimizer_config = {"optimizer": optimizer}

        # Handle PlateauWithWarmup scheduler
        if self.lr_scheduler.func == PlateauWithWarmup:
            lr_scheduler = self.lr_scheduler(optimizer)

            es_cbs = [cb for cb in self.trainer.callbacks if isinstance(cb, EarlyStopping)]
            monitor = es_cbs[0].monitor if es_cbs else "avg_val/virtual_screening/bedroc"

            optimizer_config["monitor"] = monitor
            optimizer_config["lr_scheduler"] = {
                "scheduler": lr_scheduler,
                "monitor": monitor,
                "interval": "epoch",
                "frequency": self.trainer.check_val_every_n_epoch,
                "reduce_on_plateau": True,
            }

        # Handle CosineWithWarmup scheduler
        elif self.lr_scheduler.func == CosineWithWarmup:
            lr_scheduler = self.lr_scheduler(optimizer)
        
            optimizer_config["lr_scheduler"] = {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": self.trainer.check_val_every_n_epoch,
                }
          
        return optimizer_config
    


class ProteinModel(nn.Module):
    """
    Inference-only model for extracting protein and pocket representations.

    Parameters
    ----------
    num_pocket_nodes: int, optional
        Number of virtual pocket nodes initialized per protein by the VN-EGNN model.
    protein_node: bool, optional
        Whether protein-level embeddings are used in the model.
    device: str, optional
        Device used for inference (e.g. "cuda:0" or "cpu").
    """

    def __init__(
        self,
        num_pocket_nodes = 8,
        protein_node = True,
        device = "cuda:0"
    ) -> None:
        
        super().__init__()

        # Initialize VN-EGNN
        with open("configs/model/vnegnn/vnegnn.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        cfg.pop("_target_", None)

        self.vnegnn = VNEGNN(**cfg)

        # Initialize pocket encoder
        with open("configs/model/pocket_encoder/mlp.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        cfg.pop("_target_", None)

        self.pocket_encoder = MLPEncoder(**cfg)

        # Initialize protein encoder
        with open("configs/model/protein_encoder/mlp.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        cfg.pop("_target_", None)

        self.protein_encoder = MLPEncoder(**cfg)

        # Initialize cluster method
        with open("configs/model/cluster/dbscan.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        cfg.pop("_target_", None)

        self.cluster = DBSCANCluster(**cfg)

        self.num_pocket_nodes = num_pocket_nodes
        self.protein_node = protein_node
        self.device = device


    def load_from_checkpoint(
        self, 
        checkpoint_path
    ) -> None:
        """
        Load pretrained model weights.

        Parameters
        ----------
        checkpoint_path : str
            Path to the directory containing the saved model weights vnegnn.pth, pocket_encoder.pth, protein_encoder.pth
        """

        # Load VNEGNN weights
        vnegnn_state_dict = torch.load(f'{checkpoint_path}/vnegnn.pth', weights_only=True, map_location=self.device)
        self.vnegnn.load_state_dict(vnegnn_state_dict)

        # Load pocket encoder weights
        pocket_encoder_state_dict = torch.load(f'{checkpoint_path}/pocket_encoder.pth', weights_only=True, map_location=self.device)
        self.pocket_encoder.load_state_dict(pocket_encoder_state_dict)

        # Load protein encoder weights
        protein_encoder_state_dict = torch.load(f'{checkpoint_path}/protein_encoder.pth', weights_only=True, map_location=self.device)
        self.protein_encoder.load_state_dict(protein_encoder_state_dict)


    @torch.no_grad()
    def forward(
        self, 
        proteins
        ) -> dict:
        """
        Run inference on a batch of protein graphs. The model predicts pocket nodes using VN-EGNN, clusters them into
        candidate pockets, and encodes both pockets and proteins into a normalized embedding space.

        Parameters
        ----------
        proteins: torch_geometric.data.Batch
            Batch of protein graphs represented as PyTorch Geometric objects.

        Returns
        -------
        dict
            Dictionary containing:
            protein_names: list[str]
                Names or identifiers of the proteins in the batch.
            pocket_pos: torch.Tensor
                Predicted pocket coordinates after clustering.
            confidence: torch.Tensor
                Confidence scores for each predicted pocket.
            pocket_batch_idx: torch.Tensor
                Mapping indicating which protein each pocket belongs to.
            pocket_embeddings: torch.Tensor
                L2-normalized pocket embeddings.
            protein_embeddings: torch.Tensor
                L2-normalized protein embeddings.
        """

        # Move batch to target device
        proteins = proteins.to(self.device)

        # Run VN-EGNN to predict pocket and protein features
        pocket_feats, pocket_pos, protein_feats, _, _, _, confidence, _, _, _ = self.vnegnn(proteins)

        # Reshape predicted pocket nodes per protein
        B = len(proteins)
        pocket_pos = pocket_pos.view(B, self.num_pocket_nodes, -1)
        pocket_feats = pocket_feats.view(B, self.num_pocket_nodes, -1)
        confidence = confidence.view(B, self.num_pocket_nodes)

        # Cluster predicted pocket nodes into pocket candidates
        pocket_pos, pocket_feats, confidence, pocket_batch_idx = self.cluster(pocket_pos, pocket_feats, confidence)

        # Encode pocket features into embedding space
        pocket_embeddings = self.pocket_encoder(pocket_feats)
        pocket_embeddings = F.normalize(pocket_embeddings, dim=1)

        # Encode protein features into embedding space
        protein_embeddings = self.protein_encoder(protein_feats)
        protein_embeddings = F.normalize(protein_embeddings, dim=1)

        return {
            "protein_names": proteins.name,
            "pocket_pos": pocket_pos,
            "confidence": confidence,
            "pocket_batch_idx": pocket_batch_idx,
            "pocket_embeddings": pocket_embeddings,
            "protein_embeddings": protein_embeddings
        }
