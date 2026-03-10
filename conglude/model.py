import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max
import pytorch_lightning as pl
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
            # try:
            vnegnn_state_dict = torch.load(f'{checkpoint_path}/{checkpoint_name}/vnegnn.pth', weights_only=True)
            self.vnegnn.load_state_dict(vnegnn_state_dict)
            # except:
            #     print("Unable to load VN-EGNN weights.")
        
        self.ligand_encoder = ligand_encoder
        if checkpoint_name is not None:
            try:
                ligand_encoder_state_dict = torch.load(f'{checkpoint_path}/{checkpoint_name}/ligand_encoder.pth', weights_only=True)
                self.ligand_encoder.load_state_dict(ligand_encoder_state_dict)
            except:
                print("Unable to load ligand encoder weights.")

        self.pocket_encoder = pocket_encoder
        if checkpoint_name is not None:
            try:
                pocket_encoder_state_dict = torch.load(f'{checkpoint_path}/{checkpoint_name}/pocket_encoder.pth', weights_only=True)
                self.pocket_encoder.load_state_dict(pocket_encoder_state_dict)
            except:
                print("Unable to load pocket encoder weights.")

        self.protein_encoder = protein_encoder
        if checkpoint_name is not None:
            try:
                protein_encoder_state_dict = torch.load(f'{checkpoint_path}/{checkpoint_name}/protein_encoder.pth', weights_only=True)
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
                    "virtual_screening": VirtualScreeningMetrics(ef_fractions=[0.05], calc_re=False),
                    "target_fishing": TargetFishingMetrics(),
                    "pocket_prediction": PocketPredictionMetrics(calc_iou=False),
                    "pocket_ranking": PocketRankingMetrics(),
                },

                "LB_train": {
                    "virtual_screening": VirtualScreeningMetrics(calc_re=False),
                },
            }

        # Otherwise, initialize metrics based on whether the training dataset is structure- or ligand-based
        elif dm.train_dataloader().dataset.structure_based:
            self.metrics = {
                dm.train_dataloader().dataset.dataset_name: {
                    "virtual_screening": VirtualScreeningMetrics(ef_fractions=[0.05], calc_re=False),
                    "target_fishing": TargetFishingMetrics(),
                    "pocket_prediction": PocketPredictionMetrics(calc_iou=False),
                    "pocket_ranking": PocketRankingMetrics(),
                }
            }
        else:
            self.metrics = {
                dm.train_dataloader().dataset.dataset_name: {
                    "virtual_screening": VirtualScreeningMetrics(calc_re=False),
                }
            }

        # Add metrics for each validation dataset based on whether it's structure- or ligand-based
        for dataloader in dm.val_dataloader():
            name = dataloader.dataset.dataset_name
            
            if dataloader.dataset.structure_based:
                self.metrics[name] = {
                    "virtual_screening": VirtualScreeningMetrics(ef_fractions=[0.05], calc_re=False),
                    "target_fishing": TargetFishingMetrics(),
                    "pocket_prediction": PocketPredictionMetrics(calc_iou=False),
                    "pocket_ranking": PocketRankingMetrics(),
                }
            else:
                self.metrics[name] = {
                    "virtual_screening": VirtualScreeningMetrics(calc_re=False),
                }

        # Add metrics for each test dataset based on its task
        for dataloader in dm.test_dataloader():
            name = dataloader.dataset.dataset_name
            task = dataloader.dataset.task

            if task == "vs":
                self.metrics[name] = {"virtual_screening": VirtualScreeningMetrics()}
            elif task == "tf":
                self.metrics[name] = {"target_fishing": TargetFishingMetrics()}
            elif task == "pp":
                self.metrics[name] = {"pocket_prediction": PocketPredictionMetrics(calc_iou=False)}
            elif task == "pr":
                self.metrics[name] = {"pocket_ranking": PocketRankingMetrics()}
            elif task == "all":
                self.metrics[name] = {
                    "virtual_screening": VirtualScreeningMetrics(),
                    "target_fishing": TargetFishingMetrics(),
                    "pocket_prediction": PocketPredictionMetrics(calc_iou=False),
                    "pocket_ranking": PocketRankingMetrics(),
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

    
    def on_test_epoch_end(
        self
    ) -> None:
        """
        Compute and log test metrics. 
        Optionally saves predictions or embeddings.
        """

        test_dataloaders = self.trainer.datamodule.test_dataloader()

        # Compute metrics per test dataloader
        for test_dataloader in test_dataloaders:
            loader_name = test_dataloader.dataset.dataset_name
            self.compute_and_log_metrics(loader_name)

        # If requested, save predictions or embeddings to disk
        if self.save_predictions or self.save_embeddings:
            self.save_results()


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
        if (dataset_specs["task"] in ["train", "val"] and dataset_specs["structure_based"]) or dataset_specs["task"] in ["pr", "all"]:

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
            output["labels"]["vs_labels"] = labels
            output["index"]["ligand_batch_idx"] = ligand_batch_idx
            output["index"]["ligand_idx"] = ligand_idx

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
                    output["index"]["pocket_center_batch_idx"]
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
                labels = torch.eye(vs_preds.shape[0], device = self.device)
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

                self.metrics[dataset_specs["name"]]["virtual_screening"].update(vs_preds, labels, ligand_batch_idx)

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
                self.update_save_lists(output)
        
        return loss


    def initialize_save_tensors(
        self
        ) -> None:
        """
        Initialize containers for storing predictions, embeddings, and metadata during testing.
        """
        
        # Initialize lists to store protein and pocket identifiers across batches
        self.protein_names = []
        self.pocket_names = []

        # Initialize lists for saving test predictions
        if self.save_predictions:

            self.pocket_pos = []
            self.confidence = []
            self.pocket_batch_idx = []
            self.pocket_preds = []
            self.pocket_centers = []
            self.pocket_center_batch_idx = []
            self.vs_preds = []
            self.vs_labels = []
            self.ligand_batch_idx = []
            self.ligand_idx = []

        # Initialize lists for saving test embeddings
        if self.save_embeddings:

            self.pocket_embeddings = []
            if self.protein_node:
                self.protein_embeddings = []
                self.ligand_embeddings_p = []
            self.ligand_embeddings_b = []

        # Running offset for batch indices
        self.batch_offset = 0


    def update_save_lists(
        self, 
        output: dict
    ) -> None:

        """
        Accumulate batch-wise outputs into lists.

        Parameters
        ----------
        output: dict
            Dictionary returned by `forward(...)` and `process_step(...)`. 
        """

        def append_if_exists(
            container: list, 
            dictionary: dict, 
            key: str, detach: 
            bool = True
        ) -> None:
            """
            Append a value from a dictionary to a list if the key exists.

            Parameters
            ----------
            container: list
                Target list to append the value to.
            dictionary: dict
                Dictionary potentially containing the value.
            key: str
                Key to look up in the dictionary.
            detach: bool
                If True and the value is a torch.Tensor, it is detached from the computation graph before storing.
            """
            
            if key in dictionary:
                x = dictionary[key]

            # Detach tensors to avoid storing computation graphs
            if detach and torch.is_tensor(x):
                x = x.detach()

            container.append(x)
        
        # Save protein names
        self.protein_names.extend(output["protein_names"])
       
        # Generate pocket names of the form: "<protein_name>_pocket_<running_index>"
        self.pocket_names.extend([f"{output['protein_names'][j]}_pocket_{k}" for j, protein in groupby(output["vnegnn_predictions"]["pocket_batch_idx"]) for k, _ in enumerate(protein, start=1)])
        
        # Save predictions and indices (if enabled)
        if self.save_predictions:
            append_if_exists(self.pocket_pos, output["predictions"], "pocket_pos_clustered")
            append_if_exists(self.confidence, output["predictions"], "confidence_clustered")

            if "pocket_batch_idx" in output["index"]:
                self.pocket_batch_idx.append(output["index"]["pocket_batch_idx"].detach() + self.batch_offset)

            append_if_exists(self.pocket_preds, output["predictions"], "pocket_preds")
            append_if_exists(self.pocket_centers, output["labels"], "pocket_centers")

            if "pocket_center_batch_idx" in output["index"]:
                self.pocket_center_batch_idx.append(output["index"]["pocket_center_batch_idx"].detach() + self.batch_offset)

            append_if_exists(self.vs_preds, output["predictions"], "vs_preds")
            append_if_exists(self.vs_labels, output["labels"], "vs_labels")
            append_if_exists(self.ligand_idx, output["index"], "ligand_idx")

            if "ligand_batch_idx" in output["index"]:
                self.ligand_batch_idx.append(output["index"]["ligand_batch_idx"].detach() + self.batch_offset)
            self.batch_offset += output["index"]["pocket_center_batch_idx"].max().item() + 1

         # Save protein, pocket and ligand embeddings (if enabled)
        if self.save_embeddings:
            append_if_exists(self.pocket_embeddings, output["embeddings"], "pocket_embeddings")

            if self.protein_node:
                append_if_exists(self.protein_embeddings, output["embeddings"], "protein_embeddings")
                if "ligand_embeddings" in output["embeddings"]:
                    self.ligand_embeddings_p = torch.cat([self.ligand_embeddings_p, output["embeddings"]["ligand_embeddings"][:, :output["embeddings"]["ligand_embeddings"].shape[1]//2].detach()])
                    self.ligand_embeddings_b = torch.cat([self.ligand_embeddings_b, output["embeddings"]["ligand_embeddings"][:, output["embeddings"]["ligand_embeddings"].shape[1]//2:].detach()])
            else:
                append_if_exists(self.ligand_embeddings_b, output["embeddings"], "ligand_embeddings")


    def save_results(
        self, 
        dataset_name
    ) -> None:
        """
        Save accumulated predictions and/or embeddings to disk.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset; used to create a dataset-specific subdirectory.
        """

        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

        # Save prediction results
        if self.save_predictions:

            # Create directory for saving predictions
            predictions_path = f"predictions/{dataset_name}/{timestamp}/"
            os.makedirs(predictions_path, exist_ok=True)
            
            # Concatenate stored tensors for pocket prediction results
            self.pocket_pos = torch.cat(self.pocket_pos, dim=0).cpu().numpy()
            self.confidence = torch.cat(self.confidence, dim=0).cpu().numpy()

            # Create DataFrame with predicted pocket coordinates and confidence scores
            pp_df = pd.DataFrame(self.pocket_pos, columns=["pred_x", "pred_y", "pred_z"])
            pp_df["confidence"] = self.confidence
            pp_df["pocket_name"] = self.pocket_names
            pp_df["protein_name"] = pp_df["pocket_name"].str.split("_").str[0]

            # Save pocket prediction results to CSV
            pp_df = pp_df[["protein_name", "pocket_name", "pred_x", "pred_y", "pred_z", "confidence"]]
            pp_df.to_csv(os.path.join(predictions_path, "pp_predictions.csv"), index=False)

            # Concatenate stored tensors for pocket ranking result
            self.pocket_preds = torch.cat(self.pocket_preds, dim=0).cpu().numpy()
            self.pocket_centers = torch.cat(self.pocket_centers, dim=0).cpu().numpy()
            self.ligand_idx = torch.cat(self.ligand_idx, dim=0).cpu().numpy()
            self.pocket_center_batch_idx = torch.cat(self.pocket_center_batch_idx, dim=0).cpu().numpy()
            self.pocket_batch_idx = torch.cat(self.pocket_batch_idx, dim=0).cpu().numpy()

            rows = []

            # Iterate over each ground-truth pocket center
            for i in range(self.pocket_centers.shape[0]):

                protein_idx = self.pocket_center_batch_idx[i].item()
                protein_name = self.protein_names[protein_idx]
                
                target_xyz = self.pocket_centers[i]
                ligand = self.ligand_idx[i].item()

                scores = self.pocket_preds[i]

                # Iterate over predicted pockets for this protein
                for pred_idx, score in enumerate(scores):
                    
                    # Ignore padded entries
                    if score == -100:
                        continue 

                    # Select predictions belonging to the current protein

                    pred_xyz = self.pocket_pos[self.pocket_batch_idx == protein_idx][pred_idx]
                    pocket_name = self.pocket_names[self.pocket_batch_idx == protein_idx][pred_idx]
                    conf = self.confidence[self.pocket_batch_idx == protein_idx][pred_idx].item()

                    # Compute Euclidean distance between predicted and target center
                    dist = torch.norm(pred_xyz - target_xyz).item()

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

            # Save pocket prediction results to CSV
            pr_df = pd.DataFrame(rows)
            pr_df.to_csv(os.path.join(predictions_path, "pr_predictions.csv"), index=False)

            # Concatenate stored tensors for virtual screening results
            self.vs_preds = torch.cat(self.vs_preds, dim=0).cpu().numpy()
            self.vs_labels = torch.cat(self.vs_labels, dim=0).cpu().numpy()
            self.ligand_batch_idx = torch.cat(self.ligand_batch_idx, dim=0).cpu().numpy()
            
            # Save virtual screening results to CSV
            vs_df = pd.DataFrame({
                "protein_name": [self.protein_names[i] for i in self.ligand_batch_idx.tolist()],
                "ligand_idx": self.ligand_idx,
                "vs_pred": self.vs_preds,
                "vs_label": self.vs_labels,
            })
            vs_df.to_csv(os.path.join(predictions_path, "vs_predictions.csv"), index=False)

        # Save embeddings
        if self.save_embeddings:
            
            # Create directory for saving embeddings
            embeddings_path = f"embeddings/{dataset_name}/{timestamp}/"
            os.makedirs(embeddings_path, exist_ok=True)
            
            # Save lists of protein and pocket names
            write_list_to_txt(os.path.join(embeddings_path, "protein_names.txt"), self.protein_names)
            write_list_to_txt(os.path.join(embeddings_path, "pocket_names.txt"), self.pocket_names)

            # Save protein, pocket and ligand embeddings
            self.pocket_embeddings = torch.cat(self.pocket_embeddings, dim=0)
            np.save(os.path.join(embeddings_path, "pocket_embeddings.npy"), self.pocket_embeddings)

            if self.protein_node:
                self.protein_embeddings = torch.cat(self.protein_embeddings, dim=0)
                np.save(os.path.join(embeddings_path, "protein_embeddings.npy"), self.protein_embeddings)
                self.ligand_embeddings_p = torch.cat(self.ligand_embeddings_p, dim=0)
                np.save(os.path.join(embeddings_path, "ligand_embeddings_p.npy"), self.ligand_embeddings_p)

            self.ligand_embeddings_b = torch.cat(self.ligand_embeddings_b, dim=0)
            np.save(os.path.join(embeddings_path, "ligand_embeddings_b.npy"), self.ligand_embeddings_b)
   

    def compute_and_log_metrics(
        self, 
        dataset_name: str
    ) -> dict:
        """
        Compute, log, and reset all metrics associated with a given dataset.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset whose metrics should be computed and logged, must be a key in `self.metrics`.
        """

        all_metrics = {}

         # Iterate over all possible tasks
        for task in ["virtual_screening", "target_fishing", "pocket_prediction", "pocket_ranking"]:
            
            # Only compute metrics if this dataset supports the given task
            if task in self.metrics[dataset_name]:

                # Compute aggregated metric values
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

            optimizer_config["lr_scheduler"] = {
                "scheduler": lr_scheduler,
                "monitor": self.trainer.callbacks[0].monitor,
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
        vnegnn_state_dict = torch.load(f'{checkpoint_path}/vnegnn.pth', weights_only=True)
        self.vnegnn.load_state_dict(vnegnn_state_dict)

        # Load pocket encoder weights
        pocket_encoder_state_dict = torch.load(f'{checkpoint_path}/pocket_encoder.pth', weights_only=True)
        self.pocket_encoder.load_state_dict(pocket_encoder_state_dict)

        # Load protein encoder weights
        protein_encoder_state_dict = torch.load(f'{checkpoint_path}/protein_encoder.pth', weights_only=True)
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
