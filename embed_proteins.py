import os
import torch
import argparse
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from itertools import groupby
from datetime import datetime
from typing import Tuple

from conglude.model import ProteinModel
from conglude.datamodule import ConGLUDeDataset
from conglude.utils.common import write_list_to_txt
from conglude.utils.collate_functions import custom_collate_protein



class ProteinEmbedder():
    """
    Complete pipeline for computing protein and pocket embeddings and pocket predictions.

    Parameters
    ----------
    checkpoint_path: str
        Directory containing trained model weights.
    data_dir: str
        Path to dataset directory.
    dataset_name: str
        Name of the dataset used for prediction.
    pdb_dir: str
        Directory containing PDB structure files.
    results_dir: str, optional
        Directory where results will be stored. If None, a timestamped directory will be created automatically.
    batch_size: int
        Batch size used during inference.
    overwrite: bool
        Whether dataset preprocessing files should be overwritten.
    num_workers: int
        Number of workers used for dataset preprocessing.
    save_cleaned_pdbs: bool
        Whether cleaned PDB files should be saved during preprocessing.
    save_complex_info: bool
        Whether complex information should be stored.
    save_embeddings:
        Whether to save protein/pocket embeddings.
    device: str
        Device used for inference (e.g. "cuda:0" or "cpu").
    """

    def __init__(
        self,
        checkpoint_path = "./checkpoints/best_model",
        data_dir = "./data/datasets/predict_datasets/",
        dataset_name = "embed_proteins_example", 
        pdb_dir = "./data/pdb_files",
        results_dir = None,
        batch_size = 64,        
        overwrite = False,
        num_workers = 64,
        save_cleaned_pdbs = False,
        save_complex_info = False,
        save_embeddings = True,
        device = "cuda:0"
    ) -> None:
        
        self.checkpoint_path = checkpoint_path
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.pdb_dir = pdb_dir
        self.overwrite = overwrite
        self.num_workers = num_workers
        self.save_cleaned_pdbs = save_cleaned_pdbs
        self.save_complex_info = save_complex_info
        self.save_embeddings = save_embeddings
        self.device = device

        # Create output directory   
        if results_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            self.results_dir = f"results/{dataset_name}/{timestamp}/"
        
        else:
            self.results_dir = results_dir


    def setup_dataloader(
        self
    ) -> DataLoader:
        """
        Create the dataset and dataloader used for protein inference.

        Returns
        -------
        DataLoader
            PyTorch DataLoader yielding batches of protein graphs.
        """
        
        # Initialize dataset for prediction
        dataset = ConGLUDeDataset(
            dataset_dir = f"{self.data_dir}/{self.dataset_name}",
            dataset_name = self.dataset_name,        
            fingerprint_type = None,
            load_descriptors = False,
            batch_size = self.batch_size,        
            pdb_dir = self.pdb_dir,
            overwrite = self.overwrite,
            num_workers = self.num_workers,
            calc_mol_feats = False,
            save_cleaned_pdbs = self.save_cleaned_pdbs,
            save_complex_info = self.save_complex_info
        )
        
        # Create dataloader for batched inference
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=custom_collate_protein,
        )

        return dataloader
    

    def setup_model(
        self
    ) -> ProteinModel:
        """
        Load the trained protein model for inference.

        Returns
        -------
        ProteinModel
            Model with loaded weights in evaluation mode.
        """

        model = ProteinModel()
        model.load_from_checkpoint(self.checkpoint_path)
        model.eval()
        model.to(self.device)

        return model
    

    def initialize_save_tensors(
        self
        ) -> None:
        """
        Initialize containers used to accumulate predictions and embeddings during batched inference.
        """
        
        # Initialize lists to store protein and pocket identifiers across batches
        self.protein_names = []
        self.pocket_names = []

        # Initialize lists for saving pocket predictions
        self.pocket_pos = []
        self.confidence = []
        self.pocket_batch_idx = []
         
        # Initialize lists for saving embeddings
        self.pocket_embeddings = []
        self.protein_embeddings = []

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
            Dictionary returned by the model containing predictions and embeddings for the current batch.
        """

        # Save protein names
        self.protein_names.extend(output["protein_names"])
       
        # Generate pocket names of the form: "<protein_name>_pocket_<running_index>"
        self.pocket_names.extend([f"{output['protein_names'][j]}_pocket_{k}" for j, protein in groupby(output["pocket_batch_idx"]) for k, _ in enumerate(protein, start=1)])
        
        # Save predictions and indices (if enabled)
        self.pocket_pos.append(output["pocket_pos"].detach())
        self.confidence.append(output["confidence"].detach())
        self.pocket_batch_idx.append(output["pocket_batch_idx"].detach() + self.batch_offset)

        self.batch_offset += output["pocket_batch_idx"].max().item() + 1

        # Save protein and pocket embeddings
        self.pocket_embeddings.append(output["pocket_embeddings"])
        self.protein_embeddings.append(output["protein_embeddings"])


    def save_results(
        self, 
        dataset_name
    ) -> None:
        """
        Save predicted pockets and embeddings to disk.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset; used to create a dataset-specific subdirectory.
        """

        # Create directory for saving predictions
        predictions_path = f"{self.results_dir}/predictions/"
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
            
        # Concatenate pocket and protein embeddings
        self.pocket_embeddings = torch.cat(self.pocket_embeddings, dim=0)
        self.protein_embeddings = torch.cat(self.protein_embeddings, dim=0)
        
        if self.save_embeddings:
            # Create directory for saving embeddings
            embeddings_path = f"{self.results_dir}/embeddings/"
            os.makedirs(embeddings_path, exist_ok=True)

            # Save lists of protein and pocket names
            write_list_to_txt(os.path.join(embeddings_path, "protein_names.txt"), self.protein_names)
            write_list_to_txt(os.path.join(embeddings_path, "pocket_names.txt"), self.pocket_names)

            # Save protein, pocket and ligand embeddings
            np.save(os.path.join(embeddings_path, "pocket_embeddings.npy"), self.pocket_embeddings.cpu().numpy())
            np.save(os.path.join(embeddings_path, "protein_embeddings.npy"), self.protein_embeddings.cpu().numpy())


    @torch.no_grad()
    def embed(
        self
    ) -> Tuple[list, torch.Tensor, list, torch.Tensor]:
        """
        Run inference over all proteins and save predictions and embeddings.

        Returns
        -------
        protein_names: list[str]
            List of protein identifiers in the same order as the rows in `protein_embeddings`.
        protein_embeddings: torch.Tensor
            Tensor containing the embedding vectors for all proteins.
        pocket_names: list[str]
            List of predicted pocket identifiers. Each pocket name follows the format "<protein_name>_pocket_<index>".
        pocket_embeddings : torch.Tensor
            Tensor containing embedding vectors for all predicted pockets. Each row corresponds to the pocket with the same index in `pocket_names`.
        """

        self.initialize_save_tensors()

        dataloader = self.setup_dataloader()
        model = self.setup_model()

        # Run inference over dataset
        for proteins in dataloader:
            output = model.forward(proteins)
            self.update_save_lists(output)

        # Save final results
        self.save_results(self.dataset_name)

        return self.protein_names, self.protein_embeddings, self.pocket_names, self.pocket_embeddings



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute protein and pocket embeddings using a trained model.")

    # Model and dataset paths
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/best_model", help="Directory containing trained model weights.")
    parser.add_argument("--data_dir", type=str, default="./data/datasets/predict_datasets", help="Directory containing <dataset_name>/info/protein_ids.txt file with protein PDB IDs.")
    parser.add_argument("--dataset_name", type=str, default="embed_proteins_example", help="Name of the dataset (used for results folder).")
    parser.add_argument("--pdb_dir", type=str, default="./data/pdb_files", help="Directory for raw PDB files (if .pdb files don't exist yet, they get downloaded from the PDB).")
    parser.add_argument("--results_dir", type=str, default=None, help="Directory to save predictions and embeddings (auto timestamped if None).")

    # Inference parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=64, help="Workers for dataset loading.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference.")

    # Save flags
    parser.add_argument("--overwrite", type=bool, default=False, help="Overwrite existing processed datasets.")
    parser.add_argument("--save_cleaned_pdbs", type=bool, default=False, help="Save cleaned PDB files.")
    parser.add_argument("--save_complex_info", type=bool, default=False, help="Save additional complex info.")
    parser.add_argument("--save_embeddings", type=bool, default=True, help="Whether to save protein and pocket embeddings.")

    # Parse arguments
    args = parser.parse_args()

    # Initialize embedder
    protein_embedder = ProteinEmbedder(
        checkpoint_path = args.checkpoint_path,
        data_dir = args.data_dir,
        dataset_name = args.dataset_name,
        pdb_dir = args.pdb_dir,
        results_dir = args.results_dir,
        batch_size = args.batch_size,
        overwrite = args.overwrite,
        num_workers = args.num_workers,
        save_cleaned_pdbs = args.save_cleaned_pdbs,
        save_complex_info = args.save_complex_info,
        save_embeddings = args.save_embeddings,
        device = args.device,
    )

    # Run embedding pipeline and save results
    protein_embedder.embed()
