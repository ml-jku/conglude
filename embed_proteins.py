import os
import shutil
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
    dataset_dir: str
        Full path to the dataset directory (containing info/protein_ids.txt).
    pdb_dir: str, optional
        Directory containing PDB structure files. Defaults to <dataset_dir>/raw/pdb_files.
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
    save_pymol_visualizations: bool
        Whether to save PyMOL-ready pocket scenes.
    device: str
        Device used for inference (e.g. "cuda:0" or "cpu").
    """

    def __init__(
        self,
        checkpoint_path = "./checkpoints/best_model",
        dataset_dir = "./data/datasets/predict_datasets/embed_proteins_example",
        pdb_dir = None,
        results_dir = None,
        batch_size = 64,
        overwrite = False,
        num_workers = 64,
        save_cleaned_pdbs = False,
        save_complex_info = False,
        save_embeddings = True,
        save_pymol_visualizations = False,
        device = "cuda:0"
    ) -> None:

        self.checkpoint_path = checkpoint_path
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.overwrite = overwrite
        self.num_workers = num_workers
        self.save_cleaned_pdbs = save_cleaned_pdbs
        self.save_complex_info = save_complex_info
        self.save_embeddings = save_embeddings
        self.save_pymol_visualizations = save_pymol_visualizations
        self.device = device

        if dataset_dir.startswith("./data/") or dataset_dir.startswith("data/"):
            self.data_dir = dataset_dir
        else:
            self.data_dir = os.path.join(dataset_dir, "ConGLUDe", "data")

        self.pdb_dir = pdb_dir if pdb_dir is not None else os.path.join(self.data_dir, "raw", "pdb_files")

        if results_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            dataset_name = os.path.basename(os.path.normpath(dataset_dir))
            if dataset_dir.startswith("./data/") or dataset_dir.startswith("data/"):
                self.results_dir = os.path.join("results", dataset_name, timestamp)
            else:
                self.results_dir = os.path.join(dataset_dir, "ConGLUDe", "results", timestamp)
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

        dataset = ConGLUDeDataset(
            dataset_dir = self.dataset_dir,
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

        self.protein_names = []
        self.pocket_names = []

        self.pocket_pos = []
        self.confidence = []
        self.pocket_batch_idx = []

        self.pocket_embeddings = []
        self.protein_embeddings = []

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

        self.protein_names.extend(output["protein_names"])

        self.pocket_names.extend([f"{output['protein_names'][j]}_pocket_{k}" for j, protein in groupby(output["pocket_batch_idx"]) for k, _ in enumerate(protein, start=1)])

        self.pocket_pos.append(output["pocket_pos"].detach())
        self.confidence.append(output["confidence"].detach())
        self.pocket_batch_idx.append(output["pocket_batch_idx"].detach() + self.batch_offset)

        self.batch_offset += output["pocket_batch_idx"].max().item() + 1

        self.pocket_embeddings.append(output["pocket_embeddings"])
        self.protein_embeddings.append(output["protein_embeddings"])


    def build_pymol_scene_script(self, protein_name: str, protein_df: pd.DataFrame) -> str:
        """Build a PyMOL script that loads the protein and overlays numbered pocket centers."""

        obj_name = protein_name

        lines = [
            "reinitialize",
            "set bg_rgb, white",
            f"load protein.pdb, {obj_name}",
            "hide everything",
            f"show cartoon, {obj_name}",
            f"color gray70, {obj_name}",
            f"set cartoon_transparency, 0.1, {obj_name}",
            "set label_color, black",
            "set label_size, 18",
            "set sphere_scale, 0.8",
            "set sphere_quality, 2",
        ]

        pocket_object_names = []
        ordered_df = protein_df.reset_index(drop=True)

        for pocket_number, row in enumerate(ordered_df.itertuples(index=False), start=1):
            object_name = f"{protein_name}_pocket_{pocket_number}"
            pocket_object_names.append(object_name)
            lines.extend([
                f'pseudoatom {object_name}, pos=[{row.pred_x:.3f}, {row.pred_y:.3f}, {row.pred_z:.3f}], name="{pocket_number}", resn=PKT, resi="{pocket_number}", chain=A',
                f"show spheres, {object_name}",
                f"color tv_red, {object_name}",
                f"label {object_name}, name",
            ])

        if pocket_object_names:
            lines.append(f"group pockets, {' '.join(pocket_object_names)}")

        lines.extend([
            f"orient {obj_name}",
            f"zoom {obj_name}, 10",
        ])

        return "\n".join(lines) + "\n"


    def save_results(self) -> None:
        """
        Save predicted pockets and embeddings to disk.
        """

        predictions_path = os.path.join(self.results_dir, "predictions")
        os.makedirs(predictions_path, exist_ok=True)

        self.pocket_pos = torch.cat(self.pocket_pos, dim=0).cpu().numpy()
        self.confidence = torch.cat(self.confidence, dim=0).cpu().numpy()

        pp_df = pd.DataFrame(self.pocket_pos, columns=["pred_x", "pred_y", "pred_z"])
        pp_df["confidence"] = self.confidence
        pp_df["pocket_name"] = self.pocket_names
        pp_df["protein_name"] = pp_df["pocket_name"].str.split("_").str[0]

        pp_df = pp_df[["protein_name", "pocket_name", "pred_x", "pred_y", "pred_z", "confidence"]]
        pp_df.to_csv(os.path.join(predictions_path, "pp_predictions.csv"), index=False)

        if self.save_pymol_visualizations:
            pymol_root = os.path.join(predictions_path, "pymol_visualizations")
            os.makedirs(pymol_root, exist_ok=True)

            for protein_name, protein_df in pp_df.groupby("protein_name", sort=False):
                protein_dir = os.path.join(pymol_root, protein_name)
                os.makedirs(protein_dir, exist_ok=True)

                pdb_path = os.path.join(self.data_dir, "processed", "cleaned_pdbs", protein_name, "protein.pdb")
                if not os.path.isfile(pdb_path):
                    continue

                shutil.copyfile(pdb_path, os.path.join(protein_dir, "protein.pdb"))
                scene_script = self.build_pymol_scene_script(protein_name, protein_df)
                with open(os.path.join(protein_dir, "view.pml"), "w", encoding="utf-8") as f:
                    f.write(scene_script)

        self.pocket_embeddings = torch.cat(self.pocket_embeddings, dim=0)
        self.protein_embeddings = torch.cat(self.protein_embeddings, dim=0)

        if self.save_embeddings:
            embeddings_path = os.path.join(self.results_dir, "embeddings")
            os.makedirs(embeddings_path, exist_ok=True)

            write_list_to_txt(os.path.join(embeddings_path, "protein_names.txt"), self.protein_names)
            write_list_to_txt(os.path.join(embeddings_path, "pocket_names.txt"), self.pocket_names)

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

        for proteins in dataloader:
            output = model.forward(proteins)
            self.update_save_lists(output)

        self.save_results()

        return self.protein_names, self.protein_embeddings, self.pocket_names, self.pocket_embeddings



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute protein and pocket embeddings using a trained model.")

    # Model and dataset paths
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/best_model", help="Directory containing trained model weights.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset directory containing info/protein_ids.txt.")
    parser.add_argument("--pdb_dir", type=str, default=None, help="Directory for raw PDB files. Defaults to <dataset_dir>/raw/pdb_files.")
    parser.add_argument("--results_dir", type=str, default=None, help="Directory to save predictions and embeddings (auto timestamped if None).")

    # Inference parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=64, help="Workers for dataset loading.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference.")

    # Save flags
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed datasets.")
    parser.add_argument("--save_cleaned_pdbs", action="store_true", help="Save cleaned PDB files.")
    parser.add_argument("--save_complex_info", action="store_true", help="Save additional complex info.")
    parser.add_argument("--no_save_embeddings", action="store_true", help="Disable saving protein and pocket embeddings.")
    parser.add_argument("--save_pymol_visualizations", action="store_true", help="Save PyMOL-ready pocket scenes.")

    # Parse arguments
    args = parser.parse_args()

    protein_embedder = ProteinEmbedder(
        checkpoint_path = args.checkpoint_path,
        dataset_dir = args.dataset_dir,
        pdb_dir = args.pdb_dir,
        results_dir = args.results_dir,
        batch_size = args.batch_size,
        overwrite = args.overwrite,
        num_workers = args.num_workers,
        save_cleaned_pdbs = args.save_cleaned_pdbs,
        save_complex_info = args.save_complex_info,
        save_embeddings = not args.no_save_embeddings,
        save_pymol_visualizations = args.save_pymol_visualizations,
        device = args.device,
    )

    protein_embedder.embed()
