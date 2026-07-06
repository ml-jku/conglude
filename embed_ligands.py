import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml
import os
import numpy as np
import argparse
from datetime import datetime

from conglude.utils.common import read_list_from_txt, write_json
from conglude.utils.data_processing import LigandProcessor
from conglude.modules.mlp import MLPEncoder



class LigandEmbedder():
    """
    Utility class for computing ligand embeddings using a trained ligand encoder.

    Parameters
    ----------
    checkpoint_path: str
        Directory containing the trained ligand encoder weights.
    dataset_dir: str
        Full path to the dataset directory (containing processed/ligand_embeddings/).
    smiles_path: str, optional
        Path to the file containing SMILES strings. Defaults to <dataset_dir>/info/smiles.txt.
    results_dir: str, optional
        Directory where embeddings will be saved.
    ecfp_radius: int
        Radius used for computing ECFP fingerprints.
    fp_length: int
        Length of the fingerprint bit vector.
    calc_descriptors: bool
        Whether molecular descriptors should be computed.
    batch_size: int
        Batch size used during embedding computation.
    overwrite: bool
        Whether existing processed data should be overwritten.
    num_workers: int
        Number of parallel workers used during feature computation.
    save_embeddings: bool
        Whether to save ligand embeddings
    device: str
        Device used for inference.
    """

    def __init__(
        self,
        checkpoint_path = "./checkpoints/best_model",
        dataset_dir = "./data/datasets/predict_datasets/embed_ligands_example",
        smiles_path = None,
        results_dir = None,
        ecfp_radius = 2,
        fp_length = 2048,
        calc_descriptors = True,
        batch_size = 1024,
        overwrite = False,
        num_workers = 64,
        save_embeddings = True,
        device = "cuda:0",
    ) -> None:

        self.checkpoint_path = checkpoint_path
        self.dataset_dir = dataset_dir
        self.smiles_path = smiles_path
        self.ecfp_radius = ecfp_radius
        self.fp_length = fp_length
        self.calc_descriptors = calc_descriptors
        self.batch_size = batch_size
        self.overwrite = overwrite
        self.num_workers = num_workers
        self.save_embeddings = save_embeddings
        self.device = device

        if dataset_dir.startswith("./data/") or dataset_dir.startswith("data/"):
            self.data_dir = dataset_dir
        else:
            self.data_dir = os.path.join(dataset_dir, "ConGLUDe", "data")

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
        Prepare the ligand dataset and dataloader.

        Returns
        -------
        DataLoader
            PyTorch DataLoader yielding batches of ligand representations.
        """

        ligand_emb_dir = os.path.join(self.data_dir, "processed", "ligand_embeddings")

        if not os.path.exists(os.path.join(ligand_emb_dir, "index2smiles.json")):

            if self.smiles_path is None:
                self.smiles_path = os.path.join(self.data_dir, "info", "smiles.txt")
            assert os.path.exists(self.smiles_path), f"SMILES file not found: {self.smiles_path}"

            smiles_list = read_list_from_txt(self.smiles_path)
            index2smiles_dict = {str(i): smiles for i, smiles in enumerate(smiles_list)}

            os.makedirs(ligand_emb_dir, exist_ok=True)
            write_json(os.path.join(ligand_emb_dir, "index2smiles.json"), index2smiles_dict)

        if self.overwrite \
            or not os.path.exists(os.path.join(ligand_emb_dir, f"ecfp{2*self.ecfp_radius}_{self.fp_length}.pt")) \
            or (not os.path.exists(os.path.join(ligand_emb_dir, "descriptors.pt")) and self.calc_descriptors):

            ligand_processor = LigandProcessor(
                dataset_dir = self.data_dir,
                ecfp_radius = self.ecfp_radius,
                fp_length = self.fp_length,
                calc_descriptors = self.calc_descriptors,
                num_workers = self.num_workers,
            )

            ligand_processor.process()

        fingerprint_path = os.path.join(ligand_emb_dir, f"ecfp{2*self.ecfp_radius}_{self.fp_length}.pt")
        descriptor_path = os.path.join(ligand_emb_dir, "descriptors.pt")

        fingerprints = torch.load(fingerprint_path, weights_only = False).to(dtype=torch.float32)

        if self.calc_descriptors:
            descriptors = torch.load(descriptor_path, weights_only = False).to(dtype=torch.float32)
            ligand_features = torch.cat([fingerprints, descriptors], dim=1)
        else:
            ligand_features = fingerprints

        dataset = TensorDataset(ligand_features)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        return dataloader



    def setup_model(
        self
    ) -> MLPEncoder:
        """
        Load the trained ligand encoder.

        Returns
        -------
        MLPEncoder
            Ligand encoder model with loaded weights in evaluation mode.
        """

        with open("configs/model/ligand_encoder/mlp.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        cfg.pop("_target_", None)

        model = MLPEncoder(**cfg)

        state_dict = torch.load(f'{self.checkpoint_path}/ligand_encoder.pth', weights_only = True)
        model.load_state_dict(state_dict)

        model.eval()
        model.to(self.device)

        return model


    @torch.no_grad()
    def embed(
        self
    ) -> torch.Tensor:
        """
        Compute ligand embeddings for the dataset.

        Returns
        -------
        ligand_embeddings: torch.Tensor
            Tensor containing the embedding vectors for all ligands.

        """

        ligand_embeddings = []

        dataloader = self.setup_dataloader()
        model = self.setup_model()

        for batch in dataloader:
            ligands = batch[0]
            ligands = ligands.to(self.device)
            ligand_embeddings.append(model.forward(ligands))

        ligand_embeddings = torch.cat(ligand_embeddings)

        if self.save_embeddings:
            embeddings_path = os.path.join(self.results_dir, "embeddings")
            os.makedirs(embeddings_path, exist_ok=True)
            emb_np = ligand_embeddings.cpu().numpy()
            np.save(os.path.join(embeddings_path, "ligand_embeddings.npy"), emb_np)
            write_json(os.path.join(embeddings_path, "metadata_ligand_embeddings.json"), {
                "num_ligands": emb_np.shape[0],
                "embedding_dim": emb_np.shape[1],
                "precision": "float32",
                "vs_only": False,
            })

        return ligand_embeddings


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate ligand embeddings.")

    # Model and dataset paths
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/best_model", help="Path to model checkpoint directory.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset directory containing info/smiles.txt or processed ligand features.")
    parser.add_argument("--smiles_path", type=str, default=None, help="Path to SMILES input file. Defaults to <dataset_dir>/info/smiles.txt.")
    parser.add_argument("--results_dir", type=str, default=None, help="Directory to store results (timestamped if None).")

    # Pre-processing parameters
    parser.add_argument("--ecfp_radius", type=int, default=2, help="Radius for ECFP fingerprint generation.")
    parser.add_argument("--fp_length", type=int, default=2048, help="Length of ECFP fingerprint vector.")
    parser.add_argument("--no_descriptors", action="store_true", help="Disable molecular descriptor computation.")

    # Inference parameters
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size used during embedding generation.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed ligand features.")
    parser.add_argument("--num_workers", type=int, default=64, help="Number of worker processes used for preprocessing.")
    parser.add_argument("--no_save_embeddings", action="store_true", help="Disable saving ligand embeddings.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device used for embedding (e.g., cuda:0 or cpu).")

    # Parse arguments
    args = parser.parse_args()

    ligand_embedder = LigandEmbedder(
        checkpoint_path = args.checkpoint_path,
        dataset_dir = args.dataset_dir,
        smiles_path = args.smiles_path,
        results_dir = args.results_dir,
        ecfp_radius = args.ecfp_radius,
        fp_length = args.fp_length,
        calc_descriptors = not args.no_descriptors,
        batch_size = args.batch_size,
        overwrite = args.overwrite,
        num_workers = args.num_workers,
        save_embeddings = not args.no_save_embeddings,
        device = args.device,
    )

    ligand_embedder.embed()
