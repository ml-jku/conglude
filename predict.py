import argparse
import torch
from datetime import datetime
import os
import numpy as np

from embed_proteins import ProteinEmbedder
from embed_ligands import LigandEmbedder



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make ConGLUDe predictions.")

    # Model and dataset paths
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/best_model", help="Directory containing trained model weights.")
    parser.add_argument("--data_dir", type=str, default="./data/datasets/predict_datasets", help="Directory containing <dataset_name>/info/protein_ids.txt file with protein PDB IDs.")
    parser.add_argument("--dataset_name", type=str, default="predict_example", help="Name of the dataset (used for results folder).")
    parser.add_argument("--pdb_dir", type=str, default="./data/pdb_files", help="Directory for raw PDB files (if .pdb files don't exist yet, they get downloaded from the PDB).")
    parser.add_argument("--smiles_path", type=str, default="./data/datasets/predict_datasets/predict_example/info/smiles.txt", help="Path to SMILES input file.")
    parser.add_argument("--results_dir", type=str, default=None, help="Directory to save predictions and embeddings (auto timestamped if None).")

    # Pre-processing parameters
    parser.add_argument("--ecfp_radius", type=int, default=2, help="Radius for ECFP fingerprint generation.")
    parser.add_argument("--fp_length", type=int, default=2048, help="Length of ECFP fingerprint vector.")
    parser.add_argument("--calc_descriptors", type=bool, default=True, help="Compute additional molecular descriptors.")

    # Inference parameters
    parser.add_argument("--protein_batch_size", type=int, default = 64, help="Batch size for protein embedding.")
    parser.add_argument("--ligand_batch_size", type=int, default = 1024, help="Batch size for ligand embedding.")
    parser.add_argument("--num_workers", type=int, default=64, help="Workers for dataset loading.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference.")

    # Save flags
    parser.add_argument("--overwrite", type=bool, default=False, help="Overwrite existing processed datasets.")
    parser.add_argument("--save_cleaned_pdbs", type=bool, default=False, help="Whether to save cleaned PDB files.")
    parser.add_argument("--save_complex_info", type=bool, default=False, help="Whether to save additional complex info.")
    parser.add_argument("--save_embeddings", type=bool, default=True, help="Whether to save protein and ligand embeddings.")

    # Parse arguments
    args = parser.parse_args()

    if args.results_dir is None:
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        results_dir = f"results/{args.dataset_name}/{timestamp}/"
    else:
        results_dir = args.results_dir

    # Embed proteins and pockets
    protein_embedder = ProteinEmbedder(
        checkpoint_path = args.checkpoint_path,
        data_dir = args.data_dir,
        dataset_name = args.dataset_name,
        pdb_dir = args.pdb_dir,
        results_dir = results_dir,
        batch_size = args.protein_batch_size,
        overwrite = args.overwrite,
        num_workers = args.num_workers,
        save_cleaned_pdbs = args.save_cleaned_pdbs,
        save_complex_info = args.save_complex_info,
        save_embeddings = args.save_embeddings,
        device = args.device
    )

    protein_names, protein_embeddings, pocket_names, pocket_embeddings = protein_embedder.embed()

    # Embed ligands
    ligand_embedder = LigandEmbedder(
        checkpoint_path = args.checkpoint_path,
        data_dir = args.data_dir,
        dataset_name = args.dataset_name,
        smiles_path = args.smiles_path,   
        results_dir = results_dir,     
        ecfp_radius = args.ecfp_radius,
        fp_length = args.fp_length,        
        calc_descriptors = args.calc_descriptors,        
        batch_size = args.ligand_batch_size,
        overwrite = args.overwrite,
        num_workers = args.num_workers,
        save_embeddings = args.save_embeddings,
        device = args.device
    )

    ligand_embeddings = ligand_embedder.embed()

    # Split ligand embeddings into protein and pocket parts
    encoded_ligands_p = torch.nn.functional.normalize(ligand_embeddings[:,:(ligand_embeddings.shape[1]//2)], dim=1)
    encoded_ligands_b = torch.nn.functional.normalize(ligand_embeddings[:,(ligand_embeddings.shape[1]//2):], dim=1)

    # Make predictions
    vs_preds = protein_embeddings @ encoded_ligands_p.t()
    pr_preds = pocket_embeddings @ encoded_ligands_b.t()

    # Save predictions
    predictions_path = f"{results_dir}/predictions/"
    os.makedirs(predictions_path, exist_ok=True)

    np.save(os.path.join(predictions_path, "vs_predictions.npy"), vs_preds.cpu().numpy())
    np.save(os.path.join(predictions_path, "pr_predictions.npy"), pr_preds.cpu().numpy())