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
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset directory containing info/protein_ids.txt and info/smiles.txt.")
    parser.add_argument("--pdb_dir", type=str, default=None, help="Directory for raw PDB files. Defaults to <dataset_dir>/raw/pdb_files.")
    parser.add_argument("--smiles_path", type=str, default=None, help="Path to SMILES input file. Defaults to <dataset_dir>/info/smiles.txt.")
    parser.add_argument("--results_dir", type=str, default=None, help="Directory to save predictions and embeddings (auto timestamped if None).")

    # Pre-processing parameters
    parser.add_argument("--ecfp_radius", type=int, default=2, help="Radius for ECFP fingerprint generation.")
    parser.add_argument("--fp_length", type=int, default=2048, help="Length of ECFP fingerprint vector.")
    parser.add_argument("--no_descriptors", action="store_true", help="Disable molecular descriptor computation.")

    # Inference parameters
    parser.add_argument("--protein_batch_size", type=int, default = 64, help="Batch size for protein embedding.")
    parser.add_argument("--ligand_batch_size", type=int, default = 1024, help="Batch size for ligand embedding.")
    parser.add_argument("--num_workers", type=int, default=64, help="Workers for dataset loading.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference.")

    # Save flags
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed datasets.")
    parser.add_argument("--save_cleaned_pdbs", action="store_true", help="Save cleaned PDB files.")
    parser.add_argument("--save_complex_info", action="store_true", help="Save additional complex info.")
    parser.add_argument("--no_save_embeddings", action="store_true", help="Disable saving protein and ligand embeddings.")
    parser.add_argument("--save_pymol_visualizations", action="store_true", help="Save PyMOL-ready pocket scenes.")

    # Parse arguments
    args = parser.parse_args()

    if args.dataset_dir.startswith("./data/") or args.dataset_dir.startswith("data/"):
        data_dir = args.dataset_dir
    else:
        data_dir = os.path.join(args.dataset_dir, "ConGLUDe", "data")

    if args.smiles_path is None:
        args.smiles_path = os.path.join(data_dir, "info", "smiles.txt")

    if args.results_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        dataset_name = os.path.basename(os.path.normpath(args.dataset_dir))
        if args.dataset_dir.startswith("./data/") or args.dataset_dir.startswith("data/"):
            results_dir = os.path.join("results", dataset_name, timestamp)
        else:
            results_dir = os.path.join(args.dataset_dir, "ConGLUDe", "results", timestamp)
    else:
        results_dir = args.results_dir

    os.makedirs(results_dir, exist_ok=True)

    save_embeddings = not args.no_save_embeddings
    calc_descriptors = not args.no_descriptors

    # Embed proteins and pockets
    protein_embedder = ProteinEmbedder(
        checkpoint_path = args.checkpoint_path,
        dataset_dir = args.dataset_dir,
        pdb_dir = args.pdb_dir,
        results_dir = results_dir,
        batch_size = args.protein_batch_size,
        overwrite = args.overwrite,
        num_workers = args.num_workers,
        save_cleaned_pdbs = args.save_cleaned_pdbs,
        save_complex_info = args.save_complex_info,
        save_embeddings = save_embeddings,
        save_pymol_visualizations = args.save_pymol_visualizations,
        device = args.device
    )

    protein_names, protein_embeddings, pocket_names, pocket_embeddings = protein_embedder.embed()

    # Embed ligands
    ligand_embedder = LigandEmbedder(
        checkpoint_path = args.checkpoint_path,
        dataset_dir = args.dataset_dir,
        smiles_path = args.smiles_path,
        results_dir = results_dir,
        ecfp_radius = args.ecfp_radius,
        fp_length = args.fp_length,
        calc_descriptors = calc_descriptors,
        batch_size = args.ligand_batch_size,
        overwrite = args.overwrite,
        num_workers = args.num_workers,
        save_embeddings = save_embeddings,
        device = args.device
    )

    ligand_embeddings = ligand_embedder.embed()

    # Split ligand embeddings into protein and pocket parts
    encoded_ligands_p = torch.nn.functional.normalize(ligand_embeddings[:,:(ligand_embeddings.shape[1]//2)], dim=1)
    encoded_ligands_b = torch.nn.functional.normalize(ligand_embeddings[:,(ligand_embeddings.shape[1]//2):], dim=1)

    # Make predictions (rows = ligands, columns = proteins/pockets)
    vs_preds = encoded_ligands_p @ protein_embeddings.t()
    pr_preds = encoded_ligands_b @ pocket_embeddings.t()

    # Save predictions
    predictions_path = os.path.join(results_dir, "predictions")
    os.makedirs(predictions_path, exist_ok=True)

    np.save(os.path.join(predictions_path, "vs_predictions.npy"), vs_preds.cpu().numpy())
    np.save(os.path.join(predictions_path, "pr_predictions.npy"), pr_preds.cpu().numpy())
