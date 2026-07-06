import argparse

from conglude.utils.data_processing import PDBGraphProcessor, LigandProcessor


OVERWRITE = False
NUM_WORKERS = 64
SAVE_CLEANED_PDBS = False
SAVE_COMPLEX_INFO = False

DATASET_GROUPS = {
    "test": ["asd", "coach420", "dude", "holo4k", "kinobeads", "litpcba", "pdbbind_refined", "pdbbind_time", "posebusters"],
    "train": ["SB_train_val", "LB_train_val"],
    "train_val": ["SB_train_val", "LB_train_val"],
    "vs": ["dude", "litpcba"],
    "tf": ["kinobeads"],
    "pp": ["coach420", "holo4k", "pdbbind_refined"],
    "pr": ["asd", "pdbbind_time", "posebusters"],
}

TRAIN_DATASETS = {"SB_train_val", "LB_train_val"}

config_dict = {
    "LB_train_val": {},
    "SB_train_val": {"extract_ligands": "all", "labeled_smiles": "none"},
    "litpcba": {"multi_pdb_targets": True},
    "dude": {},
    "kinobeads": {},
    "coach420": {"extract_ligands": "known", "select_chains": "chain_id", "labeled_smiles": "none", "multi_ligand": True, "calc_mol_feats": False},
    "holo4k": {"extract_ligands": "known", "labeled_smiles": "none", "multi_ligand": True, "calc_mol_feats": False},
    "pdbbind_refined": {"extract_ligands": "none", "labeled_smiles": "none", "calc_mol_feats": False, "load_pocket": True},
    "pdbbind_time": {"extract_ligands": "all", "labeled_smiles": "none", "multi_ligand": True},
    "posebusters": {"extract_ligands": "all", "labeled_smiles": "none", "multi_ligand": True},
    "asd": {"extract_ligands": "combined", "labeled_smiles": "none", "multi_ligand": True},
}

parser = argparse.ArgumentParser(description="Process downloaded datasets for ConGLUDe.")
parser.add_argument("--dataset_name", type=str, default=None,
                    help="Dataset or group to process (default: all test datasets).")
args = parser.parse_args()

if args.dataset_name in DATASET_GROUPS:
    datasets_to_process = DATASET_GROUPS[args.dataset_name]
elif args.dataset_name is not None:
    datasets_to_process = [args.dataset_name]
else:
    datasets_to_process = DATASET_GROUPS["test"]

for dataset in datasets_to_process:
    if dataset not in config_dict:
        print(f"Unknown dataset: {dataset}, skipping.")
        continue

    print(f"Processing dataset: {dataset}")

    config = config_dict[dataset].copy()
    if dataset in TRAIN_DATASETS:
        config["dataset_dir"] = f"./data/datasets/train_val_datasets/{dataset}"
    else:
        config["dataset_dir"] = f"./data/datasets/test_datasets/{dataset}"

    config["overwrite"] = OVERWRITE
    config["num_workers"] = NUM_WORKERS
    config["save_cleaned_pdbs"] = SAVE_CLEANED_PDBS
    config["save_complex_info"] = SAVE_COMPLEX_INFO

    pdb_graph_processor = PDBGraphProcessor(**config)
    pdb_graph_processor.process()

    if "calc_mol_feats" not in config or config["calc_mol_feats"]:
        ligand_processor = LigandProcessor(dataset_dir=config["dataset_dir"])
        ligand_processor.process()
