from conglude.utils.data_processing import PDBGraphProcessor, LigandProcessor


OVERWRITE = False
NUM_WORKERS = 64
SAVE_CLEANED_PDBS = False
SAVE_COMPLEX_INFO = False

config_dict = {
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


for dataset in ["litpcba", "dude", "kinobeads", "coach420", "holo4k", "pdbbind_refined", "pdbbind_time", "posebusters", "asd"]:

    print(f"Processing dataset: {dataset}")
    
    config = config_dict[dataset]
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
