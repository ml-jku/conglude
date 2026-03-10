import os
import requests
import shutil
import joblib
import lmdb
import pickle
from tqdm import tqdm

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Callable

import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import RobustScaler

from Bio.PDB import PDBParser, PDBIO, Model, Chain, Structure, Residue
from Bio.PDB.Polypeptide import is_aa
from Bio import BiopythonWarning
import warnings
from io import StringIO
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit import Chem, RDLogger
import esm

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore", category=BiopythonWarning)

from conglude.utils.common import read_list_from_txt, write_list_to_txt, read_json, write_json, execute_in_parallel
from conglude.utils.constants import THREE_TO_ONE



class PDBGraphProcessor:
    """
    A class for processing PDB files into graph representations, extracting ligands, and generating SMILES strings.

    This class provides methods to download PDB files, extract protein and ligand residues, generate SMILES strings, and save processed data.
    It supports parallel processing and can handle various configurations for ligand extraction and SMILES generation.

    Parameters
    ----------
    dataset_dir: str
        Directory where the dataset is stored.
    pdb_dir: str
        Directory where PDB files are stored.
    common_data_dir: str
        Directory where common data (e.g., reference databases) is stored.
    overwrite: bool
        Whether to overwrite existing files.   
    num_workers: int
        Number of worker processes for parallel processing (no parallization if 1).
    extract_ligands: str
        How to extract ligands from PDB files.
        - "all": extract all valid ligands
        - "known": extract only ligands from a given list of ligand IDs
        - "combined": a combination of "all" and "known"
        - "from_file": extract ligands from separate .mol2 ligand files
        - "none": extract no ligands
    labeled_smiles: str
        Whether lists of labeled active and inactive SMILES strings with or without affinity values are available for processing.
        - "none": no ligand-based data
        - "binary": lists of active and inactive ligands
        - "affinity": data frame with ligands and affinity labels
    select_chains: str
        How to select protein chains.
        - "all": use all protein chains
        - "chain_id": use chains with specified IDs
        - "uniprot": use all chains that match a certain Uniprot ID
        - "uniprot_single": use the first chain that matches a certain Uniprot ID
        - "closest": use all chains that are within a certain radius of the ligand
    multi_ligand: bool
        Whether to process multiple ligands per PDB file in one sample
    load_pocket: bool
        Whether the coordinates of the binding site center should be loaded from a dictionary
    multi_pdb_targets: bool
        Whether one target can have multiple PDB structures; if True, target2pdb_dict must be provided.
    calc_mol_feats: bool
        Whether to calculate ligand features.
    dist_to_ligand: float
        Maximum distance in Ångström a protein chain can have from all ligand atoms to be selected if `select_chains` is "closest".
    min_subunit_size: int
        Minimum size of a protein chain to be considered a subunit.
    max_num_subunits: int
        Maximum number of subunits allowed in a complex.
    pocket_cutoff: float
        Maximum distance of protein residues from ligand atoms to be considered as part of the binding site.
    neighbor_dist_cutoff: float
        Cutoff distance for defining neighboring residues in the protein graph.
    max_neighbors: int
        Maximum number of neighbors for each node in the graph.
    sanitize_smiles: bool
        Whether to sanitize the molecule with RDKit.
    remove_hs_smiles: bool
        Whether to remove explicit hydrogens.
    canonical_smiles: bool
        Whether to generate canonical SMILES.
    isomeric_smiles: bool
        Whether to include stereochemistry.
    kekule_smiles: bool
        Whether to return Kekulé SMILES.
    save_cleaned_pdbs: bool
        Whether to save cleaned PDB files after processing.
    save_complex_info: bool
        Whether to save complex information after processing.
    device: str
        Device to be used for computation of ESM features (e.g., "cpu" or "cuda").
    """

    def __init__(
        self,
        dataset_dir: str,
        pdb_dir: str = "data/pdb_files",
        common_data_dir: str = "data/datasets/common",

        overwrite: bool = True,
        num_workers: int = 64,

        extract_ligands: str = "none", # "all", "none", "known", "combined", "from_file"
        labeled_smiles: str = "binary", # "none", "binary", "affinity"
        select_chains: str = "all", # "all", "closest", "chain_id", "uniprot", "uniprot_single"
        multi_ligand: bool = True,
        load_pocket: bool = False,
        multi_pdb_targets: bool = False,
        calc_mol_feats: bool = True,
        
        dist_to_ligand: float = 4.0,
        min_subunit_size: int = 50,
        max_num_subunits: int = 24,                 
        pocket_cutoff: float = 4.0,
        neighbor_dist_cutoff: float = 10.0,
        max_neighbors: int = 10,

        sanitize_smiles: bool = True,
        remove_hs_smiles: bool = True,
        canonical_smiles: bool = True,
        isomeric_smiles: bool = True,
        kekule_smiles: bool = False,

        save_cleaned_pdbs: bool = False,
        save_complex_info: bool = False,
        device: str = "cuda:0"
    ) -> None:
        
        self.dataset_dir = dataset_dir
        self.pdb_dir = pdb_dir
        self.common_data_dir = common_data_dir

        self.overwrite = overwrite
        self.num_workers = num_workers

        self.extract_ligands = extract_ligands 
        self.labeled_smiles = labeled_smiles       
        self.select_chains = select_chains 
        self.multi_ligand = multi_ligand
        self.load_pocket = load_pocket
        self.multi_pdb_targets = multi_pdb_targets
        self.calc_mol_feats = calc_mol_feats

        self.dist_to_ligand = dist_to_ligand
        self.min_subunit_size = min_subunit_size
        self.max_num_subunits = max_num_subunits
        self.pocket_cutoff = pocket_cutoff
        self.neighbor_dist_cutoff = neighbor_dist_cutoff
        self.max_neighbors = max_neighbors

        self.sanitize_smiles = sanitize_smiles
        self.remove_hs_smiles = remove_hs_smiles
        self.canonical_smiles = canonical_smiles
        self.isomeric_smiles = isomeric_smiles
        self.kekule_smiles = kekule_smiles
        
        self.save_cleaned_pdbs = save_cleaned_pdbs
        self.save_complex_info = save_complex_info
        self.device = device

        # Save relevant metadata for later use
        self.meta_data_dict = {
            "extract_ligands": extract_ligands,
            "labeled_smiles": labeled_smiles,
            "select_chains": select_chains,
            "multi_ligand": multi_ligand,
            "multi_pdb_targets": multi_pdb_targets,
            "calc_mol_feats": calc_mol_feats,
            "dist_to_ligand": dist_to_ligand,
            "min_subunit_size": min_subunit_size,
            "max_num_subunits": max_num_subunits,
            "pocket_cutoff": pocket_cutoff,
        }

                    
    def load_valid_ligand_references(
        self
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load reference data for ligand validation from the common data directory. 
        The pre-processing code for reference data can be found in `moad.ipynb` and `ferla.ipynb` in `data_processing/common/`.

        Requires
        --------
        self.common_data_dir

        Returns
        -------
        moad_df: pandas.DataFrame
            A DataFrame of MOAD ligand data (from `moad.csv`).
        invalid_ligands: list
            A combined list of invalid ligand IDs (e.g. solvents, ions,...) derived from the MOAD database and a blog post by Mateo Ferla (https://blog.matteoferla.com/2019/11/go-away-glycerol.html).
        """

        # Load MOAD dataset
        moad_path = os.path.join(self.common_data_dir, "moad", "processed", "moad.csv")
        moad_df = pd.read_csv(moad_path)

        # Combine invalid ligands from MOAD and the Ferla blog post
        moad_invalid_path = os.path.join(self.common_data_dir, "moad", "processed", "invalid_ligands.txt")
        ferla_invalid_path = os.path.join(self.common_data_dir, "ferla", "processed", "invalid_ligands.txt")
        
        moad_invalid_ligands = read_list_from_txt(moad_invalid_path)
        ferla_invalid_ligands = read_list_from_txt(ferla_invalid_path)
        invalid_ligands = moad_invalid_ligands + ferla_invalid_ligands + ["UNK", "UNL", "UNX"]

        return moad_df, invalid_ligands
    

    def load_id2smiles(
        self
    ) -> Dict[str, str]:
        """
        Load ligand reference data from the Chemical Component Dictionary. 
        The pre-processing code for reference data can be found in `ccd.ipynb` in `data_processing/common/`.

        Requires
        --------
        self.common_data_dir

        Returns
        -------
        id2smiles_dict: dict
            A dictionary mapping ligand IDs to SMILES strings derived from the PDB's Chemical Component Dictionary (from `ccd_smiles.csv`).
        """

        # Load CCD ligand SMILES mappings
        ccd_path = os.path.join(self.common_data_dir, "ccd", "processed", "ccd_smiles.csv")
        ccd_df = pd.read_csv(ccd_path)
        id2smiles_dict = dict(zip(ccd_df["ligand_id"], ccd_df["smiles"]))

        return id2smiles_dict
    

    def handle_error(
            self, 
            protein_id: str, 
            comment: str, 
            ligand_id: str = ""
        ) -> None:
        """
        Logs an error with the specified PDB ID, comment and optional ligand ID, appending it to a CSV file.

        Requires
        --------
        self.dataset_dir

        Parameters
        ----------
        protein_id: str 
            The identifier of the protein associated with the error.
        comment: str
            Description of the error.
        ligand_id: str, default=""
            The identifier of the ligand associated with the error.
        """

        file_name = os.path.join(self.dataset_dir, "info", "processing_failed.csv")

        with open(file_name, "a") as f:
            f.write(f"{protein_id},{ligand_id},{comment}\n")


    def download_pdb(
        self, 
        pdb_id: str, 
    ) -> int:
        """
        Download a PDB file in `.pdb` format from the RCSB Protein Data Bank by its ID and save it to the given PDB directory.

        Parameters
        ----------
        pdb_id: str
            The 4-character PDB identifier (e.g., "1ABC").

        Requires
        --------
        self.pdb_dir

        Returns
        -------
        status_code: int
            The HTTP status code of the download request (200 if successful).
        """

        # Construct the download URL for the first-model PDB file (`.pdb1`)
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb1"

        # Send GET request to RCSB, enabling streaming for large files
        response = requests.get(url, stream=True)
        status_code = response.status_code

        # If download is successful, save the file in binary mode
        if status_code == 200:
            pdb_path = os.path.join(self.pdb_dir, f"{pdb_id}.pdb")
            with open(pdb_path, "wb") as file:
                for block in response.iter_content(chunk_size=1024):
                    if block:  # Skip keep-alive chunks
                        file.write(block)

        return status_code
    

    def download_alphafold(
        self,
        alphafold_id: str,
    ) -> int:
        """
        Checks if the given UniProt ID exists in the AlphaFold database and downloads the corresponding structure (v4) if available.

        Parameters
        ----------
        alphafold_id: str 
            The ID to query (AF-{uniprot_id}).

        Requires
        --------
        self.pdb_dir

        Returns
        -------
        status_code: int
            The HTTP status code of the download request (200 if successful).
        """

        # AlphaFold DB v4 base URL
        url = f"https://alphafold.ebi.ac.uk/files/{alphafold_id}-F1-model_v4.pdb"

        # Send GET request to AF DB, enabling streaming for large files
        response = requests.get(url, stream=True)
        status_code = response.status_code

        # If download is successful, save the file in binary mode
        if status_code == 200:
            pdb_path = os.path.join(self.pdb_dir, f"{alphafold_id}.pdb")
            with open(pdb_path, "wb") as file:
                for block in response.iter_content(chunk_size=1024):
                    if block:  # Skip keep-alive chunks
                        file.write(block)

        return status_code
    
    
    def dissect_structure(
        self, 
        structure: Structure.Structure,
        ligand_ids: str = None,
    ) -> Tuple[Dict[str, List[Residue.Residue]], Dict[str, List[Residue.Residue]]]:
        """
        Iterate through all chains in the given structure, classifying residues into protein residues (standard amino acids) and ligands (non-water, non-amino acid molecules with >= 4 heavy atoms).
        
        Parameters
        ----------
        structure: Bio.PDB.Structure.Structure
            The parsed PDB structure object.

        Requires
        --------
        self.extract_ligands
  
        Returns
        -------
        protein_chains: dict
            Mapping of chain ID to list of protein residues.
        ligands: dict
            Mapping of ligand ID to list of ligand residues.
        """
        
        protein_chains = {}
        ligands = {}

        ligand_chains = []
        if ligand_ids is not None:
            for ligand_id in ligand_ids:
                if str(ligand_id).startswith("chain"):
                    ligand_chains.append(ligand_id.split("_")[1])

        # Iterate over all models and chains in the structure
        for model in structure:
            for chain in model:
                chain_id = chain.id
                chain_residues = []
                ligand_residues = {}

                # Classify residues in the chain
                for residue in chain:
                    res_name = residue.resname

                    if self.extract_ligands in ["known", "all", "combined"] and chain_id in ligand_chains:
                        if len(ligand_residues) == 0:
                            ligand_residues[f"chain_{chain_id}_{res_name}_{len(ligand_residues)}"] = residue.copy()
                        else:
                            ligand_residues[f"{res_name}_{len(ligand_residues)}"] = residue.copy()

                    # Append standard amino acid residues to protein chain
                    elif residue.id[0] == " " and is_aa(residue, standard=True):
                        chain_residues.append(residue.copy())
                    
                    # Append ligand candidates (no water, no amino acid, more than 4 heavy atoms) to ligand residues if self.extract_ligands is not "none"
                    elif (
                        ((ligand_ids is not None
                         and res_name in ligand_ids)
                        or (res_name != "HOH"
                        # and res_name not in D_AMINO_ACIDS
                        # and not is_aa(residue,  standard=True)
                        and sum(1 for atom in residue if atom.element != "H") >= 4))
                        and self.extract_ligands in ["known", "all", "combined"]
                    ):
                        ligand_residues[f"{res_name}_{len(ligand_residues)}"] = residue.copy()

                if len(chain_residues) != 0:
                    protein_chains[chain_id] = chain_residues

                # Process ligands only if extraction is enabled
                if self.extract_ligands in ["known", "all", "combined"] and ligand_residues:
                    # Get coordinates of all ligand heavy atoms
                    ligand_res_names = list(ligand_residues.keys())

                    if len(ligand_res_names) != 0:
                        ligand_res_coords = [
                            np.array([atom.coord for atom in ligand_residues[res_name] if atom.element != "H"]) 
                            for res_name in ligand_res_names
                        ]

                        # Determine connectivity between consecutive residues (counted as connected if distance is smaller than 2Å)
                        connected = []
                        for j in range(len(ligand_res_names)-1):
                            dists = np.linalg.norm(
                                ligand_res_coords[j][:, np.newaxis, :] - 
                                ligand_res_coords[j+1][np.newaxis, :, :], 
                                axis=-1
                            )
                            connected.append(np.any(dists < 2))

                        # Group connected ligand residues into ligand entries
                        ligand_chain_name = "_".join(ligand_res_names[0].split("_")[:-1])
                        ligand_chain_residues = [ligand_residues[ligand_res_names[0]]]

                        for j in range(1, len(ligand_res_names)):
                            res_name = ligand_res_names[j]
                            if connected[j-1]:
                                ligand_chain_name += f" {res_name.split('_')[0]}"
                                ligand_chain_residues.append(ligand_residues[res_name])
                            else:
                                ligands[f"{ligand_chain_name}_{len(ligands)}"] = ligand_chain_residues
                                ligand_chain_name = "_".join(res_name.split("_")[:-1])
                                ligand_chain_residues = [ligand_residues[res_name]]

                        ligands[f"{ligand_chain_name}_{len(ligands)}"] = ligand_chain_residues

            # Extract only the first model
            break

        return protein_chains, ligands


    def load_ligand_from_file(
        self,
        protein_id: str,
    ) -> Dict[str, List[Residue.Residue]]:
        """
        Load ligand structure from a .pdb or .mol2 file and return it as a dictionary of residues.

        Parameters
        ----------
        protein_id : str
            Unique identifier of the protein, used to locate the corresponding ligand file.

        Requires
        --------
        self.dataset_dir, self.parser

        Returns
        -------
        ligands: Dict[str, List[Residue.Residue]]
            A dictionary mapping the ligand name (string concatenation of residue names) to a list of Bio.PDB Residue objects representing the ligand.
        """

        ligand_dir = os.path.join(self.dataset_dir, "raw", "ligand_files")

        if os.path.exists(os.path.join(ligand_dir, f"{protein_id}_ligand.pdb")):
            # Parse PDB file with PDB parser
            structure = self.parser.get_structure("ligand", os.path.join(ligand_dir, f"{protein_id}_ligand.pdb"))
        
        elif os.path.exists(os.path.join(ligand_dir, f"{protein_id}_ligand.mol2")):
            # Load MOL2 with RDKit
            mol = Chem.MolFromMol2File(os.path.join(ligand_dir, f"{protein_id}_ligand.mol2"), sanitize=False, removeHs=True)
            if mol is None:
                print(f"Could not read MOL2 file for {protein_id} ligand.")
                return []

            # Generate PDB block in memory
            try:
                pdb_block = Chem.MolToPDBBlock(mol)
            except:
                print(protein_id)

            # Parse PDB block
            structure = self.parser.get_structure("ligand", StringIO(pdb_block))

        else:
            print(f"No ligand file found for {protein_id}.")

        # Extract residues
        residues = [res for res in structure.get_residues()]

        # Extract ligand name and save ligand as dictionary
        ligand_name = ""
        for residue in residues:
            ligand_name += f"{residue.resname}_"

        ligands = {ligand_name[:-1]: residues}

        return ligands


    def get_uniprot_ids(
            self, 
            pdb_id: str
        ) -> Dict[str, str]:
        """
        Retrieve the UniProt IDs associated with each chain in a given PDB structure.

        Parameters
        ----------
        pdb_id: str
            The 4-character PDB identifier (e.g., "1ABC").

        Returns
        -------
        chain_uniprot_map: dict
            Mapping of chain ID (str) to UniProt ID (str); empty dictionary if the PDB ID is not found or the request fails.
        """

        pdb_id = pdb_id.lower()

        url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
        response = requests.get(url)

        # Check request success
        if response.status_code != 200:
            return {}
        
        data = response.json()

        # Ensure PDB ID exists in API response
        if pdb_id not in data:
            return {}  
        
        # Extract mappings from API response
        uniprot_mappings = data[pdb_id]["UniProt"]
        chain_uniprot_map = {}
                
        for uniprot_id, details in uniprot_mappings.items():
            for mapping in details["mappings"]:
                chain = mapping["chain_id"]
                chain_uniprot_map[chain] = uniprot_id
                
        return chain_uniprot_map
    

    def get_coord_df(
            self, 
            chain: List[Residue.Residue], 
            chain_id: str
        ) -> pd.DataFrame:
        """
        Extract atomic coordinates from a list of Biopython Residue objects into a DataFrame.

        Parameters
        ----------
        chain : list(Bio.PDB.Residue.Residue)
            The list of Biopython Residue objects.
        chain_id : str
            The chain identifier.

        Returns
        -------
        coord_df: pandas.DataFrame
            DataFrame containing atomic information with columns:
            - "element": str, chemical element symbol
            - "x", "y", "z": float, Cartesian coordinates in Ångström
            - "res_name": str, residue name (3-letter code)
            - "res_number": int, residue sequence number
            - "atom_name": str, atom name (PDB format)
            - "chain_id": str, chain identifier
        """

        coord_list = []
        
        # Iterate through all residues in the chain
        for residue in chain:
            for atom in residue:
                # Retrieve atom coordinates (x, y, z)
                atom_coords = atom.get_coord()

                # Append atom data to list
                coord_list.append([
                    atom.element,           # Chemical element
                    atom_coords[0],         # X coordinate
                    atom_coords[1],         # Y coordinate
                    atom_coords[2],         # Z coordinate
                    residue.resname,        # Residue name (3-letter)
                    residue.id[1],          # Residue number
                    atom.full_id[4][0],     # Atom name
                    chain_id                # Chain ID
                ])

        # Create DataFrame from collected atom data
        coord_df = pd.DataFrame(
            coord_list,
            columns=[
                "element",
                "x",
                "y",
                "z",
                "res_name",
                "res_number",
                "atom_name",
                "chain_id",
            ],
        )

        return coord_df
    

    def filter_ligands(
        self, 
        protein_id: str, 
        ligands: Dict[str, List], 
        ligand_ids: Optional[List[str]] = None
    ) -> Dict[str, List]:
        """
        Filter ligand residues based on known valid ligand IDs from the MOAD database or ligand size (max. 5 residues per ligand).

        Parameters
        ----------
        protein_id: str
            Identifier of the protein (used to query known valid ligands).
        ligands: dict
            Dictionary mapping ligand names to lists of ligand residues.
        ligand_ids: list or None, optional
            List of ligand IDs to filter by. If None, valid ligands from `self.moad_df` are used.

        Requires
        --------
        self.extract_ligands, self.moad_df, self.invalid_ligands

        Returns
        -------
        dict
            Filtered ligands dictionary with ligand names as keys and residue lists as values.
        """
        assert not(self.extract_ligands == "known" and (ligand_ids is None)), "Flag 'extract_ligands' is set to 'known', but no list of ligand IDs was provided."
        
        if self.extract_ligands == "known":
            filtered_ligand_ids = ligand_ids
        
        else:
            # Initialize list of valid ligands from MOAD data if applicable
            entry = self.moad_df.loc[
                (self.moad_df["pdb_id"] == protein_id) & (self.moad_df["validity"] == "valid")
            ]
            valid_ligands = list(entry["ligand_id"])

            if len(valid_ligands) != 0:
                filtered_ligand_ids = valid_ligands
    
                # Combine ligand_ids with valid ligands if self.extract_ligands is "combined" 
                if self.extract_ligands == "combined" and ligand_ids is not None:
                    filtered_ligand_ids += ligand_ids
                    filtered_ligand_ids = list(set(filtered_ligand_ids))

            else:
                filtered_ligand_ids = []
                for ligand_name, ligand in ligands.items():
                    if len(ligand) <= 5 and not "_".join(ligand_name.split("_")[:-1]) in self.invalid_ligands:
                        filtered_ligand_ids.append("_".join(ligand_name.split("_")[:-1]))

                if self.extract_ligands == "combined" and ligand_ids is not None:
                    filtered_ligand_ids += ligand_ids
                    filtered_ligand_ids = list(set(filtered_ligand_ids))

        filtered_ligands = {}
        for ligand_name, ligand in ligands.items():
            if "_".join(ligand_name.split("_")[:-1]) in filtered_ligand_ids:
                filtered_ligands[ligand_name] = ligand

        return filtered_ligands


    def get_closest_chains(
        self,
        chain_coord_dfs: Dict[str, pd.DataFrame],
        ligand_coord_array: np.ndarray
    ) -> Dict[str, List[str]]:
        """
        Identify protein chains that have at least one atom within a certain radius (`self.dist_to_ligand`) of any ligand atom.

        Parameters
        ----------
        chain_coord_dfs : dict[str, pd.DataFrame]
            Dictionary mapping chain IDs to DataFrame containing atom coordinates and other atom-wise information.
        ligand_coord_array : np.ndarray
            NumPy array of shape (n_atoms, 3) for ligand coordinates.

        Requires
        --------
        self.dist_to_ligand

        Returns
        -------
        selected_chains: list[str]
            List of protein chain IDs that are within a radius of `self.dist_to_ligand` of the given ligand.
        """

        selected_chains = []          

        for chain_id, chain_coord_df in chain_coord_dfs.items():
            # Compute pairwise distances between protein chain atoms and ligand atoms
            chain_coord_array = chain_coord_df[["x", "y", "z"]].to_numpy()
            dists = np.linalg.norm(
                chain_coord_array[:, np.newaxis, :] - ligand_coord_array[np.newaxis, :, :],
                axis=-1
            )

            # Select chains for which any atom is closer than self.dist_to_ligand to the ligand
            if np.any(dists < self.dist_to_ligand):
                selected_chains.append(chain_id)

        return selected_chains
    

    def get_pocket(
        self,
        ligand_coord_array: np.ndarray,
        chain_coord_dfs: Dict[str, pd.DataFrame],
        ca_coord_dfs: Dict[str, pd.DataFrame],
    ) -> Tuple[Optional[np.ndarray], pd.DataFrame]:
        """
        Identify the binding site of a ligand relative to protein chains.

        Parameters
        ----------
        ligand_coord_array: np.ndarray
            Cartesian coordinates of the ligand atoms.
        chain_coord_dfs: dict[str, pd.DataFrame]
            Dictionary mapping chain IDs to coordinate DataFrames of protein chains.
        ca_coord_dfs: dict[str, pd.DataFrame]
            Dictionary mapping chain IDs to coordinate DataFrames of protein chains reduced to C-alpha atoms.

        Requires
        --------
        self.pocket_cutoff

        Returns
        -------
        pocket_center: np.ndarray
            Mean coordinates of binding site (shape (3,)) or None if no binding site residues are found.
        ca_coord_dfs_bs: dict[str, pd.DataFrame]
            Dictionary mapping chain IDs to coordinate DataFrames of protein chains reduced to C-alpha atoms with binding site annotation.
        """

        ca_coord_dfs_bs = {}
        pocket_df = []
        
        for chain_id, chain_coord_df in chain_coord_dfs.items():
            # Compute minimum distance between ligand and each protein atoms
            chain_coord_array = chain_coord_df[["x", "y", "z"]].to_numpy()
            dists = np.linalg.norm(
                ligand_coord_array[:, np.newaxis, :] - chain_coord_array[np.newaxis, :, :], 
                axis=-1
            )
            min_dists = dists.min(axis=0)

            # Mark atoms within binding site cutoff
            chain_coord_df["min_dist_to_ligand"] = min_dists
            chain_coord_df["pocket"] = chain_coord_df["min_dist_to_ligand"] <= self.pocket_cutoff

            # Identify binding site residues
            binding_residues = (
                chain_coord_df.groupby(["res_number"])["pocket"]
                .any()
                .reset_index()
            )

            #  Add binding site information to DataFrame with only C-alpha atoms 
            ca_coord_df = ca_coord_dfs[chain_id]
            ca_coord_df = ca_coord_df.merge(binding_residues, on=["res_number"])

            # Ensure consistent column names (merge adds _x and _y suffixes if needed)
            ca_coord_df = ca_coord_df.rename(columns={"pocket_y": "pocket"})
            
            ca_coord_dfs_bs[chain_id] = ca_coord_df
            pocket_df.append(ca_coord_df.loc[lambda x: x["pocket"] == True, ["x", "y", "z"]])
            
        pocket_df = pd.concat(pocket_df)

        # Calculate binding site center
        if len(pocket_df) != 0:
            pocket_center = torch.tensor(pocket_df.mean().values, dtype=torch.float32).unsqueeze(0)
            assert pocket_center.shape == (1, 3)
        else:
            pocket_center = None

        return pocket_center, ca_coord_dfs_bs
    

    def mol2smiles(
        self,
        mol: Chem.Mol,
    ) -> Optional[str]:
        """
        Standardize a molecule and return its cleaned SMILES representation.
    
        Parameters
        ----------
        mol: rdkit.Chem.Mol
            Input molecule.

        Requires
        --------
        self.sanitize_smiles, self.remove_hs_smiles, self.canonical_smiles, self.isomeric_smiles, self.kekule_smiles

        Returns
        -------
        smiles: str or None
            Cleaned SMILES string, or None if processing fails.
        """

        try:
            # Optionally sanitize the molecule (check for valence, aromaticity, etc.)
            if self.sanitize_smiles:
                Chem.SanitizeMol(mol)

            # Optionally remove explicit hydrogens from the molecule
            if self.remove_hs_smiles:
                mol = Chem.RemoveHs(mol, implicitOnly=False)

            # Optionally kekulize the molecule (convert aromatic bonds to alternating single/double bonds)
            if self.kekule_smiles:
                Chem.Kekulize(mol, clearAromaticFlags=True)

            # Convert molecule to a SMILES string with specified options
            smiles = Chem.MolToSmiles(
                mol,
                canonical=self.canonical_smiles,      # Use canonical form if enabled
                isomericSmiles=self.isomeric_smiles,  # Encode stereochemistry/isomers if enabled
                kekuleSmiles=self.kekule_smiles       # Encode kekulized form if enabled
            )

            return smiles

        except Exception:
            # Return None if any error occurs during processing (invalid molecule, etc.)
            return None
        

    def get_smiles(
        self,
        ligand: list[Residue.Residue],
        ligand_id: str,
    ) -> Optional[str]:
        """
        Attempt to look up the ligand in the Chemical Component Dictionary (via `self.id2smiles_dict`); if not found, reconstruct the ligand from the provided residues and parse it via RDKit.

        Parameters
        ----------
        ligand: list[Bio.PDB.Residue.Residue]
            List of Biopython Residue objects representing the ligand.
        ligand_id: str
            Ligand identifier.

        Requires
        --------
        self.id2smiles_dict
        self.mol2smiles()
            self.sanitize_smiles, self.remove_hs_smiles, self.canonical_smiles, self.isomeric_smiles, self.kekule_smiles

        Returns
        -------
        smiles: str or None
            SMILES representation if successful, otherwise None.
        """

        # Check if ligand is in Chemical Component Dictionary, if so, get SMILES from there
        if self.extract_ligands != "from_file" and ligand_id in self.id2smiles_dict:
            reference_smiles = self.id2smiles_dict[ligand_id]
            mol = Chem.MolFromSmiles(reference_smiles)

        # Extract SMILES from PDB structure
        else:
            # Build temporary in-memory PDB structure from residues
            structure = Structure.Structure("X")
            model = Model.Model(0)
            chain = Chain.Chain("A")
            for res in ligand:
                chain.add(res.copy())  # Copy to avoid modifying original
            model.add(chain)
            structure.add(model)

            # Serialize structure to PDB string in memory
            io = PDBIO()
            io.set_structure(structure)
            pdb_str = StringIO()
            io.save(pdb_str)
            pdb_block = pdb_str.getvalue()

            # Parse PDB block to RDKit molecule
            mol = Chem.MolFromPDBBlock(pdb_block, sanitize=True)

        # Standardize molecule and get its SMILES representation
        smiles = self.mol2smiles(mol)

        return smiles
    

    def save_pdb_files(
        self,
        protein_chains: Dict[str, List[Residue.Residue]],
        sample_name: str,
        ligands: Optional[Dict[str, List[Residue.Residue]]] = None
    ) -> None:
        """
        Save cleaned protein and optional ligand structures to PDB files.

        Parameters
        ----------
        protein_chains : dict[str, list[Bio.PDB.Residue.Residue]]
            Dictionary mapping chain IDs to lists of protein residues.
        sample_name : str
            Identifier used for the file names.
        ligands : dict[str, list[Bio.PDB.Residue.Residue]], optional
            Dictionary mapping ligand names to lists of ligand residues to save as separate PDB files. If None, only the protein PDB is saved.
        """

        # Create the output directory for this sample
        output_dir = os.path.join(self.dataset_dir, "processed", "cleaned_pdbs", sample_name)
        os.makedirs(output_dir, exist_ok=True)

        # Create an empty Structure object with one model
        structure = Structure.Structure("protein")
        model = Model.Model(0)
        structure.add(model)

        # For each chain add residues and add to the model
        for chain_id, residues in protein_chains.items():
            
            chain = Chain.Chain(chain_id)
            for residue in residues:
                chain.add(residue)
            model.add(chain)

        # Write the protein structure to disk
        io = PDBIO()
        io.set_structure(structure)
        io.save(os.path.join(output_dir, "protein.pdb"))

        if ligands is not None and self.extract_ligands != "from_dict":
            
            for ligand_name, ligand in ligands.items():
                # Create an empty Structure object with one model
                ligand_structure = Structure.Structure("ligand")
                ligand_model = Model.Model(0)
                ligand_structure.add(ligand_model)

                # Add ligand residues to chain and chain to the model
                chain = Chain.Chain("A")
                for residue in ligand:
                    chain.add(residue)
                ligand_model.add(chain)

                # Write the ligand structure to disk
                io.set_structure(ligand_structure)
                io.save(os.path.join(output_dir, f"{ligand_name}_ligand.pdb"))


    def process_pdb(
        self,
        protein_id: str,
    ) -> List[Dict[str, object]]:
        """
        Main pipeline for processing a protein structure from PDB (or a pre-existing file).

        Depending on the settings (ligand extraction, chain selection, saving cleaned PDBs),
        this function:
        - ownloads and parses the PDB if needed
        - Extracts protein chains and ligands
        - Filters and selects relevant ligands and chains
        - Computes binding site coordinates
        - Saves intermediate data and metadata
        - Returns prepared "complex objects" for downstream use

        Parameters
        ----------
        protein_id: str
            Identifier of the protein to process.

        Requires
        --------
        self.dataset_dir, self.pdb_dir
        self.multi_pdb_targets, self.extract_ligands, self.select_chains,
        self.pdb2target_dict, self.protein_ligand_dict, self.protein_chain_dict, self.meta_data_dict
        self.parser
        self.min_subunit_size, self.max_num_subunits, self.dist_to_ligand, self.save_cleaned_pdbs, self.multi_ligand
        self.handle_error(), self.download_pdb(), self.dissect_structure(), self.load_ligand_from_file(), self.get_uniprot_ids(), self.get_coord_df(), self.save_pdbs()
        self.filter_ligands()
            self.moad_df, self.invalid_ligands
        self.get_pocket():
            self.pocket_cutoff
        self.get_smiles():
            self.id2smiles_dict
            self.mol2smiles()
                self.sanitize_smiles, self.remove_hs_smiles, self.canonical_smiles, self.isomeric_smiles, self.kekule_smiles

        Returns
        -------
        complex_objects: list[dict[str, object]] or None
            A list of complex objects, where each complex object is a dictionary containing:
            - name (str): Sample name including protein and ligand identifiers
            - residue_coordinates (torch.Tensor): 3D coordinates of selected protein chains
            - pocket_labels (torch.Tensor, optional): Boolean labels marking C-alpha atoms in the binding site (if ligands present)
            - pocket_center (torch.Tensor, optional): Mean coordinates of the binding site center
            - ligand_coordinates (torch.Tensor, optional): 3D coordinates of the ligand atoms
            - info_dict (dict): Metadata including protein/ligand identifiers, sequences, Uniprot IDs, and SMILES
            Returns None if processing fails at any step.
        """
        
        # Determine the target name (e.g. Uniprot ID) from the pdb2target_dict if applicable
        target_name = self.pdb2target_dict[protein_id] if (self.multi_pdb_targets and protein_id in self.pdb2target_dict) else protein_id
      
        # Get predefined ligand IDs and chain mapping (if using specific extraction settings)
        ligand_ids = self.protein_ligand_dict[protein_id] if self.extract_ligands in ["known", "combined"] else None
        ligand_chain_dict = self.protein_chain_dict[protein_id] if self.select_chains == "chain_id" else None
              
        # Attempt to download PDB file if it doesn't exist
        if not os.path.exists(os.path.join(self.pdb_dir, f"{protein_id}.pdb")):
            if len(protein_id) == 4:
                status_code = self.download_pdb(protein_id)
                if status_code != 200:
                    self.handle_error(protein_id, "Could not download .pdb file from PDB.")
                    return None
            elif protein_id.startswith("AF-"):
                status_code = self.download_alphafold(protein_id)
                if status_code != 200:
                    self.handle_error(protein_id, "Could not download .pdb file from Alphafold DB.")
                    return None
            else:
                self.handle_error(protein_id, "Protein cannot be downloaded and no existing .pdb file was provided.")
                return None                
        
        # Attempt to load the protein structure with the already initialized PDB parser      
        try:
            structure = self.parser.get_structure("protein", os.path.join(self.pdb_dir, f"{protein_id}.pdb"))
        except:
            self.handle_error(protein_id, "Could not load/parse protein structure.")
            return None
        
        # Extract protein chains and ligands from the PDB structure       
        protein_chains, ligands = self.dissect_structure(structure, ligand_ids)

        # Load ligand from file
        if self.extract_ligands == "from_file":
            ligands = self.load_ligand_from_file(protein_id)

            if len(ligands) == 0:
                self.handle_error(protein_id, "Could not load ligand from file.")
                return None
        
        # Map each chain to its Uniprot ID (only applicable if protein_id is a PDB ID)
        if len(protein_id) == 4:
            chain_uniprot_map = self.get_uniprot_ids(protein_id)
        else:
            chain_uniprot_map = {}
        
        # Select relevant protein chains according to the `self.select_chains` flag
        if self.select_chains in ["uniprot", "uniprot_single"] and target_name in chain_uniprot_map.values(): 
            selected_chains = [chain for chain, uniprot in chain_uniprot_map.items() if uniprot == target_name and chain in protein_chains]
            if len(selected_chains) == 0:
                selected_chains = list(protein_chains.keys()) 
            else:
                if self.select_chains == "uniprot_single":
                    selected_chains = [selected_chains[0]]
                protein_chains = {k: v for k,v in protein_chains.items() if k in selected_chains}
            
        # Select all protein chains if `self.select_chains` is "all" or no chain was found for the given Uniprot ID
        elif self.select_chains in ["uniprot", "uniprot_single", "all"]:
            selected_chains = list(protein_chains.keys()) 

        # Build coordinate DataFrames for each chain
        chain_coord_dfs = {chain_id: self.get_coord_df(chain_obj, chain_id) for chain_id, chain_obj in protein_chains.items()}

        # Keep only C-alpha atoms
        ca_coords_dfs = {chain_id: chain_coord_df.loc[chain_coord_df["atom_name"] == "CA"].reset_index(drop=True) for chain_id, chain_coord_df in chain_coord_dfs.items()}
        
        # Get the amino acid sequence for each protein chain
        chain_sequences = {chain_id: c_alpha_coords_df["res_name"].map(THREE_TO_ONE).str.cat() for chain_id, c_alpha_coords_df in ca_coords_dfs.items()}

        # Extract ligands and corresponding binding sites if ligand extraction is enabled
        if self.extract_ligands != "none":

            if len(ligands) == 0 and self.extract_ligands != "from_dict":
                self.handle_error(protein_id, "No ligands could be extracted.")
                return None

            # Filter the extracted ligands based on given ligand IDs, reference databases and size
            if self.extract_ligands not in ["from_file", "from_dict"]:
                ligands = self.filter_ligands(protein_id, ligands, ligand_ids)
                if len(ligands) == 0:
                    self.handle_error(protein_id, "No ligands left after filtering.")
                    return None
            elif self.extract_ligands == "from_dict":
                ligands = {f"{protein_id}_lig": self.ligand_dict[protein_id][1]}
                                         
            complex_objects = []

            for ligand_name, ligand in ligands.items():

                # Convert ligand structure to coordinate DataFrame and numpy array
                if self.extract_ligands == "from_dict":
                    ligand_coord_array = np.array(ligand)
                else:
                    ligand_coord_df = self.get_coord_df(ligand, ligand_name)
                    ligand_coord_array = ligand_coord_df[["x", "y", "z"]].to_numpy()

                # Select all protein chains that are within a certain radius of the ligand
                if self.select_chains == "closest":
                    selected_chains = self.get_closest_chains(chain_coord_dfs, ligand_coord_array)
                    if len(selected_chains) == 0:
                        self.handle_error(protein_id, f"No protein chains are within {self.dist_to_ligand} Ångström of the ligand.", ligand_name.split("_")[0])
                        continue
                
                # Get all protein chains that were provided for this ligand
                elif self.select_chains == "chain_id":
                    if "_".join(ligand_name.split("_")[:-1]) in ligand_chain_dict:
                        selected_chains = [ligand_chain_dict["_".join(ligand_name.split("_")[:-1])]]
                    else:
                        self.handle_error(protein_id, f"Chain selection mode is 'chain_id', but no chain IDs were provided for this ligand.", ligand_name.split("_")[0])
                        continue

                # Create sample-specific names and subsets
                if self.select_chains == "all":
                    sample_name = protein_id if self.multi_ligand else f"{protein_id}_{ligand_name}"
                    sample_chain_coord_dfs = chain_coord_dfs                    
                    sample_chain_sequences = chain_sequences
                else:
                    sample_name = f"{protein_id}_{'-'.join(selected_chains)}" if self.multi_ligand else f"{protein_id}_{'-'.join(selected_chains)}_{ligand_name}"
                    sample_chain_coord_dfs = {k: v for k,v in chain_coord_dfs.items() if k in selected_chains}
                    sample_chain_sequences = {k: v for k,v in chain_sequences.items() if k in selected_chains}

                # Validate protein subunit count
                num_large_subunits = len([len(coord_df) for chain_id, coord_df in sample_chain_coord_dfs.items() if len(coord_df) >= self.min_subunit_size])
                if num_large_subunits not in range(1,self.max_num_subunits + 1):
                    self.handle_error(protein_id, f"Protein contains {num_large_subunits} large subunits.", ligand_name.split('_')[0])
                    continue                 

                # Extract binding site residues and center
                pocket_center, ca_coord_dfs_bs = self.get_pocket(ligand_coord_array, sample_chain_coord_dfs, ca_coords_dfs)
                if pocket_center is None and not self.load_pocket:
                    self.handle_error(protein_id, "No binding residues found for this ligand.", ligand_name)
                    continue
                elif self.load_pocket:
                    pocket_center = torch.tensor(self.pocket_dict[protein_id], dtype=torch.float32).unsqueeze(0)

                # Convert ligand to SMILES representation
                if self.calc_mol_feats:
                    if self.extract_ligands == "from_dict":
                        smiles = self.ligand_dict[protein_id][0]
                    else:
                        smiles = self.get_smiles(ligand, ligand_name.split("_")[0])
                        if smiles is None or smiles == "":
                            self.handle_error(protein_id, "Ligand could not be parsed to SMILES string.", ligand_name)
                            continue
                else:
                    smiles = ""
                
                # Save .pdb files of protein and ligand
                if self.save_cleaned_pdbs and not self.multi_ligand:
                    sample_chains = {k: v for k,v in protein_chains.items() if k in selected_chains}
                    self.save_pdb_files(sample_chains, sample_name, ligands={ligand_name: ligand})

                # Get Uniprot ID for each protein chain
                if len(protein_id) == 4:
                    uniprot_ids = [chain_uniprot_map.get(chain_id, None) for chain_id in selected_chains]
                elif len(protein_id) == 6:
                    uniprot_ids = [protein_id]
                elif protein_id.startswith("AF-"):
                    uniprot_ids = [protein_id[3:]]
                else:
                    uniprot_ids = [None]*len(selected_chains)
                
                # Save protein information in info_dict
                info_dict = {
                    "protein_id": protein_id,
                    "target_name": target_name,
                    "ligand_names": [ligand_name],
                    "uniprot_ids": uniprot_ids,
                    "chain_ids": selected_chains,
                    "chain_sequences": list(sample_chain_sequences.values()),
                    "ligand_smiles": [smiles],
                    "meta_data": self.meta_data_dict
                }

                # Summarize protein, ligand and binding site data in complex object
                complex_object = {
                    "name": sample_name,
                    "residue_coordinates": torch.tensor(np.concatenate([ca_coord_dfs_bs[chain_id][["x", "y", "z"]].to_numpy() for chain_id in selected_chains])),
                    "chain_indices": torch.repeat_interleave(torch.arange(len(info_dict["chain_sequences"])), torch.tensor([len(seq) for seq in info_dict["chain_sequences"]])),
                    "pocket_labels": torch.tensor(np.concatenate([ca_coord_dfs_bs[chain_id]["pocket"].astype(int).values for chain_id in selected_chains])),
                    "pocket_center": pocket_center,
                    "ligand_coordinates": torch.tensor(ligand_coord_array, dtype=torch.float32),
                    "ligand_indices": torch.zeros(ligand_coord_array.shape[0], dtype=torch.long),
                    "info_dict": info_dict,
                }

                if len(complex_object["residue_coordinates"]) == 0 or \
                    not (len(complex_object["residue_coordinates"]) == len(complex_object["chain_indices"]) == len(complex_object["pocket_labels"])) or \
                    len(complex_object["ligand_coordinates"]) != len(complex_object["ligand_indices"]) or \
                    len(info_dict["chain_ids"]) != len(info_dict["chain_sequences"]):

                    self.handle_error(protein_id, "Inconsistent complex object data.", ligand_name)
                
                else:
                    if not self.multi_ligand:
                        write_json(os.path.join(self.dataset_dir, "info", "info_dicts", f"{sample_name}.json"), info_dict)
                    complex_objects.append(complex_object)

                # break

            if len(complex_objects) == 0:
                self.handle_error(protein_id, "Could not successfully process any ligands.")
                return None
                      
            if self.multi_ligand and len(complex_objects) > 1:

                # Check that the same protein chains were extracted for each ligand
                for i in range(1,len(complex_objects)):
                    assert torch.allclose(complex_objects[0]["residue_coordinates"], complex_objects[i]["residue_coordinates"])
                assert len(set([tuple(complex_object["info_dict"]["chain_ids"]) for complex_object in complex_objects])) == 1, "Inconsistent chain IDs across complex objects."

                # If multiple ligands were successfully processed, combine them into a single complex object
                combined_info_dict = {
                    "protein_id": complex_objects[0]["info_dict"]["protein_id"],
                    "target_name": complex_objects[0]["info_dict"]["target_name"],
                    "ligand_names": [complex_object["info_dict"]["ligand_names"][0] for complex_object in complex_objects],
                    "uniprot_ids": complex_objects[0]["info_dict"]["uniprot_ids"],
                    "chain_ids": complex_objects[0]["info_dict"]["chain_ids"],
                    "chain_sequences": complex_objects[0]["info_dict"]["chain_sequences"],
                    "ligand_smiles": [complex_object["info_dict"]["ligand_smiles"][0] for complex_object in complex_objects],
                    "meta_data": self.meta_data_dict,
                }

                combined_complex_object = {
                    "name": protein_id if self.select_chains == "all" else f"{protein_id}_{'-'.join(selected_chains)}",
                    "residue_coordinates": complex_objects[0]["residue_coordinates"],
                    "pocket_labels": torch.stack([complex_object["pocket_labels"] for complex_object in complex_objects]).bool().any(dim=0).int(),
                    "pocket_center": torch.cat([complex_object["pocket_center"] for complex_object in complex_objects], dim=0),
                    "ligand_coordinates": torch.cat([complex_object["ligand_coordinates"] for complex_object in complex_objects], dim=0),
                    "ligand_indices": torch.repeat_interleave(torch.arange(len(complex_objects)), torch.tensor([complex_object["ligand_coordinates"].shape[0] for complex_object in complex_objects])),
                    "chain_indices": complex_objects[0]["chain_indices"],
                    "info_dict": combined_info_dict,
                }

                if len(combined_complex_object["ligand_coordinates"]) != len(combined_complex_object["ligand_indices"]) or \
                    not(len(combined_complex_object["pocket_center"]) == len(combined_info_dict["ligand_smiles"]) == len(combined_info_dict["ligand_names"])):

                    self.handle_error(protein_id, "Inconsistent combined complex object data.")
                    return None

                else:
                    complex_objects = [combined_complex_object]
                    info_dict = combined_info_dict

            if self.multi_ligand:
                if self.save_cleaned_pdbs:
                    self.save_pdb_files(protein_chains, complex_objects[0]["name"], ligands={ligand_name: ligand for ligand_name, ligand in ligands.items() if ligand_name in complex_objects[0]["info_dict"]["ligand_names"]})
                write_json(os.path.join(self.dataset_dir, "info", "info_dicts", f"{complex_objects[0]['name']}.json"), info_dict)


        else:
            # Sample coordinate DataFrames and sequences for the selected protein chains
            if self.select_chains == "all":
                sample_name = protein_id
            else:                
                sample_name = f"{protein_id}_{'-'.join(selected_chains)}"

            # Validate protein subunit count
            num_large_subunits = len([len(coord_df) for chain_id, coord_df in chain_coord_dfs.items() if len(coord_df) >= self.min_subunit_size])
            if num_large_subunits not in range(1, self.max_num_subunits + 1):
                self.handle_error(protein_id, f"Protein contains {num_large_subunits} large subunits.")
                return None 
            
            # Save cleaned protein in .pdb file
            if self.save_cleaned_pdbs:
                self.save_pdb_files(protein_chains, sample_name, ligands=None)

            # Get Uniprot ID for each protein chain
            if len(protein_id) == 4:
                uniprot_ids = [chain_uniprot_map.get(chain_id, None) for chain_id in selected_chains]
            elif len(protein_id) == 6:
                uniprot_ids = [protein_id]
            elif protein_id.startswith("AF-"):
                uniprot_ids = [protein_id[3:]]
            else:
                uniprot_ids = [None]*len(selected_chains)

            # Save protein information in info_dict
            info_dict = {
                "protein_id": protein_id,
                "target_name": target_name,
                "uniprot_ids": uniprot_ids,
                "chain_ids": selected_chains,
                "chain_sequences": list(chain_sequences.values()),
                "meta_data": self.meta_data_dict
            }

            # Summarize protein, ligand and binding site data in complex object
            complex_object = {
                "name": sample_name,
                "residue_coordinates": torch.tensor(np.concatenate([coord_df[["x", "y", "z"]].to_numpy() for coord_df in ca_coords_dfs.values()])),
                "chain_indices": torch.repeat_interleave(torch.arange(len(info_dict["chain_sequences"])), torch.tensor([len(seq) for seq in info_dict["chain_sequences"]])),
                "info_dict": info_dict,
            }

            if self.load_pocket:
                pocket_center = torch.tensor(self.pocket_dict[protein_id], dtype=torch.float32).unsqueeze(0)
                complex_object["pocket_center"] = pocket_center


            if len(complex_object["residue_coordinates"]) == 0 or \
                len(complex_object["residue_coordinates"]) != len(complex_object["chain_indices"]) or \
                len(info_dict["chain_ids"]) != len(info_dict["chain_sequences"]):

                self.handle_error(protein_id, "Inconsistent complex object data.")
                return None
            
            else:
                complex_objects = [complex_object]
                write_json(os.path.join(self.dataset_dir, "info", "info_dicts", f"{sample_name}.json"), info_dict)

        return complex_objects
    
    
    def load_esm_model(
        self
    ) -> Tuple[torch.nn.Module, Callable]:
        """
        Load a pre-trained ESM-2 model and initialize its batch converter.

        Returns
        -------
        esm_model: torch.nn.Module
            Pre-trained ESM-2 model moved to the configured device.
        esm_batch_converter: callable
            Function to convert protein sequences into model-ready batches.
        """

        # Load 33-layer ESM-2 model with 650M parameters trained on the UR50/D dataset.
        esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        esm_model.to(self.device)
        esm_model.eval()

        # Retrieve the batch converter for preparing protein sequence inputs.
        esm_batch_converter = esm_alphabet.get_batch_converter()

        return esm_model, esm_batch_converter

    
    def calculate_esm_features(
        self,
        sequence: str
    ) -> torch.Tensor:
        """
        Compute ESM (Evolutionary Scale Modeling) embeddings for protein sequences.

        Parameters
        ----------
        sequence: str
            Protein sequence as one-letter codes.

        Returns
        -------
        esm_features: torch.Tensor
            ESM embeddings of shape (sequence_length, embedding_dim) for the input sequence.
        """

        # Replace unknown amino acids ('X') with ESM's unknown token
        esm_sequence = sequence.replace("X", "<unk>")

        # Prepare batch for ESM
        data = [("protein1", esm_sequence)]
        _, _, batch_tokens = self.esm_batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            # Extract final layer representation (layer 33 for esm2_t33_650M_UR50D)
            results = self.esm_model(batch_tokens, repr_layers=[33])

        # Remove special tokens (start/end)
        token_representations = results["representations"][33][0][1:-1]

        # Sanity check: token count must match sequence length
        assert token_representations.shape[0] == len(sequence), f"Token count ({token_representations.shape[0]}) does not match sequence length ({len(sequence)})."

        # Store features on CPU
        esm_features = token_representations.detach().cpu()

        # Free GPU memory for next chain
        del results, batch_tokens

        return esm_features
        
    
    def get_smiles2index(
        self,
        complex_object: Dict[str, object]
    ) -> Dict[str, object]:
        """
        Map SMILES strings for a given target to unique indices and populate the complex object with active, inactive, and labeled ligands.

        Parameters
        ----------
        complex_object: dict[str, object]
            Dictionary representing a protein-ligand complex.  

        Requires
        --------
        self.labeled_smiles, self.dataset_dir

        Returns
        -------
        complex_object: dict[str, object]
            Updated `complex_object` with added keys:
            - "actives" (list[int]): Indices of active ligands.
            - "inactives" (list[int]): Indices of inactive ligands.
            - "labeled_ligands" (list[int]): Indices of ligands with affinity labels.
            - "affinities" (list[float]): Affinity values for labeled ligands.
        """
        actives = []
        inactives = []
        labeled_ligands = []
        affinities = []

        if self.calc_mol_feats:
            # Create smiles2index dictionary if it doesn't exist yet
            if not hasattr(self, "smiles2index_dict"):
                self.smiles2index_dict = {}

            target = complex_object["info_dict"]["target_name"]

            if self.labeled_smiles == "binary":

                # Create dictionaries for caching already processed targets
                if not hasattr(self, "target2actives_dict"):
                    self.target2actives_dict = {}
                    self.target2inactives_dict = {}
                
                # If target already processed, use cached values
                if target in self.target2actives_dict:
                    actives = self.target2actives_dict[target]
                    inactives = self.target2inactives_dict[target]
                
                else:
                    smiles_files_dir = os.path.join(self.dataset_dir, "raw", "smiles_files", target)
                    
                    # Process active ligands
                    if os.path.isfile(os.path.join(smiles_files_dir, "actives.txt")):
                        active_smiles = read_list_from_txt(os.path.join(smiles_files_dir, "actives.txt"))
                        valid = 0
                        for smiles in active_smiles:
                            mol = Chem.MolFromSmiles(smiles)
                            smiles = self.mol2smiles(mol)
                            if smiles is not None and smiles != "":
                                valid += 1
                                if smiles in self.smiles2index_dict:
                                    actives.append(self.smiles2index_dict[smiles])
                                else:
                                    actives.append(len(self.smiles2index_dict))
                                    self.smiles2index_dict[smiles] = len(self.smiles2index_dict)
                        
                    self.target2actives_dict[target] = actives

                    # Process inactive ligands
                    if os.path.isfile(os.path.join(smiles_files_dir, "inactives.txt")):
                        inactive_smiles = read_list_from_txt(os.path.join(smiles_files_dir, "inactives.txt"))
                        valid = 0
                        for smiles in inactive_smiles:
                            mol = Chem.MolFromSmiles(smiles)
                            smiles = self.mol2smiles(mol)
                            if smiles is not None and smiles != "":
                                valid += 1
                                if smiles in self.smiles2index_dict:
                                    inactives.append(self.smiles2index_dict[smiles])
                                else:
                                    inactives.append(len(self.smiles2index_dict))
                                    self.smiles2index_dict[smiles] = len(self.smiles2index_dict)

                    self.target2inactives_dict[target] = inactives

            elif self.labeled_smiles == "affinity":
                    
                if not hasattr(self, "target2actives_dict"):
                    self.target2labeled_dict = {}
                    self.target2affinities_dict = {}

                if target in self.target2labeled_dict:
                    labeled_ligands = self.target2labeled_dict[target]
                    affinities = self.target2affinities_dict[target]

                else:
                    # Process ligands with affinity labels
                    smiles_files_dir = os.path.join(self.dataset_dir, "raw", "smiles_files", target)
                    if os.path.isfile(os.path.join(smiles_files_dir, "smiles_affinities.csv")):
                        smiles_affinity_df = pd.read_csv(os.path.join(smiles_files_dir, "smiles_affinities.csv"))
                        for smiles, aff in zip(smiles_affinity_df["smiles"], smiles_affinity_df["affinity"]):
                            mol = Chem.MolFromSmiles(smiles)
                            smiles = self.mol2smiles(mol)
                            if smiles is not None and smiles != "":
                                if smiles in self.smiles2index_dict:
                                    labeled_ligands.append(self.smiles2index_dict[smiles])
                                else:
                                    labeled_ligands.append(len(self.smiles2index_dict))
                                    self.smiles2index_dict[smiles] = len(self.smiles2index_dict)
                                affinities.append(aff)

                    self.target2labeled_dict[target] = labeled_ligands
                    self.target2affinities_dict[target] = affinities

            else:
                # Simple case: only one ligand SMILES from complex_object, extracted from PDB file
                smiles_list = complex_object["info_dict"]["ligand_smiles"]
                for smiles in smiles_list:
                    if smiles in self.smiles2index_dict:
                        actives.append(self.smiles2index_dict[smiles])
                    else:
                        actives.append(len(self.smiles2index_dict))
                        self.smiles2index_dict[smiles] = len(self.smiles2index_dict)

        # Store results in the complex object

        complex_object["actives"] = actives
        complex_object["inactives"] = inactives
        complex_object["labeled_ligands"] = labeled_ligands
        complex_object["affinities"] = affinities

        return complex_object


    def calculate_features(self,
        complex_objects: List[Dict[str, object]]
    ) -> List[Dict[str, object]]:
        """
        Compute protein and ligand features for a list of complex objects.

        For each complex:
        - Extract protein sequences for all chains, filter by length, and store them in a FASTA list.
        - Compute or retrieve cached ESM features for each chain sequence.
        - Map ligand SMILES strings to unique integer indices using `self.get_smiles2index()`.
        - Save mappings and sequences to disk.

        Parameters
        ----------
        complex_objects: list[dict]
            List of complex objects, i.e. dictionaries representing protein-ligand complexes.

        Requires
        --------
        self.dataset_dir: str
            Base directory for dataset files.
        self.min_subunit_size: int, default=50
            Minimum number of residues for a chain to be considered a protein subunit.

        Returns
        -------
        new_complex_objects: list[dict]
            List of complex objects updated with protein features and ligand indices.
        """

        new_complex_objects = []
        fasta_dict = {}
        esm_feature_dict = {}

        # Iterate through all complex objects
        for complex_object in tqdm(complex_objects):
            
            residue_features = []
            for i, seq in enumerate(complex_object["info_dict"]["chain_sequences"]):

                # Store protein subunit sequences in FASTA dictionary
                if len(seq) > self.min_subunit_size:
                    fasta_dict[f">{complex_object['name']}_{complex_object['info_dict']['chain_ids'][i]}"] = seq

                # Retrieve ESM features from cache or compute new ones
                if seq in esm_feature_dict:
                    residue_features.append(esm_feature_dict[seq])
                else:
                    esm_features = self.calculate_esm_features(seq)
                    esm_feature_dict[seq] = esm_features
                    residue_features.append(esm_features)

            # Concatenate features for all chains in this complex
            residue_features = torch.cat(residue_features)
            complex_object["residue_features"] = torch.FloatTensor(residue_features)

            # Map ligand SMILES to indices and store in complex object
            complex_object = self.get_smiles2index(complex_object)

            new_complex_objects.append(complex_object)
        
        # Save SMILES mapping
        if self.calc_mol_feats:
            self.index2smiles_dict = {v: k for k, v in self.smiles2index_dict.items()}

        # Save FASTA file with protein sequences
        fasta_list = [item for pair in fasta_dict.items() for item in pair]
        write_list_to_txt(os.path.join(self.dataset_dir, "info", "protein_sequences.fasta"), fasta_list)

        return new_complex_objects
    
    
    def get_neighbor_nodes(
        self,
        graph_coords: np.array, 
    ) -> Tuple[List[int], List[int]]:
        """
        Compute neighbor node indices based on distance cutoff and maximum number of neighbors.

        Parameters
        ----------
        coords: np.ndarray
            Array of shape (N, 3) containing node coordinates.

        Requires
        --------
        self.neighbor_dist_cutoff: float, default=10.0
            The maximum distance between two connected nodes to.
        self.max_neighbors: int, default=10
            The maximum number of neighbors each node can have

        Returns
        -------
        src_list: list[int]
            Lists of source node indices.
        dst_list: list[int]
            Lists of destination node indices.
        """

         # Compute full pairwise distance matrix
        pairwise_dists = np.linalg.norm(graph_coords[:, np.newaxis, :] - graph_coords[np.newaxis, :, :], axis=-1)
        np.fill_diagonal(pairwise_dists, np.inf)

        src_list = []
        dst_list = []

        # Apply cutoff and max neighbor constraints
        for i in range(pairwise_dists.shape[0]):
            src = list(np.where(pairwise_dists[i, :] < self.neighbor_dist_cutoff)[0])
            if len(src) > self.max_neighbors:
                src = list(np.argsort(pairwise_dists[i, :]))[:self.max_neighbors]
            dst = [i] * len(src)
            src_list.extend(src)
            dst_list.extend(dst)

        return src_list, dst_list


    def get_graph(self, 
        complex_object: Dict[str, object]
    ) -> HeteroData:
        """
        Build a heterogeneous PyTorch Geometric graph for the given protein-ligand complex.

        Parameters
        ----------
        complex_object: dict[str, object]
            Dictionary containing protein/ligand coordinates, features, and metadata.

        Returns
        -------
        HeteroData
            Graph object saved to disk and returned.
        """

        graph = HeteroData()

        # Protein residue positions and features
        graph["residue"].pos = torch.FloatTensor(np.copy(complex_object["residue_coordinates"]))
        graph["residue"].x = torch.FloatTensor(np.copy(complex_object["residue_features"]))
        graph["residue"].chain_indices = complex_object["chain_indices"].long()

        # Labels and binding site info
        if "pocket_labels" in complex_object:
            graph["residue"].y = torch.FloatTensor(np.copy(complex_object["pocket_labels"]))
        else:
            graph["residue"].y = torch.FloatTensor(np.zeros((complex_object["residue_coordinates"].shape[0],)))

        # Graph-level attributes for virtual node sampling
        graph.mean_feature = complex_object["residue_features"].mean(dim=0).float().unsqueeze(0)
        graph.centroid = torch.FloatTensor(complex_object["residue_coordinates"].mean(axis=0)).unsqueeze(0)
        graph.radius = torch.tensor(np.max(np.linalg.norm(complex_object["residue_coordinates"] - graph.centroid, axis=1)), dtype=torch.float32).unsqueeze(0)

        # Residue-residue edges
        src_list, dst_list = self.get_neighbor_nodes(complex_object["residue_coordinates"])
        graph["residue", "to", "residue"].edge_index = torch.LongTensor([src_list, dst_list])

        # Ligand nodes
        if "ligand_coordinates" in complex_object:
            graph["ligand"].ligand_coordinates = complex_object["ligand_coordinates"].to(dtype=torch.float32)
            graph["ligand"].indices = complex_object["ligand_indices"].long()
            graph["ligand"].num_nodes = len(complex_object["ligand_coordinates"])
        else:
            graph["ligand"].ligand_coordinates = torch.FloatTensor(np.empty((0,3)))
            graph["ligand"].indices = torch.LongTensor(np.empty((0,)))
            graph["ligand"].num_nodes = 0

        # Sanity check
        assert (
            graph["residue"].x.shape[0]
            == graph["residue"].pos.shape[0]
            == graph["residue"].y.shape[0]
        )
       
       # Metadata
        graph.name = complex_object["name"]
        if "pocket_center" in complex_object:
            graph.pocket_center = torch.FloatTensor(np.copy(complex_object["pocket_center"]))
        else:
            graph.pocket_center = torch.FloatTensor(np.empty((0,3)))
        graph.actives = complex_object["actives"]
        graph.inactives = complex_object["inactives"]
        graph.labeled_ligands = complex_object["labeled_ligands"]
        graph.affinities = complex_object["affinities"]

        # Save the graph
        save_path = os.path.join(self.dataset_dir, "processed", "graphs", f"{self.max_neighbors}_neighbors_{self.neighbor_dist_cutoff}_cutoff", f"{graph.name}.pt")
        torch.save(graph, save_path)

        return graph
    

    def process(self):

        complex_info_path = os.path.join(self.dataset_dir, "processed", "complex_info", "complex_info.lmdb")

        # If processed complex info already exists and overwrite is False, load it
        if os.path.exists(complex_info_path) and not self.overwrite:

            print("Loading already processed complex objects.")

            complex_objects = []
            env = lmdb.open(complex_info_path, create=False)
            with env.begin(write=True) as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    complex_object = pickle.loads(value)
                    complex_objects.append(complex_object)
            env.close()

        else:
            if os.path.isfile(complex_info_path):
                shutil.rmtree(complex_info_path)
                if os.path.isfile(os.path.join(self.dataset_dir, "processed", "index2smiles.json")):
                    shutil.rmtree(os.path.join(self.dataset_dir, "processed", "index2smiles.json"))

            # Read protein IDs for processing from the dataset directory
            protein_ids = read_list_from_txt(os.path.join(self.dataset_dir, "info", "protein_ids.txt"))

            # Read known ligand and chain IDs if required
            if self.extract_ligands in ["known", "combined"] or self.select_chains == "chain_id":
                if os.path.exists(os.path.join(self.dataset_dir, "info", "protein_ligand_pairs.csv")):
                    protein_ligand_pairs = pd.read_csv(os.path.join(self.dataset_dir, "info", "protein_ligand_pairs.csv"))
                    if self.extract_ligands in ["known", "combined"]:
                        self.protein_ligand_dict = protein_ligand_pairs.groupby("protein_id")["ligand_id"].apply(list).to_dict()
                    if self.select_chains == "chain_id":
                        self.protein_chain_dict = (
                            protein_ligand_pairs
                            .groupby("protein_id")
                            .apply(lambda g: dict(zip(g["ligand_id"], g["chain"])))
                            .to_dict()
                        )
                else:
                    raise FileNotFoundError("There is no protein_ligand_pairs file, altough self.extract_ligands is 'known' or 'combined, or self.select_chains is 'chain_id'.")
            
            # If load_pocket is True, load the dictionary mapping protein IDs to binding site centers
            if self.load_pocket:
                self.pocket_dict = read_json(os.path.join(self.dataset_dir, "info", "pockets.json"))

            if self.extract_ligands == "from_dict":
                self.ligand_dict = read_json(os.path.join(self.dataset_dir, "info", "ligands.json"))

            # If multi_pdb_targets is True, load the dictionary mapping target names to PDB IDs and reverse it
            if self.multi_pdb_targets:
                self.target2pdb_dict = read_json(os.path.join(self.dataset_dir, "info", "target2pdb.json"))
                self.pdb2target_dict = {}
                for target in self.target2pdb_dict:
                    for pdb_id in self.target2pdb_dict[target]:
                        self.pdb2target_dict[pdb_id] = target


            # Load valid ligand references and ID to SMILES mapping if required
            if self.extract_ligands in ["known", "combined", "all"]:
                self.moad_df, self.invalid_ligands = self.load_valid_ligand_references()
                self.id2smiles_dict = self.load_id2smiles()

            # Create directories for saving processed data
            if self.save_cleaned_pdbs:
                os.makedirs(os.path.join(self.dataset_dir, "processed", "cleaned_pdbs"), exist_ok=True)

            if self.save_complex_info:
                os.makedirs(os.path.join(self.dataset_dir, "processed", "complex_info"), exist_ok=True)

            # Create data directories
            os.makedirs(self.pdb_dir, exist_ok=True)
            os.makedirs(os.path.join(self.dataset_dir, "info", "info_dicts"), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_dir, "processed", "ligand_embeddings"), exist_ok=True)

            # Initialize error file
            with open(os.path.join(self.dataset_dir, "info", "processing_failed.csv"), "a") as f:
                f.write(f"protein_id,ligand_id,comment\n")

            # Initialize PDB parser
            self.parser = PDBParser(QUIET=True)

            # Process each protein ID to extract complex objects
            print("Processing PDB files.")
            if self.num_workers > 1:
                complex_objects = execute_in_parallel(func=self.process_pdb, variable_args=protein_ids, n_jobs=self.num_workers)

            else:
                complex_objects = []
                for protein_id in protein_ids:
                    complex_object = self.process_pdb(protein_id)
                    complex_objects.append(complex_object)
            
            # Flatten list of complex objects
            complex_objects =  [co for co_list in complex_objects if co_list is not None for co in co_list]

            # Calculate ESM features for each complex object
            print("Computing ESM features")
            self.esm_model, self.esm_batch_converter = self.load_esm_model()
            complex_objects = self.calculate_features(complex_objects)
            del self.esm_model, self.esm_batch_converter

            # Save index2smiles dictionary
            if self.calc_mol_feats:
                write_json(os.path.join(self.dataset_dir, "processed", "ligand_embeddings", "index2smiles.json"), self.index2smiles_dict)

            # Save complex objects to LMDB if required
            if self.save_complex_info:
                map_size = 171798691840
                env = lmdb.open(complex_info_path, create=True, map_size=map_size)
                with env.begin(write=True) as txn:
                    for complex_object in complex_objects:
                        serialized_obj = pickle.dumps(complex_object)
                        txn.put(complex_object["name"].encode(), serialized_obj)
                env.close()

        os.makedirs(os.path.join(self.dataset_dir, "processed", "graphs", f"{self.max_neighbors}_neighbors_{self.neighbor_dist_cutoff}_cutoff"), exist_ok=True)

        print("Creating graphs.")
        if self.num_workers > 1:
            graphs = execute_in_parallel(func=self.get_graph, variable_args=complex_objects, n_jobs=self.num_workers)

        else:
            graphs = []
            for complex_object in complex_objects:
                graph = self.get_graph(complex_object)
                graphs.append(graph)

        successfully_processed = []
        for graph in graphs:
            torch.save(graph, os.path.join(self.dataset_dir, "processed", "graphs", f"{self.max_neighbors}_neighbors_{self.neighbor_dist_cutoff}_cutoff", f"{graph.name}.pt"))
            successfully_processed.append(graph.name.split("_")[0])
        successfully_processed = list(set(successfully_processed))
        write_list_to_txt(os.path.join(self.dataset_dir, "info", "processed_protein_ids.txt"), successfully_processed)

        return graphs
    


class LigandProcessor:
    """
    Utility class for generating and preprocessing ligand features
    (ECFP fingerprints and optional molecular descriptors), as well as
    handling feature normalization.

    Parameters
    ----------
    dataset_dir: str
        Directory where the dataset is stored.
    ecfp_radius: int
        Radius used for generating Morgan (ECFP) fingerprints.
    fp_length: int
        Length (dimensionality) of the folded ECFP fingerprint vector.
    calc_descriptors: bool
        Whether to calculate additional molecular descriptors and concatenate them to the fingerprint representation.
    num_workers: int
        Number of worker processes used for parallel ligand processing. If set to 1, processing runs in a single process.
    smiles_batch_size: int, default=1000
        Number of SMILES strings processed per batch during parallel feature computation. 
    scaler_dir: str, default="data/common/scalers"
        Directory where fitted feature scalers are stored or loaded from.
    load_scaler: bool
        If True, attempts to load an existing fitted scaler from `scaler_dir` for feature normalization.
    save_scaler: bool
        If True, saves the fitted scaler to `scaler_dir` after computing normalization statistics.
    """

    def __init__(self, 
        dataset_dir: str,
        ecfp_radius: int = 2,
        fp_length: int = 2048,
        calc_descriptors: bool = True,
        num_workers: int = 64,
        smiles_batch_size: int = 1000,
        scaler_dir: str = "data/common/scalers",
        load_scaler: bool = True,
        save_scaler: bool = False,

    ):
        self.dataset_dir = dataset_dir
        self.ecfp_radius = ecfp_radius
        self.fp_length = fp_length
        self.calc_descriptors = calc_descriptors  

        self.num_workers = num_workers
        self.smiles_batch_size = smiles_batch_size

        self.scaler_dir = scaler_dir
        self.load_scaler = load_scaler
        self.save_scaler = save_scaler

        self.index2smiles_dict = read_json(os.path.join(self.dataset_dir, "processed", "ligand_embeddings", "index2smiles.json"))


    def calculate_ecfp(
        self, 
        mol: Chem.Mol,
    ) -> torch.Tensor:
        """
        Calculate the ECFP (Extended Connectivity Fingerprint) count vector for a molecule using RDKit.

        Parameters
        ----------
        mol : rdkit.Chem.Mol
            RDKit molecule object for which the fingerprint is computed.

        Returns
        -------
        np.ndarray
            1D array of shape (self.ecfp_length,) containing integer ECFP counts.
        """
        
        # # Compute a hashed Morgan fingerprint (ECFP) with counts
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=self.ecfp_radius)
        ecfp_counts = gen.GetCountFingerprint(mol)

        # Initialize a zero vector to store counts
        ecfp = np.zeros((self.fp_length,), dtype=int)
        # Fill in the counts for bits that are nonzero; the modulus operation handles hash collisions
        for idx, count in ecfp_counts.GetNonzeroElements().items():
            ecfp[idx % self.fp_length] += count
            
        # Convert to a single tensor for downstream compatibility
        ecfp = torch.tensor(ecfp, dtype=torch.float32)

        return ecfp
    

    def calculate_descriptors(
        self, 
        mol: Chem.Mol,
    ) -> torch.Tensor:
        """
        Calculate a vector of chemical descriptors for a molecule using RDKit's built-in descriptor functions.

        Parameters
        ----------
        mol: rdkit.Chem.Mol
            RDKit molecule object for which descriptors are computed.

        Returns
        -------
        torch.Tensor
            1D tensor of shape (n_descriptors,) containing descriptor values.
            Values are floats; `inf` is used if a descriptor cannot be computed.
        """
        descriptors = []
        
        # Iterate over all descriptor functions provided by RDKit
        for descr in Descriptors._descList:
            _, descr_calc_fn = descr
            try:
                # Compute descriptor value
                descriptors.append(descr_calc_fn(mol))
            except:
                # If descriptor calculation fails, append 0.0
                descriptors.append(0.0)

        # Convert list to a single tensor for downstream compatibility
        descriptors = torch.tensor(descriptors, dtype=torch.float32)

        return descriptors
    

    def get_ligand_embeddings(
        self,
        smiles_list: List[str]
    ):
        """
        Generate ligand feature embeddings (ECFP, descriptors, or both) from SMILES strings.

        Parameters
        ----------
        smiles_list: list[str]
            List of SMILES strings representing molecules.

        Returns
        -------
        ecfps: torch.Tensor or None
            ECFP embeddings for the molecules.
        descriptors: torch.Tensor or None
            Chemical descriptors for the molecules.
        """

        fingerprints = []
        descriptors = []

        for smiles in smiles_list:
            # Convert SMILES to RDKit molecule
            mol = Chem.MolFromSmiles(smiles)

            # Compute fingerprints
            fingerprint = self.calculate_ecfp(mol)
            fingerprints.append(fingerprint)

            # Compute chemical descriptors if required
            if self.calc_descriptors:
                descriptor = self.calculate_descriptors(mol)
                descriptors.append(descriptor)
        
        # Convert lists to tensors (or None if no features were computed)
        fingerprints = torch.stack(fingerprints, dim=0)
        descriptors = torch.stack(descriptors, dim=0) if len(descriptors) != 0 else None

        return fingerprints, descriptors
    

    def clean_features(
        self, 
        feature_matrix: torch.Tensor
    ) -> torch.Tensor:
        """        
        Clean the feature matrix by replacing NaN, positive infinity, and negative infinity values.

        Parameters
        ----------
        feature_matrix: torch.Tensor
            Matrix of features to clean.

        Returns
        -------
        feature_matrix: torch.Tensor
            Cleaned feature matrix.
        """

        # Replace NaN values with 0.0
        feature_matrix = torch.nan_to_num(feature_matrix, 0.0)

        # Replace very high/low values with highest and lowest column values, respectively
        pos_inf_mask = feature_matrix > 1e+10
        neg_inf_mask = feature_matrix < -1e+10

        feature_matrix[pos_inf_mask] = 0
        feature_matrix[neg_inf_mask] = 0

        col_max = torch.max(feature_matrix, dim=0).values
        col_min = torch.min(feature_matrix, dim=0).values

        feature_matrix[pos_inf_mask] = col_max.expand_as(feature_matrix)[pos_inf_mask]
        feature_matrix[neg_inf_mask] = col_min.expand_as(feature_matrix)[neg_inf_mask]

        return feature_matrix
    
    
    def normalize_features(self, feature_matrix, feature_type):
        """
        Normalize a feature matrix using robust scaling.

        Parameters
        ----------
        feature_matrix : torch.Tensor
            Feature matrix of shape (n_samples, n_features)
        feature_type : str
           Used to determine the filename of the stored scaler, e.g. "ecfp2_2048" or "descriptors"

        Returns
        -------
        feature_matrix : torch.Tensor
            Normalized feature matrix
        """

        if self.load_scaler:
            scaler = joblib.load(f"{self.scaler_dir}/robust_scaler_{feature_type}.pkl")
        else:
            scaler = RobustScaler()
            scaler.fit(feature_matrix)

        feature_matrix = scaler.transform(feature_matrix)

        if self.save_scaler:
            joblib.dump(scaler, f"{self.scaler_dir}/robust_scaler_{feature_type}.pkl")

        feature_matrix = torch.tensor(feature_matrix, dtype=torch.float32)

        return feature_matrix

    
    def process(self):

        """
        Generate, normalize, and store ligand feature embeddings.
        """

        if self.save_scaler:
            os.makedirs(os.path.join(self.scaler_dir), exist_ok=True)
  
        smiles_list = [self.index2smiles_dict[str(i)] for i in range(len(self.index2smiles_dict))]
        
        print("Calculate ligand features")
        if self.num_workers > 1:
            num_batches = int(np.ceil(len(smiles_list)/self.smiles_batch_size))
            smiles_batches = [smiles_list[i*self.smiles_batch_size : (i+1)*self.smiles_batch_size] for i in range(num_batches-1)]
            smiles_batches.append(smiles_list[(num_batches-1)*self.smiles_batch_size :])

            results = execute_in_parallel(func=self.get_ligand_embeddings, variable_args=smiles_batches, n_jobs=self.num_workers)
            fingerprints_batches, descriptors_batches = zip(*results)

            fingerprints = torch.cat([fp for fp in fingerprints_batches if fp is not None], dim=0)
            if self.calc_descriptors:
                descriptors = torch.cat([descriptor for descriptor in descriptors_batches if descriptor is not None], dim=0)

        else:
            fingerprints, descriptors = self.get_ligand_embeddings(smiles_list)

        # Normalize and clean ligand embeddings
        fingerprints = self.clean_features(fingerprints)
        fingerprints = self.normalize_features(fingerprints, f"ecfp{2*self.ecfp_radius}_{self.fp_length}")
        fingerprints = self.clean_features(fingerprints)
        
        print(f"Fingerprints shape: {fingerprints.shape}")
        torch.save(fingerprints, os.path.join(self.dataset_dir, "processed", "ligand_embeddings", f"ecfp{2*self.ecfp_radius}_{self.fp_length}.pt"), pickle_protocol=4)

        fp_fingerprints = np.memmap(os.path.join(self.dataset_dir, "processed", "ligand_embeddings", f"ecfp{2*self.ecfp_radius}_{self.fp_length}.dat"), dtype='float32', mode='w+', shape=fingerprints.shape)
        fp_fingerprints[:] = np.array(fingerprints)[:]
            
        if self.calc_descriptors:
            descriptors = self.clean_features(descriptors)
            descriptors = self.normalize_features(descriptors, "descriptors")
            descriptors = self.clean_features(descriptors)

            print(f"Descriptors shape: {descriptors.shape}")
            torch.save(descriptors, os.path.join(self.dataset_dir, "processed", "ligand_embeddings", "descriptors.pt"), pickle_protocol=4)

            fp_descriptors = np.memmap(os.path.join(self.dataset_dir, "processed", "ligand_embeddings", "descriptors.dat"), dtype='float32', mode='w+', shape=descriptors.shape)
            fp_descriptors[:] = np.array(descriptors)[:]
        
            ligand_metadata = {"num_ligands": descriptors.shape[0], "descriptor_length": descriptors.shape[1], "fingerprint_length": fingerprints.shape[1]}

        else:
            ligand_metadata = {"num_ligands": fingerprints.shape[0], "fingerprint_length": fingerprints.shape[1]}

        write_json(os.path.join(self.dataset_dir, "processed", "ligand_embeddings", f"metadata_ecfp{2*self.ecfp_radius}_{self.fp_length}.json"), ligand_metadata)

