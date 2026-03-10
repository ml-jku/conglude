from torch_geometric.data import Dataset, HeteroData
from torch.utils.data import IterableDataset, DataLoader
import pytorch_lightning as pl
import os
import numpy as np
import torch
import random
from collections import Counter
from torch_geometric.data import Batch
from typing import List, Tuple, Union, Optional, Callable, Iterator, Any

from conglude.utils.common import read_list_from_txt, read_json
from conglude.utils.graph import sample_fibonacci_grid, sample_uniform_in_sphere
from conglude.utils.collate_functions import custom_collate
from conglude.utils.data_processing import PDBGraphProcessor, LigandProcessor



class ConGLUDeDataset(Dataset):
    """
    Custom PyTorch Dataset for protein–ligand learning tasks. 
    This dataset handles protein graph and ligand feature processing and data sampling.
    Per sample it returns a protein graph, ligand features, binary labels and ligand indices.
    
    Parameters
    ----------
    dataset_dir: str
        Root directory of the dataset containing processed graphs, ligand embeddings, and metadata.
    dataset_name: str
        Name of the dataset.
    task: str
        Task identifier ("vs" for virtual_screening, "tf" for target fishing, "pp" for pocket prediction,
        "pr" for pocket ranking, "all" for all of the above, and "train"/"val" for training and validation).
    structure_based: bool
        Whether the dataset is structure-based (has annotated binding sites).
    split: str
        Dataset split ("train", "val", or "test").
    fingerprint_type: str
        Type of ligand fingerprint (e.g., "ecfp4_2048"). If None, fingerprints are not used.
    load_descriptors: bool
        If True, load molecular descriptors in addition to fingerprints.
    memmap: bool
        If True, load ligand features using NumPy memory mapping to reduce RAM usage.
    max_num_actives: int or None
        Maximum number of active ligands sampled per protein. If None, all available ligands are used.
    inactive_active_ratio: int or None
        Ratio of inactives to actives during sampling. If None, all inactives are used.
    num_pocket_nodes: int
        Number of sampled pocket nodes per protein graph.
    protein_node: bool
        If True, add a global protein node connected to all residues.
    sampling_strategy: str
        Strategy for sampling pocket nodes ("fibonacci" or "uniform").
    random_rotations: bool
        If True, apply random rotations to sampled pocket grids.
    neighbor_dist_cutoff: float
        Distance cutoff (in Å) for constructing residue neighbor edges.
    max_neighbors: int
        Maximum number of neighbors per residue node.
    batch_size: int
        Batch size.
    debug: bool
        If True, reduce dataset size for debugging.
    pdb_dir: str
        Directory containing raw PDB files.
    overwrite: bool
        If True, force regeneration of processed protein graphs.
    num_workers: int
        Number of worker processes used during preprocessing.
    extract_ligands: str
        Strategy for ligand extraction from PDB files.
    select_chains: str
        Which protein chains to include ("all" or specific chain IDs).
    labeled_smiles: str
        Whether the ligands are provided as actives and inactives in separate files ("binary") or extracted from PDB files ("none").
    multi_ligand: bool
        If True, supports multiple ligands per PDB entry.
    multi_pdb_targets: bool
        If True, one target can have multiple PDB structures and predictions are averaged across those.
    calc_mol_feats: bool
        If True, compute ligand fingerprints and descriptors if missing.
    max_num_subunits: int
        Maximum allowed number of protein subunits during preprocessing.
    load_pocket: bool
        If True, load precomputed pocket centers instead of extracting them from the PDB file.
    save_cleaned_pdbs: bool
        If True, save cleaned PDB structures during preprocessing.
    save_complex_info: bool
        If True, save metadata about protein–ligand complexes during preprocessing.
    ecfp_radius: int
        Radius used for ECFP fingerprint computation (ECFP4 = 2).
    fp_length: int
        Length of the fingerprint bit vector.
    load_scaler: bool
        If True, load scaler for ligand feature normalization.
    save_scaler: bool
        If True, save fitted scaler after ligand feature normalization.
    """

    def __init__(
        self,
        dataset_dir: str,
        dataset_name: str,

        task: str = "all",
        structure_based: bool = False,
        split: str = "test",
        
        fingerprint_type: str = "ecfp4_2048",
        load_descriptors: bool = True,
        memmap: bool = True,
        max_num_actives: int = None,
        inactive_active_ratio: int = None,
        
        num_pocket_nodes: int = 8,
        protein_node: bool = True,
        sampling_strategy: str = "fibonacci",
        random_rotations: bool = True,
        neighbor_dist_cutoff: float = 10.0,
        max_neighbors: int = 10,

        batch_size: int = 16,        
        debug: bool = False,

        pdb_dir: str = "./data/pdb_files",
        overwrite: bool = False,
        num_workers: int = 64,
        extract_ligands: str = "none",
        select_chains: str = "all",
        labeled_smiles: str = "binary",
        multi_ligand: bool = False,
        multi_pdb_targets: bool = False,
        calc_mol_feats: bool = True,
        max_num_subunits: int = 24,
        load_pocket: bool = False,
        save_cleaned_pdbs: bool = False,
        save_complex_info: bool = False,

        ecfp_radius: int = 2,
        fp_length: int = 2048,
        load_scaler: bool = True,
        save_scaler: bool = False,
    ) -> None:
        
        # Prefix used for train/val splits
        if split in ["train", "val"]:
            self.prefix = f"{task}_{split}_"
        else:
            self.prefix = ""
        
        print(f"Initializing dataset {self.prefix}{dataset_name}.")

        super().__init__()
        
        # Basic dataset configuration
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.task = task
        self.structure_based = structure_based

        # Protein graph configuration
        self.multi_pdb_targets = multi_pdb_targets
        self.num_pocket_nodes = num_pocket_nodes
        self.protein_node = protein_node
        self.sampling_strategy = sampling_strategy
        self.random_rotations = random_rotations
        self.neighbor_dist_cutoff = neighbor_dist_cutoff
        self.max_neighbors = max_neighbors

        # Ligand feature configuration
        self.fingerprint_type = fingerprint_type
        self.load_descriptors = load_descriptors
        self.memmap = memmap
        self.max_num_actives = max_num_actives
        self.inactive_active_ratio = inactive_active_ratio

        self.batch_size = batch_size
        self.debug = debug

        # Set data paths
        self.processed_ids_path, self.task_ids_path, self.excluded_ids_path, self.graph_dir, self.ligand_metadata_path, self.fingerprint_path, self.descriptor_path = self.processed_file_names
        
        # Process proteins if missing
        if not os.path.isfile(self.processed_ids_path) or not os.path.isfile(self.task_ids_path) or not os.path.isdir(self.graph_dir):
            graph_processor = PDBGraphProcessor(
                    dataset_dir=dataset_dir,
                    pdb_dir=pdb_dir,
                    overwrite=overwrite,
                    num_workers=num_workers,
                    extract_ligands=extract_ligands,
                    select_chains=select_chains,
                    labeled_smiles=labeled_smiles,
                    multi_ligand=multi_ligand,
                    multi_pdb_targets=multi_pdb_targets,
                    neighbor_dist_cutoff=neighbor_dist_cutoff,
                    max_neighbors=max_neighbors,
                    calc_mol_feats=calc_mol_feats,
                    max_num_subunits=max_num_subunits,
                    load_pocket=load_pocket,
                    save_cleaned_pdbs=save_cleaned_pdbs,
                    save_complex_info=save_complex_info
                    )
            graph_processor.process()

        # Load protein graphs
        self.graph_files = self.get_graph_files()

        # Process ligands if missing
        if (fingerprint_type is not None or load_descriptors) and calc_mol_feats:
            if not os.path.isfile(self.fingerprint_path) or not os.path.isfile(self.descriptor_path) or (not os.path.isfile(self.ligand_metadata_path) and self.memmap):
                
                ligand_processor = LigandProcessor(
                    dataset_dir=dataset_dir,
                    ecfp_radius=ecfp_radius,
                    fp_length=fp_length,
                    calc_descriptors=load_descriptors,
                    num_workers=num_workers,
                    load_scaler=load_scaler,
                    save_scaler=save_scaler
                )
                
                ligand_processor.process()

            # Load ligands
            self.ligand_features = self.load_ligand_data()

        # Get counter for number of pockets per PDB entry
        self.pocket_counter = Counter([file_name.split("_")[0] for file_name in self.graph_files])

        if self.multi_pdb_targets:
            self.target2pdb_dict = read_json(os.path.join(self.dataset_dir, "info", "target2pdb.json"))
            self.targets = list(self.target2pdb_dict.keys())

    
    @property
    def processed_file_names(
        self
    ) -> Tuple[str]:
        """
        Return all file and directory paths required for a fully processed dataset.

        Returns
        -------
        Tuple[str]
            processed_ids_path: str
                Contains a list of protein IDs that have already been successfully processed into graph files.
            task_ids_path: str
                Contains the protein IDs assigned to the current task/split.
            excluded_ids_path: str
                Contains protein IDs that were excluded during processing.
            graph_dir: str
                Directory containing processed protein graph files (.pt).
            ligand_metadata_path: str
                Path to ligand metadata JSON file. Stores information required for loading memory-mapped ligand embeddings (number of ligands, fingerprint  and descriptor dimensionality).
            fingerprint_path: str
                Path to stored molecular fingerprints.
            descriptor_path : str
                Path to stored molecular descriptors.
        """

        processed_ids_path = os.path.join(self.dataset_dir, "info", "processed_protein_ids.txt")
        task_ids_path = os.path.join(self.dataset_dir, "info", f"{self.prefix}protein_ids.txt")
        excluded_ids_path = os.path.join(self.dataset_dir, "info", "excluded_protein_ids.txt")

        graph_dir = os.path.join(self.dataset_dir, "processed", "graphs", f"{self.max_neighbors}_neighbors_{self.neighbor_dist_cutoff}_cutoff")

        ligand_metadata_path = os.path.join(self.dataset_dir, "processed", "ligand_embeddings", f"metadata_{self.fingerprint_type}.json")

        # Switch between memmap and torch storage
        if self.memmap:
            fingerprint_path = os.path.join(self.dataset_dir, "processed", "ligand_embeddings", f"{self.fingerprint_type}.dat")
            descriptor_path  = os.path.join(self.dataset_dir, "processed", "ligand_embeddings", f"descriptors.dat")
        else:
            fingerprint_path = os.path.join(self.dataset_dir, "processed", "ligand_embeddings", f"{self.fingerprint_type}.pt")
            descriptor_path = os.path.join(self.dataset_dir, "processed", "ligand_embeddings", f"descriptors.pt")

        return processed_ids_path, task_ids_path, excluded_ids_path, graph_dir, ligand_metadata_path, fingerprint_path, descriptor_path


    def get_graph_files(
        self
    ) -> List[str]:
        """
        Retrieve all valid processed protein graph files for the current dataset split.

        Returns
        -------
        List[str]
            A list of filenames (not full paths) corresponding to valid processed protein graph `.pt` files located in `self.graph_dir`.
        """

        # Load list of proteins that were successfully processed
        processed_ids = read_list_from_txt(self.processed_ids_path)

        # Load list of proteins assigned to the current task/split
        task_ids = read_list_from_txt(self.task_ids_path)

        # Load list of excluded proteins (if the file exists)
        excluded_ids = read_list_from_txt(self.excluded_ids_path) if os.path.isfile(self.excluded_ids_path) else []
        
        # Select graph files whose filename prefix (protein ID) matches the filtered valid protein IDs
        protein_ids = list((set(processed_ids) & set(task_ids)) - set(excluded_ids))
        graph_files = [f for f in os.listdir(self.graph_dir) if f.endswith('.pt') and f[:-3].split('_')[0] in protein_ids]

        # In debug mode (training only), limit dataset size
        if self.debug and self.split == "train":
            graph_files = graph_files[:10]

        return graph_files


    def load_ligand_data(
        self
    ) -> Union[torch.Tensor, np.memmap]:
        """
        Load precomputed ligand feature representations (molecular fingerprints and/or descriptors)
        either as PyTorch tensors or as NumPy memory-mapped arrays.

        Returns
        -------
        torch.Tensor or np.memmap
            A tensor or memory-mapped array of shape (num_ligands, feature_dim).
        """

        if self.memmap:
            ligand_metadata = read_json(self.ligand_metadata_path)

        # Load fingerprints
        if self.fingerprint_type is not None:
            if self.memmap:
                fingerprints = np.memmap(self.fingerprint_path, dtype="float32", mode="r", shape=(ligand_metadata["num_ligands"], ligand_metadata["fingerprint_length"]))
            else:
                fingerprints = torch.load(self.fingerprint_path).to(dtype=torch.float32)

        # Load descriptors
        if self.load_descriptors:
            if self.memmap:
                descriptors = np.memmap(self.descriptor_path, dtype="float32", mode="r", shape=(ligand_metadata["num_ligands"], ligand_metadata["descriptor_length"]))
            else:
                descriptors = torch.load(self.descriptor_path).to(dtype=torch.float32)

        # Concatenate fingerprints and descriptors if both exist
        if self.fingerprint_type is not None:
            if self.load_descriptors:
                if self.memmap:
                    ligand_features = np.concatenate([fingerprints, descriptors], axis=1)
                else:
                    ligand_features = torch.cat([fingerprints, descriptors], dim=1)
            else:
                ligand_features = fingerprints
        else:
            ligand_features = descriptors

        return ligand_features


    def load_graph(
        self, 
        graph_file: str, 
    ) -> HeteroData:
        """
        Load a processed protein graph and construct pocket/protein nodes.

        Parameters
        ----------
        graph_file: str
            Path to the serialized PyTorch Geometric graph (.pt file).

        Returns
        -------
        graph: torch_geometric.data.HeteroData
            A heterogenous graph containing residue nodes, pocket nodes, and optionally a protein node.
        """

        # Load serialized graph object
        graph = torch.load(graph_file, weights_only=False)

        # Initialize pocket node positions
        if self.sampling_strategy == "fibonacci":
            graph["pocket"].pos = sample_fibonacci_grid(graph.centroid, graph.radius, self.num_pocket_nodes, random_rotations=self.random_rotations)
        else:
            graph["pocket"].pos = sample_uniform_in_sphere(graph.centroid, graph.radius, self.num_pocket_nodes)

        # Initialize pocket node features
        graph["pocket"].x = graph.mean_feature.repeat(self.num_pocket_nodes, 1)

        # Edges between pocket nodes and all residue nodes
        num_res = graph["residue"].num_nodes
        src_atom = torch.arange(num_res).repeat(self.num_pocket_nodes)
        dst_pocket_node = torch.arange(self.num_pocket_nodes).repeat_interleave(num_res)

        edge_index = torch.stack([src_atom, dst_pocket_node], dim=0)

        graph["residue", "to", "pocket"].edge_index = edge_index
        graph["pocket", "to", "residue"].edge_index = edge_index.flip(0)

        # Optionally add a global protein node
        if self.protein_node:
            graph["protein"].x = graph.mean_feature

            src_atom = torch.arange(num_res)
            dst_protein_node = torch.arange(1).repeat_interleave(num_res)

            edge_index = torch.stack([src_atom, dst_protein_node], dim=0)

            graph["residue", "to", "protein"].edge_index = edge_index
            graph["protein", "to", "residue"].edge_index = edge_index.flip(0)

        return graph
    

    def get(
        self, 
        idx: int
    ) -> Tuple[object, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Retrieve a dataset sample.

        Parameters
        ----------
        idx: int
            Index of the graph file to load.

        Returns
        -------
        protein: object
            Loaded protein graph object with virtual nodes (PyG HeteroData object).
        ligands: Optional[torch.Tensor]
            Tensor of shape (num_ligands, feature_dim) containing ligand features.
        labels: Optional[torch.Tensor]
            Tensor of shape (num_ligands,) with binary labels: 1 for active ligands, 0 for inactive ligands.
        ligand_idx : Optional[torch.Tensor]
            Tensor of shape (num_ligands,) containing ligand indices.
        """

        # Handle multi-target PDB setting
        if self.multi_pdb_targets:
            return self.get_multi_pdb_targets(idx)

        # Load protein graph
        graph_file = os.path.join(self.graph_dir, self.graph_files[idx])
        protein = self.load_graph(graph_file)

        # If ligand features are enabled
        if not (self.fingerprint_type is None and self.load_descriptors is False):
            active_inds = protein["actives"]
            inactive_inds = protein["inactives"]

            # Ensure at least one inactive if both are empty
            if len(active_inds) + len(inactive_inds) == 0:
                inactive_inds = [0]

            # No inactive/active ratio constraint
            if self.inactive_active_ratio is None:
                ligand_idx = np.array(active_inds + inactive_inds)
                labels = torch.cat([torch.ones(len(active_inds)), torch.zeros(len(inactive_inds))]).long()

                # Optional subsampling of ligands
                if self.max_num_actives is not None and len(ligand_idx) > self.max_num_actives:
                    selected = random.sample(range(len(ligand_idx)), self.max_num_actives)
                    ligand_idx = ligand_idx[selected]
                    labels = labels[selected]

            # Enforce inactive:active sampling ratio
            else:
                n_actives = min(len(active_inds), self.max_num_actives)
                n_inactives = min(n_actives*self.inactive_active_ratio, len(inactive_inds))

                sampled_active_inds = random.sample(active_inds, n_actives)
                sampled_inactive_inds = random.sample(inactive_inds, n_inactives)

                ligand_idx = np.array(sampled_active_inds + sampled_inactive_inds)
                labels = torch.cat([torch.ones(len(sampled_active_inds)), torch.zeros(len(sampled_inactive_inds))]).lon()

            # Load ligand features
            if self.memmap:
                ligands = torch.from_numpy(self.ligand_features[ligand_idx]).to(dtype=torch.float32)
            else:
                ligands = self.ligand_features[ligand_idx]

            ligand_idx = torch.tensor(ligand_idx, dtype=torch.long)

        # If ligand features are disabled
        else:
            ligand_idx = None
            ligands = None
            labels = None

        return protein, ligands, labels, ligand_idx
    

    def get_multi_pdb_targets(
        self, 
        idx: int
    ) -> Tuple[List[object], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load multiple protein graphs corresponding to the same biological target and retrieve the associated ligand features and labels.

        Parameters
        ----------
        idx: int
            Index of the target in `self.targets`.

        Returns
        -------
        proteins: List[object]
            List of loaded protein graph objects (PyG HeteroData objects), one per PDB structure associated with the target.
        ligands: torch.Tensor
            Tensor of shape (num_ligands, feature_dim) containing ligand features.
        labels: torch.Tensor
            Tensor of shape (num_ligands,) containing binary labels: 1 for active ligands and 0 for inactive ligands.
        ligand_idx: torch.Tensor
            Tensor of shape (num_ligands,) containing ligand indices used to retrieve ligand features.
        """

        # Retrieve target identifier
        target = self.targets[idx]

        # Load all protein graphs associated with the target
        graph_files = [f for f in self.graph_files if f.split(".")[0].split("_")[0] in self.target2pdb_dict[target]]

        proteins = []
        for graph_file in graph_files:
            protein = self.load_graph(os.path.join(self.graph_dir, graph_file))
            proteins.append(protein)

        # Active/inactive indices are assumed to be identical across PDBs for the same target -> take them from the last loaded protein.
        active_inds = protein["actives"]
        inactive_inds = protein["inactives"]

        ligand_idx = np.array(active_inds + inactive_inds)

        # Load ligand features
        if self.memmap:
            ligands = torch.from_numpy(self.ligand_features[ligand_idx]).to(dtype=torch.float32)
        else:
            ligands = self.ligand_features[ligand_idx]

        # Convert ligand indices to tensor
        ligand_idx = torch.tensor(ligand_idx, dtype=torch.long)

        # Create binary labels (1 = active, 0 = inactive)
        labels = torch.cat([torch.ones(len(active_inds)), torch.zeros(len(inactive_inds))]).long()

        return proteins, ligands, labels, ligand_idx

        
    def len(
        self
    ) -> int:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            - If `self.multi_pdb_targets` is True:
                Returns the number of unique biological targets (i.e., length of `self.targets`).
            - If `self.multi_pdb_targets` is False:
                Returns the number of individual protein graph files (i.e., length of `self.graph_files`).
        """
        
        if self.multi_pdb_targets:
            return len(self.targets)
        else:
            return len(self.graph_files)
        


class MixedDataset(IterableDataset):
    """
    Iterable dataset that mixes ligand-based (LB) and structure-based (SB)
    batches during training.

    At each iteration step, a batch is sampled either from the LB loader
    or the SB loader with probability `p_LB`. Once one loader is exhausted,
    the remaining batches are drawn from the other.

    Parameters
    ----------
    LB_dataset: ConGLUDeDataset
        Ligand-based dataset instance.
    SB_dataset: ConGLUDeDataset
        Structure-based dataset instance.
    LB_batch_size: int
        Batch size used for the ligand-based DataLoader.
    SB_batch_size: int
        Batch size used for the structure-based DataLoader.
    LB_collate_fn: Callable
        Custom collate function for the ligand-based DataLoader.
    SB_collate_fn: Callable
        Custom collate function for the structure-based DataLoader.
    p_LB: float
        Probability of sampling a ligand-based (LB) batch when both loaders still have data available.
    num_workers: int
        Number of worker processes used by each DataLoader.
    shuffle: bool
        Whether to shuffle each dataset inside its DataLoader.
    """

    def __init__(
        self,
        LB_dataset: ConGLUDeDataset,
        SB_dataset: ConGLUDeDataset,
        LB_batch_size: int = 16,
        SB_batch_size: int = 64,
        LB_collate_fn: Optional[Callable] = None,
        SB_collate_fn: Optional[Callable] = None,
        p_LB: float = 0.5,
        num_workers: int = 0,
        shuffle: bool = True
    ) -> None:
        
        super().__init__()

        self.LB_batch_size = LB_batch_size
        self.SB_batch_size = SB_batch_size
        self.p_LB = p_LB

        # Create internal DataLoaders for each dataset
        self.LB_loader = DataLoader(
            LB_dataset,
            batch_size=LB_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=LB_collate_fn
        )
        self.SB_loader = DataLoader(
            SB_dataset,
            batch_size=SB_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=SB_collate_fn
        )

        # Combine pocket counters of both datasets
        self.pocket_counter = LB_dataset.pocket_counter + SB_dataset.pocket_counter


    def __iter__(
        self
    ) -> Iterator[Tuple[Batch, bool]]:
        """
        Iterate over mixed LB and SB batches.

        Yields
        ------
            batch: Batch
                The batch returned by the corresponding DataLoader
            is_structure_based: bool
                True if strucute-based batch, False if ligand-based
        """
                
        iter_LB = iter(self.LB_loader)
        iter_SB = iter(self.SB_loader)

        LB_done, SB_done = False, False

        # Continue until both loaders are exhausted
        while not (LB_done and SB_done):
            # If both loaders still have data, randomly choose
            if not LB_done and not SB_done:
                if random.random() < self.p_LB:
                    try:
                        yield next(iter_LB), False
                    except StopIteration:
                        LB_done = True
                else:
                    try:
                        yield next(iter_SB), True
                    except StopIteration:
                        SB_done = True

            # If only LB remains
            elif not LB_done:
                try:
                    yield next(iter_LB), False
                except StopIteration:
                    LB_done = True
            
            # If only SB remains
            elif not SB_done:
                try:
                    yield next(iter_SB), True
                except StopIteration:
                    SB_done = True


    def __len__(self):
        """
        Return the total number of batches across both loaders.

        Returns
        -------
        int
            Total number of batches: len(LB_loader) + len(SB_loader)
        """
        return len(self.LB_loader) + len(self.SB_loader)
    


class DatasetList:
    """
    Simple container class for Hydra to collect multiple datasets into a list.
    
    Parameters
    ----------
    **datasets : Any
        Arbitrary keyword arguments representing datasets passed by Hydra.
    """

    def __init__(
        self, 
        **datasets: Any
    ) -> None:

        # Convert the passed datasets into a list, ignoring the names.
        self.datasets = list(datasets.values())


    def __iter__(
        self
    ) -> Iterator[Any]:
        """
        Return an iterator over the stored datasets.

        Returns
        -------
        Iterator[Any]
            Iterator over the dataset objects.
        """

        return iter(self.datasets)


    def __len__(
        self
    ) -> int:
        """
        Return the number of datasets stored in the container.

        Returns
        -------
        int
            Number of datasets.
        """

        return len(self.datasets)
    

    def __getitem__(
        self, 
        index
    ) -> Any:
        """
        Retrieve a dataset by index.

        Parameters
        ----------
        index : int
            Position of the dataset to retrieve.

        Returns
        -------
        Any
            The dataset at the specified index.
        """

        return self.datasets[index]



class PLDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule supporting ligand-Based datasets, structure-Based datasets  and mixed training via `MixedDataset`.
    
    Parameters
    ----------
    train_datasets: list, optional
        List of training datasets (0, 1 or 2 ("LB_train" and "SB_train"))
    val_datasets: list, optional
        List of validation datasets.
    test_datasets: list, optional
        List of test datasets.
    num_workers : int
        Number of DataLoader worker processes.
    """

    def __init__(
        self,
        train_datasets: list = None,
        val_datasets: list = None,
        test_datasets: list = None,
        num_workers: int = 16,
    ) -> None:
        
        super().__init__()

        self.train_datasets = train_datasets or []
        self.val_datasets = val_datasets or []
        self.test_datasets = test_datasets or []

        self.num_workers = num_workers

 
    def train_dataloader(
        self
        ) -> Optional[DataLoader]:
        """
        Create the training DataLoader.

        Returns
        -------
        Optional[DataLoader]
            None if no training datasets are provided, standard DataLoader if a single dataset is used, a DataLoader wrapping a `MixedDataset` if two datasets ("SB_train" and "LB_train") are provided.
        """

        # No training datasets
        if len(self.train_datasets) == 0:
            return None

        # Single dataset training
        elif len(self.train_datasets) == 1:
            return DataLoader(self.train_datasets[0], batch_size=self.train_datasets[0].batch_size, collate_fn=custom_collate, shuffle=True, num_workers=self.num_workers)                         

        # Mixed training (SB + LB)
        elif len(self.train_datasets) == 2:

            # Ensure correct dataset naming
            assert set([dataset.name for dataset in self.train_datasets]) == {"SB_train", "LB_train"}, "For mixed training, datasets have to be names 'SB_train' and 'LB_train'"
            
            # Separate SB and LB datasets
            for train_dataset in self.train_datasets:
                if train_dataset.name == "SB_train":
                    train_SB_dataset = train_dataset
                    SB_batch_size = train_dataset.batch_size
                elif train_dataset.name == "LB_train":
                    train_LB_dataset = train_dataset
                    LB_batch_size = train_dataset.batch_size

            # Create mixed dataset (already yields full batches)
            p_LB = np.ceil(len(train_LB_dataset)/LB_batch_size)/(np.ceil(len(train_LB_dataset)/LB_batch_size) + np.ceil(len(train_SB_dataset)/SB_batch_size))
            mixed_dataset = MixedDataset(train_LB_dataset, train_SB_dataset, LB_batch_size, SB_batch_size, LB_collate_fn=custom_collate, SB_collate_fn=custom_collate, p_LB=p_LB, num_workers=self.num_workers, shuffle=True)

            return DataLoader(mixed_dataset, batch_size=None)  # batch_size=None because MixedDataset already returns batches
        
        else:
            raise Exception("More than 2 dataloaders.")
                                     

    def val_dataloader(
        self
        ) -> List[DataLoader]:
        """
        Create validation DataLoaders.

        Returns
        -------
        List[DataLoader]
            A list of DataLoaders, one per validation dataset.
        """
            
        val_dataloaders = []

        for val_dataset in self.val_datasets:

            val_dataloaders.append(DataLoader(val_dataset, batch_size=val_dataset.batch_size, collate_fn=custom_collate, shuffle=False, num_workers=self.num_workers))

        return val_dataloaders


    def test_dataloader(
        self
        ) -> List[DataLoader]:
        """
        Create test DataLoaders.

        Returns
        -------
        List[DataLoader]
            A list of DataLoaders, one per test dataset.
        """

        test_dataloaders = []

        for test_dataset in self.test_datasets:

            test_dataloaders.append(DataLoader(test_dataset, batch_size=test_dataset.batch_size, collate_fn=custom_collate, shuffle=False, num_workers=self.num_workers))

        return test_dataloaders
     
    