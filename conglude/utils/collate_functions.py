import torch
from torch_geometric.data import Batch


def custom_collate(batch):
    """
    Custom collate function for DataLoader, which handles batching of protein graphs together with optional
    ligand features and labels.

    Parameters
    ----------
    batch : list of tuples
        Each element is a tuple consisting of:
        - proteins: PyG Data object(s) representing protein graphs
        - ligands: Tensor of ligand features per molecule (or None)
        - labels: Tensor of binary labels (or None)
        - ligand_idx: Tensor of ligand's indices in the dataset (to enable mapping back to SMILES) (or None)

    Returns
    -------
    proteins_batch: torch_geometric.data.Batch
        Batched protein graph(s).
    ligands_batch: torch.Tensor or None
        Concatenated ligand features.
    labels_batch: torch.Tensor or None
        Concatenated labels.
    ligand_batch_idx_batch: torch.Tensor or None
        Batch index for each ligand atom (maps to protein index in batch).
    ligand_idx_batch: torch.Tensor or None
        Concatenated ligand index tensor.
    """

    # Unzip batch elements
    proteins, ligands, labels, ligand_idx = zip(*batch)

    # Multi-PDB targets (dataset items contain more than one protein)
    if type(proteins[0]) == list:
        assert len(proteins) == 1, "For multi-PDB targets, only a batch size of 1 ist supported."
        proteins_batch = Batch.from_data_list(proteins[0])

        # Each PDB entity should have the same ligands, take from the first one
        ligands_batch = ligands[0]
        labels_batch = labels[0]
        ligand_idx_batch = ligand_idx[0]

        # All ligands belong to protein index 0
        ligand_batch_idx_batch = torch.full((ligand_idx_batch.shape[0],), fill_value=0)

    # Standard batching (one protein per dataset item)
    else:
        proteins_batch = Batch.from_data_list(proteins)

        if ligands[0] is not None:
            # Concatenate ligand features, labels and ligand indices across batch
            ligands_batch = torch.cat(ligands, dim=0)
            labels_batch = torch.cat(labels, dim=0)
            ligand_idx_batch = torch.cat(ligand_idx)
            
            # Create batch index mapping each ligand atom to the corresponding protein in the batch
            ligand_batch_idx = [torch.full((mol.shape[0],), fill_value=i) for i, mol in enumerate(ligands)]
            ligand_batch_idx_batch = torch.cat(ligand_batch_idx, dim=0)

        else:
            # Handle case where no ligands are provided
            ligands_batch = None
            labels_batch = None
            ligand_idx_batch = None
            ligand_batch_idx_batch = None


    return proteins_batch, ligands_batch, labels_batch, ligand_batch_idx_batch, ligand_idx_batch



def custom_collate_protein(batch):
    """
    Custom collate function for DataLoader, which handles batching of protein graphs only.
    
    Parameters
    ----------
    batch : list of tuples
        Each element is a tuple consisting of:
        - proteins: PyG Data object(s) representing protein graphs
        - ligands: Tensor of ligand features per molecule (or None)
        - labels: Tensor of binary labels (or None)
        - ligand_idx: Tensor of ligand's indices in the dataset (to enable mapping back to SMILES) (or None)

    Returns
    -------
    proteins_batch: torch_geometric.data.Batch
        Batched protein graph(s).
    """

    # Unzip batch elements
    proteins, _, _, _ = zip(*batch)

    # Only return protein graphs
    proteins_batch = Batch.from_data_list(proteins)

    return proteins_batch