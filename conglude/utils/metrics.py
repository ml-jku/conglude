import torch
import numpy as np

from torchmetrics import Metric, JaccardIndex
from rdkit.ML.Scoring.Scoring import CalcBEDROC
from torchmetrics.functional import average_precision, auroc

from typing import Dict, List, Tuple
from torch import Tensor



def enrichment_factor(
    y_true: Tensor, 
    y_score: Tensor, 
    fraction: float
) -> float:
    """
    Compute the enrichment factor (EF) at a given top fraction.

    Args:
        y_true (Tensor): Binary labels (0/1), shape (N,)
        y_score (Tensor): Predicted scores, shape (N,)
        fraction (float): Top fraction to evaluate (e.g., 0.01 for top 1%)

    Returns:
        ef (float): Enrichment factor for the given fraction.
    """
    y_true = y_true.flatten()
    y_score = y_score.flatten()

    n = y_true.numel()
    n_pos = y_true.sum().item()

    if n_pos == 0:
        return float("nan")  # cannot compute EF if no positives

    # Number of items in the top fraction
    k = max(1, int(n * fraction))

    # Sort predictions descending
    _, indices = torch.sort(y_score, descending=True)
    top_k_true = y_true[indices[:k]]

    # EF formula
    ef = (top_k_true.sum().item() / k) / (n_pos / n)

    return ef



class VirtualScreeningMetrics(Metric):
    """
    Metrics for evaluating virtual screening performance.

    Metrics supported:
    - AUC
    - BEDROC
    - Enrichment Factor (EF) at multiple fractions

    Metrics are accumulated per unique target with `update()` and averaged in `compute()`.

    Args:
        calc_auc (bool): Whether to compute the Area Under the ROC Curve (AUC). Default: True.
        calc_bedroc (bool): Whether to compute BEDROC, emphasizing early recognition. Default: True.
        calc_ef (bool): Whether to compute Enrichment Factor at specified fractions. Default: True.
        ef_fractions (List[float]): List of top fractions (e.g., [0.005, 0.01, 0.05]) at which EF is computed. Default: [0.005, 0.01, 0.05].
    """

    def __init__(
        self,
        calc_auc: bool = True,
        calc_bedroc: bool = True,
        calc_ef: bool = True,
        ef_fractions: List[float] = [0.005, 0.01, 0.05],
    ) -> None:

        # Disable synchronization at each step; sync happens on compute()
        super().__init__(dist_sync_on_step=False)

        self.calc_auc = calc_auc
        self.calc_bedroc = calc_bedroc
        self.calc_ef = calc_ef
        self.ef_fractions = ef_fractions

        if self.calc_auc:
            self.add_state("auc", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")

        if self.calc_bedroc:
            self.add_state("bedroc", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        
        if self.calc_ef:
            for fraction in ef_fractions:
                self.add_state(f"ef_{fraction}", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")

        self.add_state("total", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")

    
    def update(
        self,
        preds: Tensor,
        targets: Tensor,
        indexes: Tensor,
    ) -> None:
        """
        Update metric state with a batch of predictions.

        Args:
            preds (Tensor): predicted scores
            targets (Tensor): binary activity labels
            indexes (Tensor): group identifiers (per target protein
        """

        # Ensure shapes are consistent
        assert preds.shape == targets.shape == indexes.shape, "Shapes of predictions, labels and indexes do not match in VirtualScreeningMetrics."

        preds = torch.flatten(preds)
        targets = torch.flatten(targets)
        indexes = torch.flatten(indexes)

        for i in indexes.unique():

            mask = (indexes == i)
            self.total += 1

            y_score = preds[mask]
            y_true = targets[mask]

            # Randomize order before sorting (tie-breaking)
            shuffle_idx = torch.randperm(y_score.shape[0])
            y_score = y_score[shuffle_idx]
            y_true = y_true[shuffle_idx]

            if self.calc_auc:
                auc = auroc(y_score, y_true, task="binary")
                self.auc+=auc

            if self.calc_bedroc:
                scores_targets = torch.cat((torch.unsqueeze(y_score, axis=1), torch.unsqueeze(y_true, axis=1)), axis=1)
                scores_targets = scores_targets[torch.argsort(-scores_targets[:,0])]

                bedroc = CalcBEDROC(scores_targets, 1, 80.5)
                self.bedroc+=bedroc
            
            if self.calc_ef:
                for i, fraction in enumerate(self.ef_fractions):
                    ef = enrichment_factor(y_true, y_score, fraction)
                    setattr(self, f"ef_{fraction}", getattr(self, f"ef_{fraction}") + torch.tensor(ef, dtype=torch.float32))


    def compute(
        self
    ) -> Dict[str, float]:
        """
        Compute average metrics over all groups.
        """

        results = {}

        if self.calc_auc:
            results["auc"] = self.auc.item()/self.total.item()

        if self.calc_bedroc:
            results["bedroc"] = self.bedroc.item()/self.total.item()

        if self.calc_ef:
            for fraction in self.ef_fractions:
                results[f"ef_{fraction}"] = getattr(self, f"ef_{fraction}").item()/self.total.item()

        return results


class TargetFishingMetrics(Metric):
    """
    Metrics for evaluating target fishing performance.

    Metrics supported:
    - AUC
    - DeltaAUPRC
    - Enrichment Factor (EF) at multiple fractions

    Metrics are accumulated per unique target with `update()` and averaged in `compute()`.
    """

    def __init__(
        self,
        calc_auc: bool = True,
        calc_auprc: bool = True,
        calc_ef: bool = True,
        ef_fractions: List[float] = [0.05, 0.01, 0.005],
    ) -> None:
        
        # Disable synchronization at each step; sync happens on compute()
        super().__init__(dist_sync_on_step=False)

        self.calc_auc = calc_auc
        self.calc_auprc = calc_auprc
        self.calc_ef = calc_ef
        self.ef_fractions = ef_fractions      

        # Accumulated data
        self.add_state("scores", default=torch.empty(0, dtype=torch.float32), dist_reduce_fx=None)
        self.add_state("labels", default=torch.empty(0, dtype=torch.long), dist_reduce_fx=None)
        self.add_state("mol_inds", default=torch.empty(0, dtype=torch.long), dist_reduce_fx=None)


    def update(
        self,
        scores: Tensor,        # (N,)
        labels: Tensor,        # (N,)
        mol_inds: Tensor,      # (N,)
    ) -> None:
        """
        Accumulate predictions for target fishing metrics.

        Args:
            scores (Tensor): Predicted interaction scores for protein-ligand pairs.
            labels (Tensor): Binary ground-truth labels (0/1) for each pair.
            mol_inds (Tensor): Molecule identifiers for ranking per ligand.
        """

        assert scores.shape == labels.shape == mol_inds.shape, "Shapes of scores, labels and indexes do not match in TargetFishingMetrics."

        # Flatten
        scores = scores.flatten()
        labels = labels.flatten().long()
        mol_inds = mol_inds.flatten().long()

        # Accumulate
        self.scores = torch.cat([self.scores, scores])
        self.labels = torch.cat([self.labels, labels])
        self.mol_inds = torch.cat([self.mol_inds, mol_inds])


    def compute(
        self
    ) -> Dict[str, float]:
        """
        Compute target fishing metrics averaged across molecules.
        """

        # Flatten
        self.scores = self.scores.flatten()
        self.labels = self.labels.flatten()
        self.mol_inds = self.mol_inds.flatten()

        results = {}

        # Group by molecule (query)
        _, inverse = torch.unique(
            self.mol_inds, return_inverse=True
        )

        # Make molecule groups contiguous
        sorted_inverse, perm = torch.sort(inverse)

        scores_sorted = self.scores[perm]
        labels_sorted = self.labels[perm]

        counts = torch.bincount(sorted_inverse)

        score_groups = torch.split(scores_sorted, counts.tolist())
        label_groups = torch.split(labels_sorted, counts.tolist())

        auprc_sum = 0.0
        delta_auprc_sum = 0.0
        auc_sum = 0.0
        ef_sum = torch.zeros(len(self.ef_fractions), device=self.scores.device)

        n_mols = 0

        for y_score, y_true in zip(score_groups, label_groups):

            n = y_true.numel()
            n_pos = y_true.sum().item()

            # Skip molecules with no positive targets
            if n_pos == 0:
                continue

            pos_rate = n_pos / n

            # Compute AUPRC
            if self.calc_auprc:
                auprc = average_precision(y_score, y_true, task="binary")
                auprc_sum += auprc
                delta_auprc_sum += auprc - pos_rate

            # Compute AUC
            if self.calc_auc:
                auc_sum += auroc(y_score, y_true, task="binary")

            # Enrichment factors
            if self.calc_ef:
                for i, fraction in enumerate(self.ef_fractions):
                    ef = enrichment_factor(y_true, y_score, fraction)
                    ef_sum[i] += ef

            n_mols += 1

        # Final averages
        if self.calc_auprc:
            results["auprc"] = auprc_sum / n_mols
            results["delta_auprc"] = delta_auprc_sum / n_mols

        if self.calc_auc:
            results["auc"] = auc_sum / n_mols

        if self.calc_ef:
            for i, fraction in enumerate(self.ef_fractions):
                results[f"ef_{fraction}"] = ef_sum[i].item() / n_mols

        return results



class PocketPredictionMetrics(Metric):
    """
    Metric class for evaluating predicted binding pockets in proteins.
    
    Metrics supported:
        - DCC (Distance to Closest Center) and DCC_ranked
        - DCA (Distance to Closest Atom) and DCA_ranked
        - IOU (Jaccard index / segmentation overlap)
    
    The `update` method accumulates batch-level predictions, while `compute`
    returns averages over all accumulated data.
    
    Args:
        calc_dcc (bool): Whether to compute DCC metric.
        calc_dcc_ranked (bool): Whether to compute DCC using top-ranked pockets.
        calc_dca (bool): Whether to compute DCA metric.
        calc_dca_ranked (bool): Whether to compute DCA using top-ranked pockets.
        calc_iou (bool): Whether to compute IOU between predicted and true segmentation.
        threshold (float): Distance threshold (Å) for DCC/DCA calculations.
        rank_descending (bool): Whether to rank pockets descending by confidence.
    """

    def __init__(
        self, 
        calc_dcc=True,
        calc_dcc_ranked=True,
        calc_dca=True,
        calc_dca_ranked=True,
        calc_iou=True,
        threshold=4,
        rank_descending=True
    ) -> None:
        
        # Disable synchronization at each step; sync happens on compute()
        super().__init__(dist_sync_on_step=False)

        self.calc_dcc = calc_dcc
        self.calc_dcc_ranked = calc_dcc_ranked
        self.calc_dca = calc_dca
        self.calc_dca_ranked = calc_dca_ranked
        self.calc_iou = calc_iou

        self.threshold = threshold
        self.rank_descending = rank_descending

        # States to accumulate metrics
        if self.calc_dcc:
            self.add_state("dcc", default=torch.tensor(0), dist_reduce_fx="sum")

        if self.calc_dcc_ranked:
            self.add_state("dcc_ranked", default=torch.tensor(0), dist_reduce_fx="sum")

        if self.calc_dca:
            self.add_state("dca", default=torch.tensor(0), dist_reduce_fx="sum")
        
        if self.calc_dca_ranked:
            self.add_state("dca_ranked", default=torch.tensor(0), dist_reduce_fx="sum")

        # IOU uses TorchMetrics JaccardIndex (binary segmentation)
        if calc_iou:
            self.iou = JaccardIndex(task="binary")

        # Count of processed items (for averaging)
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")


    def update_dcc(
        self, 
        pocket_pos_clustered: Tensor,
        pocket_batch_idx: Tensor,
        pocket_centers: Tensor,
        pocket_center_batch_idx: Tensor,
    ) -> int:
        """
        Compute the DCC (Distance to Closest Center) count for a batch.

        For each ground-truth pocket center, checks whether there exists at least one
        predicted pocket center within `self.threshold` Å in the same batch item.

        Args:
            pocket_pos_clustered (Tensor): Predicted pocket center coordinates, shape (N_pockets, 3).
            pocket_batch_idx (Tensor): Batch indices for predicted pockets, shape (N_pockets,).
            pocket_centers (Tensor): Ground-truth pocket center coordinates, shape (N_centers, 3).
            pocket_center_batch_idx (Tensor): Batch indices for ground-truth centers, shape (N_centers,).

        Returns:
            int: Number of ground-truth pocket centers matched within the distance threshold.
        """

        # Pairwise distances between ground-truth centers and predicted pockets
        diffs = pocket_pos_clustered[None, :, :] - pocket_centers[:, None, :]
        dists = torch.norm(diffs, dim=-1)  # (N_pocket_centers, N_pockets)

        # Mask out predicted pockets from different batch items
        mask = pocket_batch_idx[None, :] != pocket_center_batch_idx[:, None]
        dists = dists.masked_fill(mask, float("inf"))

        # Count centers with at least one predicted pocket within threshold
        dcc_count = (dists < self.threshold).any(dim=1).sum().item()

        return dcc_count
    

    def update_dcc_ranked(
        self, 
        pocket_pos_clustered: Tensor,
        confidence_clustered: Tensor,
        pocket_batch_idx: Tensor,
        pocket_centers: Tensor,
        pocket_center_batch_idx: Tensor,
        pocket_counts
    ) -> int:
        """
        Compute ranked DCC by selecting top-confidence predicted pockets per cluster.

        For each cluster (e.g. protein), only the top `pocket_counts[i]` predicted pockets
        ranked by `confidence_clustered` are retained before computing DCC.

        Args:
            pocket_pos_clustered (Tensor): Predicted pocket center coordinates, shape (N_pockets, 3).
            confidence_clustered (Tensor): Confidence score for each predicted pocket, shape (N_pockets,).
            pocket_batch_idx (Tensor): Cluster/batch indices for predicted pockets, shape (N_pockets,).
            pocket_centers (Tensor): Ground-truth pocket center coordinates, shape (N_centers, 3).
            pocket_center_batch_idx (Tensor): Batch indices for ground-truth centers, shape (N_centers,).
            pocket_counts: Number of pockets to keep per cluster (indexed by cluster id).

        Returns:
            int: Ranked DCC count (number of ground-truth centers matched within threshold).
        """

        mask = torch.zeros_like(pocket_batch_idx, dtype=torch.bool)

        # Select top-ranked pockets per protein
        for i in pocket_batch_idx.unique():
            # Rank predicted pockets by confidence for protein i
            selected_rows = torch.argsort(confidence_clustered[pocket_batch_idx == i], descending=self.rank_descending)[:pocket_counts[i]]

            # Map local indices back to global indices
            selected_indices = (pocket_batch_idx == i).nonzero(as_tuple=True)[0][selected_rows]
            mask[selected_indices] = True

        # Filter predicted pockets to top-ranked ones
        pocket_pos_clustered = pocket_pos_clustered[mask, :]
        pocket_batch_idx = pocket_batch_idx[mask]

        # Compute standard DCC on the filtered predictions
        dcc_ranked = self.update_dcc(pocket_pos_clustered, pocket_batch_idx, pocket_centers, pocket_center_batch_idx)

        return dcc_ranked
    

    def update_dca(
        self, 
        pocket_pos_clustered: Tensor,
        pocket_batch_idx: Tensor,
        ligand_coords: Tensor, 
        ligand_batch_index: Tensor,
        ligand_inds: Tensor
    ) -> int:
        """
        Compute the DCA (Distance to Closest Atom) count for a batch.

        For each ligand, checks whether at least one of its atoms lies within
        `self.threshold` Å of any predicted pocket belonging to the same batch item.

        Args:
            pocket_pos_clustered (Tensor): Predicted pocket center coordinates, shape (N_pockets, 3).
            pocket_batch_idx (Tensor): Batch indices for predicted pockets, shape (N_pockets,).
            ligand_coords (Tensor): Ligand atom coordinates, shape (N_atoms, 3).
            ligand_batch_index (Tensor): Batch indices for ligand atoms, shape (N_atoms,).
            ligand_inds (Tensor): Ligand identifiers within each batch, shape (N_atoms,).

        Returns:
            int: Number of ligands with at least one atom within the distance threshold.
        """

         # Pairwise distances between ligand atoms and predicted pockets
        diffs = pocket_pos_clustered[None, :, :] - ligand_coords[:, None, :]
        dists = torch.norm(diffs, dim=-1)  # (N_ligand_atoms, N_pockets)

        # Mask out predicted pockets from different batch items
        mask = pocket_batch_idx[None, :] != ligand_batch_index[:, None]
        dists = dists.masked_fill(mask, float("inf"))

        # Identify ligand atoms close to any pocket
        atom_close = (dists < self.threshold).any(dim=1)  # (N_ligand_atoms,)
        
        # Construct unique ligand IDs across batch items
        pair_ids = ligand_batch_index * ligand_inds.max().add(1) + ligand_inds

        # Count ligands with at least one atom close to a pocket
        dca_count = 0
        for i in pair_ids.unique():
            if atom_close[pair_ids == i].any().item():
                dca_count += 1

        return dca_count

    
    def update_dca_ranked(
        self, 
        pocket_pos_clustered: Tensor,
        confidence_clustered: Tensor,
        pocket_batch_idx: Tensor,
        ligand_coords: Tensor, 
        ligand_batch_index: Tensor,
        ligand_inds: Tensor,
        pocket_counts
    ) -> int:
        """
        Compute ranked DCA by selecting top-confidence predicted pockets per cluster.

        For each cluster (e.g. protein), only the top `pocket_counts[i]` predicted pockets
        ranked by `confidence_clustered` are retained before computing DCA. DCA is then
        evaluated at the ligand level using the filtered pocket set.

        Args:
            pocket_pos_clustered (Tensor): Predicted pocket center coordinates, shape (N_pockets, 3).
            confidence_clustered (Tensor): Confidence score for each predicted pocket, shape (N_pockets,).
            pocket_batch_idx (Tensor): Cluster/batch indices for predicted pockets, shape (N_pockets,).
            ligand_coords (Tensor): Ligand atom coordinates, shape (N_atoms, 3).
            ligand_batch_index (Tensor): Batch indices for ligand atoms, shape (N_atoms,).
            ligand_inds (Tensor): Ligand identifiers within each batch, shape (N_atoms,).
            pocket_counts: Number of pockets to retain per cluster (indexed by cluster id).

        Returns:
            int: Ranked DCA count (number of ligands matched within the distance threshold).
        """

        # Boolean mask indicating which predicted pockets are retained
        mask = torch.zeros_like(pocket_batch_idx, dtype=torch.bool)

        # Select top-ranked pockets per protein
        for i in pocket_batch_idx.unique():
            selected_rows = torch.argsort(confidence_clustered[pocket_batch_idx == i], descending=self.rank_descending)[:pocket_counts[i]]

            # Map local cluster indices to global indices
            selected_indices = (pocket_batch_idx == i).nonzero(as_tuple=True)[0][selected_rows]
            mask[selected_indices] = True

        # Filter predicted pockets to top-ranked ones
        pocket_pos_clustered = pocket_pos_clustered[mask, :]
        pocket_batch_idx = pocket_batch_idx[mask]

        # Compute standard DCA on the filtered predictions
        dca_ranked = self.update_dca(pocket_pos_clustered, pocket_batch_idx, ligand_coords, ligand_batch_index, ligand_inds)

        return dca_ranked
    

    def update(
        self, 
        pocket_pos_clustered: Tensor, 
        confidence_clustered: Tensor, 
        pocket_batch_idx: Tensor,
        pocket_centers: Tensor, 
        pocket_center_batch_idx: Tensor, 
        pocket_counts: Tensor, 
        ligand_coords: Tensor, 
        ligand_batch_index: Tensor, 
        ligand_inds:  Tensor, 
        pred_segm: Tensor, 
        y_segm: Tensor, 
    ) -> None:
        """
        Update pocket prediction metrics for a batch.

        Accumulates DCC, DCA, and IOU metrics (and their ranked variants) depending
        on the enabled flags. Distance-based metrics are accumulated as counts and
        averaged later in `compute()`.

        Args:
            pocket_pos_clustered (Tensor): Predicted pocket center coordinates, shape (N_pockets, 3).
            confidence_clustered (Tensor): Confidence scores for predicted pockets, shape (N_pockets,).
            pocket_batch_idx (Tensor): Cluster/batch indices for predicted pockets.
            pocket_centers (Tensor): Ground-truth pocket center coordinates, shape (N_centers, 3).
            pocket_center_batch_idx (Tensor): Batch indices for ground-truth centers.
            pocket_counts (Tensor): Number of pockets to retain per cluster for ranked metrics.
            ligand_coords (Tensor): Ligand atom coordinates, shape (N_atoms, 3).
            ligand_batch_index (Tensor): Batch indices for ligand atoms.
            ligand_inds (Tensor): Ligand identifiers within each batch.
            pred_segm (Tensor): Predicted binary pocket segmentation.
            y_segm (Tensor): Ground-truth binary pocket segmentation.
        """

        if self.calc_dcc:
            self.dcc += self.update_dcc(pocket_pos_clustered, pocket_batch_idx, pocket_centers, pocket_center_batch_idx)

        if self.calc_dcc_ranked:
            self.dcc_ranked += self.update_dcc_ranked(pocket_pos_clustered, confidence_clustered, pocket_batch_idx, pocket_centers, pocket_center_batch_idx, pocket_counts)

        if self.calc_dca:
            self.dca += self.update_dca(pocket_pos_clustered, pocket_batch_idx, ligand_coords, ligand_batch_index, ligand_inds)
        
        if self.calc_dca_ranked:
            self.dca_ranked += self.update_dca_ranked(pocket_pos_clustered, confidence_clustered, pocket_batch_idx, ligand_coords, ligand_batch_index, ligand_inds, pocket_counts)
        
        if self.calc_iou:
            self.iou.update(pred_segm, y_segm)

        # Accumulate number of evaluated batch items (for averaging)
        self.total += len(pocket_batch_idx)


    def compute(
        self
    ) -> Dict[str, float]:
        """
        Compute pocket prediction metrics averaged across proteins.
        """

        results = {}

        if self.calc_dcc:
            results["dcc"] = self.dcc.item()/self.total.item()

        if self.calc_dcc_ranked:
            results["dcc_ranked"] = self.dcc_ranked.item()/self.total.item()

        if self.calc_dca:
            results["dca"] = self.dca.item()/self.total.item()

        if self.calc_dca_ranked:
            results["dca_ranked"] = self.dca_ranked.item()/self.total.item()

        if self.calc_iou:
            results["iou"] = self.iou.compute()
            self.iou.reset()

        return results
        


class PocketRankingMetrics(Metric):
    """
    Metrics for evaluating pocket ranking and confidence-based selection.

    This class evaluates whether the top-ranked or highest-confidence predicted
    pocket correctly matches a ground-truth pocket within a distance threshold.
    Metrics are accumulated per protein–ligand pair and aggregated across the dataset.

    Metrics supported:
        - dcc_rank: Success rate when selecting the top-ranked pocket
        - dcc_conf: Success rate when selecting the highest-confidence pocket
        - Wilson confidence intervals for both success rates
        - Average number of predicted pockets per protein

    Args:
        threshold (float): Distance threshold (Å) for pocket success.
        wilson_z (float): Z-score for Wilson confidence interval (default 1.96 for 95% CI).
    """

    def __init__(
        self, 
        threshold: float = 4.0, 
        wilson_z: float = 1.96
    ) -> None:

        # Disable synchronization at each step; sync happens on compute()
        super().__init__(dist_sync_on_step=False)

        self.threshold = threshold
        self.wilson_z = wilson_z

        # Accumulated success counts
        self.add_state("success_rank", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("success_conf", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")

        # Total number of evaluated protein–ligand pairs
        self.add_state("total", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")

        # Statistics for average number of predicted pockets
        self.add_state("num_pockets", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("num_proteins", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")

    
    def update(
        self, 
        pocket_pos_clustered: Tensor,
        confidence_clustered: Tensor,
        pocket_batch_idx: Tensor,
        pocket_preds: Tensor,
        ligand_inds: Tensor,
        mol_inds: Tensor,
        pocket_centers: Tensor,
        pocket_center_batch_idx: Tensor,
    ) -> None:
        """
        Update ranking-based pocket metrics for a batch.

        For each protein, and for each molecule:
            - Select the pocket with highest rank score based on similarity between pocket and ligand representations
            - Select the pocket with highest confidence score
            - Check whether either selection lies within `threshold` Å of
              any ground-truth pocket center

        Args:
            pocket_pos_clustered (Tensor): Predicted pocket center coordinates, (N_pockets, 3).
            confidence_clustered (Tensor): Confidence score for each predicted pocket.
            pocket_batch_idx (Tensor): Protein/cluster index for each predicted pocket.
            pocket_preds (Tensor): Pocket ranking scores per ligand.
            ligand_inds (Tensor): Ligand indices aligned with `pocket_preds`.
            mol_inds (Tensor): Molecule identifiers per ligand.
            pocket_centers (Tensor): Ground-truth pocket center coordinates.
            pocket_center_batch_idx (Tensor): Protein indices for ground-truth pocket centers.
        """

        # Iterate over proteins
        for i in pocket_batch_idx.unique():
            
            # Predicted pockets for this protein
            pocket_pos_i = pocket_pos_clustered[pocket_batch_idx == i]
            confidence_i = confidence_clustered[pocket_batch_idx == i]

            # Ligands and predictions associated with this protein
            mol_inds_i = mol_inds[ligand_inds == i]
            pocket_preds_i = pocket_preds[ligand_inds == i]

            # Ground-truth pocket centers for this protein
            pocket_centers_i = pocket_centers[pocket_center_batch_idx == i]

            # Iterate over molecules bound to this protein
            for j in mol_inds_i.unique():
                
                # Pocket ranking predictions for this protein–ligand pair
                pocket_preds_ij = pocket_preds_i[mol_inds_i == j][0]
                
                # Top-ranked pocket evaluation
                top_rank = torch.argmax(pocket_preds_ij)
                selected_pocket_rank = pocket_pos_i[top_rank]

                # Distance to closest ground-truth pocket
                d_rank = torch.norm(pocket_centers_i[mol_inds_i == j] - selected_pocket_rank, dim=1).min()  # min distance

                if d_rank <= self.threshold:
                    self.success_rank += 1

                # Top-confidence pocket evaluation
                top_conf = torch.argmax(confidence_i)
                selected_pocket_conf = pocket_pos_i[top_conf]
                d_conf = torch.norm(pocket_centers_i[mol_inds_i == j] - selected_pocket_conf, dim=1).min()

                if d_conf <= self.threshold:
                    self.success_conf += 1

                # Count evaluated protein–ligand pair
                self.total += 1

            # Track statistics for average pocket count
            self.num_pockets += len(pocket_pos_i)
            self.num_proteins += 1


    def wilson_ci(
        self,
        successes: float,
        total: float,
    ) -> Tuple[float, float]:
        """
        Compute Wilson confidence interval for a binomial proportion.

        Args:
            successes (float): Number of successful trials.
            total (float): Total number of trials.

        Returns:
            tuple[float, float]:
                (lower_bound, upper_bound) of the Wilson confidence interval.
        """
        
        if total == 0:
            return (0, 0)
        
        p_hat = successes / total
        denominator = 1 + (self.wilson_z**2 / total)
        center = (p_hat + (self.wilson_z**2 / (2*total))) / denominator
        margin = (self.wilson_z * np.sqrt((p_hat * (1 - p_hat) + (self.wilson_z**2 / (4*total))) / total)) / denominator

        return (center - margin, center + margin)


    def compute(
        self
    ) -> Dict[str, float]:
        """
        Compute pocket ranking metrics averaged across protein-ligand pairs.
        """

        results = {}

        results["dcc_rank"] = self.success_rank.item()/self.total.item()
        results["ci_lower_rank"], results["ci_upper_rank"] = self.wilson_ci(self.success_rank.item(), self.total.item())
        results["dcc_conf"] = self.success_conf.item()/self.total.item()
        results["ci_lower_conf"], results["ci_upper_conf"] = self.wilson_ci(self.success_conf.item(), self.total.item())
        results["avg_num_pockets"] = self.num_pockets.item()/self.num_proteins.item()

        return results
       


