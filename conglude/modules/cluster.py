import pytorch_lightning as pl
import torch
import numpy as np
from sklearn.cluster import MeanShift, DBSCAN



class MeanShiftCluster(pl.LightningModule):
    """
    Performs clustering of predicted pocket positions and features using Mean Shift.

    Parameters
    ----------
    bandwidth : float or None
        Bandwidth parameter for MeanShift clustering. If None, bandwidth is estimated automatically by the sklearn MeanShift implementation.
    """

    def __init__(
        self, 
        bandwidth: float = None
    ):
        
        super().__init__()

        # Initialize sklearn's MeanShift clustering object
        self.mean_shift = MeanShift(bandwidth=bandwidth)


    def forward(self, vn_pos_rearranged, vn_feats_rearranged, confidence_rearranged):
        """
        Cluster predicted pocket positions and aggregate corresponding features.

        Parameters
        ----------
        vn_pos_rearranged : torch.Tensor, shape [batch_size, num_positions, 3]
            Predicted coordinates for all candidate pockets per protein.
        vn_feats_rearranged : torch.Tensor, shape [batch_size, num_positions, feat_dim]
            Node features corresponding to each predicted pocket.
        confidence_rearranged : torch.Tensor, shape [batch_size, num_positions]
            Confidence scores for each predicted pocket.

        Returns
        -------
        vn_pos_clustered : torch.Tensor, shape [num_clusters, 3]
            Clustered pocket coordinates.
        vn_feats_clustered : torch.Tensor, shape [num_clusters, feat_dim]
            Clustered features, averaged per cluster.
        confidence_clustered : torch.Tensor, shape [num_clusters]
            Clustered confidence scores (averaged per cluster).
        pocket_batch_idx : torch.Tensor, shape [num_clusters]
            Index mapping each cluster back to its original protein in the batch.
        """

        # Lists to collect clustered outputs
        vn_pos_clustered = []
        vn_feats_clustered = []
        confidence_clustered = []
        pocket_batch_idx = []

        # Loop over proteins in the batch
        for i in range(vn_pos_rearranged.shape[0]):

            # Extract predicted positions and confidences for this protein
            vn_pos_sample = vn_pos_rearranged[i, :, :]
            confidence_sample = confidence_rearranged[i, :]

            # Cluster positions using MeanShift on combined coordinates and confidence
            cluster_labels = self.mean_shift.fit_predict(
                torch.cat([vn_pos_sample, confidence_sample.unsqueeze(1)], dim=1)
                .detach()
                .cpu()
                .numpy()
            )

            # Find unique cluster labels
            unique_labels = np.unique(cluster_labels)

            # Aggregate positions, features, and confidence per cluster
            for cluster_label in unique_labels:
                vn_pos_clustered.append(
                    torch.mean(vn_pos_sample[cluster_labels == cluster_label], dim=0)
                )
                vn_feats_clustered.append(
                    torch.mean(
                        vn_feats_rearranged[i, cluster_labels == cluster_label, :], dim=0
                    )
                )
                confidence_clustered.append(
                    torch.mean(confidence_sample[cluster_labels == cluster_label], dim=0)
                )

            # Keep track of which protein each cluster belongs to
            pocket_batch_idx.extend([i] * len(unique_labels))

        # Stack lists into tensors
        vn_pos_clustered = torch.stack(vn_pos_clustered)
        vn_feats_clustered = torch.stack(vn_feats_clustered)
        confidence_clustered = torch.stack(confidence_clustered)
        pocket_batch_idx = torch.tensor(pocket_batch_idx)

        return vn_pos_clustered, vn_feats_clustered, confidence_clustered, pocket_batch_idx



class DBSCANCluster(pl.LightningModule):
    """
    Cluster predicted pocket positions using DBSCAN.

    Parameters
    ----------
    eps: float
        Maximum distance between two points to be considered neighbors in DBSCAN.
    """

    def __init__(
        self, 
        eps: float = 4.0
    ):
        
        super().__init__()
        self.eps = eps


    def forward(self, vn_pos_rearranged, vn_feats_rearranged, confidence_rearranged):
        """
        Cluster predicted pockets using DBSCAN and aggregate their features.

        Parameters
        ----------
        vn_pos_rearranged : torch.Tensor, shape [batch_size, num_positions, 3]
            Predicted pocket coordinates per protein in the batch.
        vn_feats_rearranged : torch.Tensor, shape [batch_size, num_positions, feat_dim]
            Node features corresponding to each predicted pocket.
        confidence_rearranged : torch.Tensor, shape [batch_size, num_positions]
            Confidence scores for each predicted pocket.

        Returns
        -------
        vn_pos_clustered : torch.Tensor, shape [num_clusters, 3]
            Clustered pocket coordinates.
        vn_feats_clustered : torch.Tensor, shape [num_clusters, feat_dim]
            Clustered node features (averaged per cluster).
        confidence_clustered : torch.Tensor, shape [num_clusters]
            Clustered confidence scores (averaged per cluster).
        pocket_batch_idx : torch.Tensor, shape [num_clusters]
            Index mapping each cluster back to its original protein in the batch.
        """

        # Lists to collect final clustered outputs
        vn_pos_clustered = []
        vn_feats_clustered = []
        confidence_clustered = []
        pocket_batch_idx = []

        # Loop over proteins in the batch
        for i in range(vn_pos_rearranged.shape[0]):

            # Extract predicted pocket positions for this protein
            vn_pos_sample = vn_pos_rearranged[i, :, :]

            # Perform DBSCAN clustering on positions
            cluster_labels = DBSCAN(
                eps=self.eps, min_samples=1
            ).fit_predict(vn_pos_sample.detach().cpu().numpy())

            # Get unique cluster labels
            unique_labels = np.unique(cluster_labels)

            # Aggregate positions, features, and confidence per cluster
            for cluster_label in unique_labels:
                if cluster_label == -1:
                    continue  # Ignore noise points flagged by DBSCAN

                mask = cluster_labels == cluster_label  # Points in this cluster

                # Average coordinates of points in the cluster
                vn_pos_clustered.append(torch.mean(vn_pos_sample[mask], dim=0))

                # Average node features of points in the cluster
                vn_feats_clustered.append(torch.mean(vn_feats_rearranged[i, mask, :], dim=0))

                # Average confidence scores of points in the cluster
                confidence_clustered.append(torch.mean(confidence_rearranged[i, mask], dim=0))

            # Keep track of which protein each cluster belongs to
            pocket_batch_idx.extend([i] * len([l for l in unique_labels if l != -1]))

        # Convert lists to stacked tensors for downstream use
        vn_pos_clustered = torch.stack(vn_pos_clustered)
        vn_feats_clustered = torch.stack(vn_feats_clustered)
        confidence_clustered = torch.stack(confidence_clustered)
        pocket_batch_idx = torch.tensor(pocket_batch_idx)

        return vn_pos_clustered, vn_feats_clustered, confidence_clustered, pocket_batch_idx
