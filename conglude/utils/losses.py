import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from typing import Tuple
from torch import Tensor



class VNPositionHuberLoss(nn.Module):
    """
    Equivariant Huber loss for VN position prediction selecting the closest VN per graph.
    """

    def forward(
        self, 
        true_positions: Tensor, 
        pred_vn_positions: Tensor, 
        vn_batch_index: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute VN position loss by selecting the minimum Huber distance per graph.

        Args:
            true_positions (Tensor): Ground-truth coordinates indexed by vn_batch_index, shape (N_nodes, 3).
            pred_vn_positions (Tensor): Predicted VN coordinates with multiple hypotheses per graph, shape (B * G, 3).
            vn_batch_index (Tensor): Batch index mapping VN predictions to graphs, shape (B * G,).

        Returns:
            Tuple[Tensor, Tensor]: Mean minimum Huber loss (scalar) and selected VN index per graph (B,).
        """

        number_vns = int(pred_vn_positions.shape[0] / torch.unique(vn_batch_index).shape[0])

        # Compute equivariant distance-based Huber loss
        distances_positions = F.huber_loss(torch.norm(pred_vn_positions - true_positions[vn_batch_index], dim=-1), torch.zeros_like(pred_vn_positions[..., 0]), reduction="none")

        # Reshape to (batch_size, num_vns)
        distances_positions_rearranged = rearrange(distances_positions, "(b g) -> b g", g=number_vns)

        # Select closest VN per graph
        distance_position_min_index = torch.argmin(distances_positions_rearranged, dim=-1)
        distance_position_min = distances_positions_rearranged[torch.arange(distances_positions_rearranged.size(0)), distance_position_min_index]

        return distance_position_min.mean(), distance_position_min_index

    

class ConfidenceLoss(nn.Module):
    """
    Regression loss for pocket confidence prediction.

    Encourages high confidence for pockets close to the ligand
    and low confidence for distant pockets using a distance-based target.
    """

    def __init__(
        self, 
        gamma: float = 4.0, 
        c0: float = 0.001):
        """
        Args:
            gamma (float): Distance cutoff controlling confidence decay.
            c0 (float): Minimum confidence assigned to distant pockets.
        """

        super().__init__() 

        self.gamma = gamma
        self.c0 = c0
        self.loss = nn.MSELoss()


    def forward(
        self, 
        pocket_dists: Tensor, 
        confidence_predictions: Tensor
    ) -> Tensor:
        """
        Compute confidence regression loss.

        Args:
            pocket_dists (Tensor): Pocket–ligand distances, shape (N_pockets,).
            confidence_predictions (Tensor): Predicted confidence scores, shape (N_pockets,).

        Returns:
            Tensor: Scalar MSE loss.
        """

        # Detach to avoid gradients through distances
        c = pocket_dists.detach().clone()

        # Decay confidence for close pockets
        c[c <= self.gamma] = 1 - c[c <= self.gamma] / (self.gamma * 2)

        # Constant low confidence for distant pockets
        c[c > self.gamma] = self.c0

        return self.loss(confidence_predictions.squeeze(), c)



class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.
    """

    def __init__(
        self,
        smooth: float = 1.0
    ) -> None:
        """
        Args:
            smooth (float): Smoothing constant to avoid division by zero.
        """

        super().__init__()
        self.smooth = smooth


    def forward(
        self, 
        input: Tensor, 
        targets: Tensor
    ) -> Tensor:
        """
        Compute Dice loss.

        Args:
            input (Tensor): Raw logits.
            targets (Tensor): Binary ground-truth labels.

        Returns:
            Tensor: Scalar Dice loss.
        """

        # Convert logits to probabilities
        probs = torch.sigmoid(input)
        
        # Flatten for global overlap
        probs, targets = probs.view(-1), targets.view(-1)

        # Foreground overlap
        intersection = (probs * targets).sum()
        dice = (2 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

        return 1 - dice

    

class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss for contrastive learning.
    """

    def __init__(
        self, 
        temperature: float = 0.07, 
        dim: int = 0
    ) -> None:
        """
        Args:
            temperature (float): Softmax temperature.
            dim (int): Contrastive dimension (0 or 1).
        """

        super().__init__()

        self.temperature = temperature 
        self.dim = dim


    def forward(
        self,
        preds: Tensor, 
        labels: Tensor
    ) -> Tensor:
        """
        Compute InfoNCE loss.

        Args:
            preds (Tensor): Similarity logits.
            labels (Tensor): One-hot encoded positives; ignored entries set to -100.

        Returns:
            Tensor: Scalar contrastive loss.
        """
        # Temperature-scale logits
        logits = preds.T / self.temperature if self.dim == 0 else preds / self.temperature
        labels = labels.T if self.dim == 0 else labels

        # Mask invalid pairs
        logits[labels == -100] = -float("inf")

        return F.cross_entropy(logits, labels.argmax(dim=1))



class BCELoss(nn.Module):
    """
    Binary cross-entropy loss with optional group-wise averaging.
    """

    def __init__(
        self, 
        scaling: float = 1.0, 
        shift: float = 0.0
    ) -> None:
        """
        Initialize BCE loss.

        Args:
            scaling (float): Logit scaling factor.
            shift (float): Logit shift.
        """

        super().__init__() 
        self.scaling = scaling
        self.shift = shift
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")


    def forward(
        self, 
        preds: Tensor, 
        labels: Tensor = None,
        group_indices: Tensor = None
    ) -> Tensor:
        """
        Compute BCE loss with optional group-wise averaging.

        Args:
            preds (Tensor): Logit similarity matrix.
            labels (Tensor, optional): Binary label matrix; identity if None.
            group_indices (Tensor, optional): Group labels for per-group averaging.

        Returns:
            Tensor: Scalar BCE loss.
        """

        # Default positives on diagonal
        if labels is None: 
            labels = torch.eye(preds.size(0), device=preds.device)

        # Affine logit transform
        logits = (preds + self.shift) * self.scaling

        loss = self.bce_loss(logits, labels)

        if group_indices is None: 
            return loss.mean()
        
        # Average per group
        return torch.stack([loss[group_indices == g].mean() for g in torch.unique(group_indices)]).mean()
