from typing import Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import (
    Aggregation,
    MeanAggregation,
    MessagePassing,
    SumAggregation,
)
from torch_geometric.typing import Adj, OptPairTensor, OptTensor
from torch_geometric.nn.norm import LayerNorm


class CoordsNorm(nn.Module):
    """
    This module normalizes input coordinates to unit length and then rescales them with a learnable scalar parameter.
    From: https://github.com/lucidrains/egnn-pytorch

    Parameters
    ----------
    eps : float
        Small constant added for numerical stability to avoid division by zero.
    scale_init : float
        Initial value of the learnable scaling parameter applied after normalization.
    """

    def __init__(
        self, 
        eps: float = 1e-8, 
        scale_init: float = 1.0
    ):
        
        super().__init__()

        self.eps = eps

        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)


    def forward(
        self, 
        coords: Tensor
    ) -> Tensor:
        """
        Normalize and rescale coordinates.

        Parameters
        ----------
        coords: torch.Tensor
            Tensor of shape (..., 3)

        Returns
        -------
        torch.Tensor
            Normalized coordinates scaled by a learnable factor.
        """
        norm = coords.norm(dim=-1, keepdim=True)
        normed_coords = coords / norm.clamp(min=self.eps)

        return normed_coords * self.scale



class GNNLayer(MessagePassing):
    """
    Generic message passing GNN layer performing message computation between connected nodes, aggregation of incoming messages,
    node feature update using aggregated messages with optional residual connections and feature normalization.

    Parameters
    ----------
    node_features: int
        Dimensionality of input node features.
    hidden_features: int
        Hidden dimensionality inside message and update MLPs.
    out_features: int
        Output dimensionality of node features.
    act: nn.Module
        Activation function.
    dropout: nn.Module
        Dropout module applied inside MLPs.
    residual: bool
        Whether to use residual (skip) connections. Requires node_features == out_features.
    node_aggr: str
        Aggregation method ("mean", "sum", etc.).
    norm_feats: bool
        Whether to apply LayerNorm to node features before message computation.
    initialization_gain: float
        Gain factor for Xavier weight initialization.
    """
        
    def __init__(
        self,
        node_features: int,
        hidden_features: int,
        out_features: int,
        act: nn.Module,
        dropout: nn.Module = nn.Dropout(0.5),
        residual: bool = True,
        node_aggr: str = "mean",
        norm_feats: bool = True,
        initialization_gain: float = 1.0,
    ):
        
        super().__init__(aggr=node_aggr)  # Use sum aggregation

        self.residual = residual
        self.act = act
        self.initialization_gain = initialization_gain

         # Residual connections require matching dimensions
        if node_features != out_features and residual:
            raise ValueError("Residual connections require input and output dims to match.")

        # Message network
        self.message_net = nn.Sequential(
            nn.Linear(2 * node_features, hidden_features),
            dropout,
            act,
            nn.Linear(hidden_features, hidden_features),
        )

        # Update network
        self.update_net = nn.Sequential(
            nn.Linear(node_features + hidden_features, hidden_features),
            dropout,
            act,
            nn.Linear(hidden_features, out_features),
        )

        # Optional feature normalization before message computation
        self.node_norm = LayerNorm(node_features) if norm_feats else nn.Identity()

        # Initialize weights
        self.apply(self.init_weights)


    def init_weights(
        self, 
        module: nn.Module
    ):
        """
        Xavier initialization for linear layers.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=self.initialization_gain)
            nn.init.zeros_(module.bias)


    def message(
        self, 
        x_i: Tensor, 
        x_j: Tensor, 
        edge_attr: OptTensor = None
     ) -> Tensor:
        """
        Compute messages for each edge.

        Parameters
        ----------
        x_i: 
            target node features
        x_j: 
            source node features
        edge_attr:
            Optional edge features

        Returns
        -------
        Tensor            
            Messages to be aggregated for each target node.
        """

        # Normalize node features before message computation
        input = [self.node_norm(x_i), self.node_norm(x_j)]

         # Optionally include edge attributes
        if edge_attr is not None:
            input.append(edge_attr)
        
        # Concatenate features and pass through message MLP
        input = torch.cat(input, dim=-1)

        return self.message_net(input)


    def update(
        self, 
        aggr_out, 
        x: Union[torch.Tensor, OptPairTensor]
    ) -> Tensor:
        """
        Update node features using aggregated messages. Optionally add residual connection.     

        Parameters
        ----------
        aggr_out: Tensor
            Aggregated messages for each node.
        x: Union[torch.Tensor, OptPairTensor]
            Original node features (or pair of (src, dst) features if heterogeneous).   
        
        Returns
        -------
        Tensor
            Updated node features.
        """
        # In bipartite case, x[1] corresponds to target node features
        x_ = x if isinstance(x, torch.Tensor) else x[1]

        # Concatenate original node features with aggregated messages
        input = torch.cat([x_, aggr_out], dim=-1)

        # Pass through update MLP
        return self.update_net(input)


    def forward(
        self,
        x: Union[torch.Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
    ):
        """
        Forward pass of the GNN layer.

        Parameters
        ----------
        x : torch.Tensor or pair of tensors
            Node features. If bipartite graph, passed as (x_src, x_dst).
        edge_index : torch.Tensor
            Edge index in COO format.
        edge_attr : torch.Tensor, optional
            Edge features.

        Returns
        -------
        torch.Tensor
            Updated node features.
        """

        # Save residual (only target node features in bipartite case)
        if self.residual:
            residual = x if isinstance(x, torch.Tensor) else x[1]
        
        # Perform message passing (calls message + aggregate + update)
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Add residual connection if enabled
        if self.residual:
            x = x + residual

        return x



class EGNNLayer(MessagePassing):
    """
    E(n)-Equivariant Graph Neural Network (EGNN) layer performing joint updates of node features and coordinates while preserving E(n)-equivariance (i.e., equivariance to rotations,
    translations, and reflections in n-dimensional space).

    Parameters
    ----------
    node_features: int
        Dimensionality of input node features.
    edge_features: int
        Dimensionality of edge features.
    hidden_features: int
        Hidden dimensionality inside message/update MLPs.
    out_features: int
        Output dimensionality of node features.
    act: nn.Module
        Activation function.
    dropout: nn.Module
        Dropout module used inside MLPs.
    node_aggr: Aggregation
        Aggregation function for feature messages (e.g., sum, mean).
    coord_aggr: Aggregation
        Aggregation function for coordinate updates.
    residual: bool
        Whether to use residual feature connections. Requires node_features == out_features.
    update_coords: bool
        Whether to update node coordinates.
    norm_coords: bool
        Whether to normalize relative coordinates before computing updates.
    norm_coords_scale_init: float
        Initial scaling factor for coordinate normalization.
    norm_feats: bool
        Whether to apply LayerNorm to node features before message computation.
    initialization_gain: float
        Gain factor for Xavier initialization.
    return_pos: bool
        Whether to return updated coordinates.
    """


    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_features: int,
        out_features: int,
        act: nn.Module,
        dropout: nn.Module = nn.Dropout(0.5),
        node_aggr: Aggregation = SumAggregation,
        coord_aggr: Aggregation = MeanAggregation,
        residual: bool = True,
        update_coords: bool = True,
        norm_coords: bool = True,
        norm_coords_scale_init: float = 1e-2,
        norm_feats: bool = True,
        initialization_gain: float = 1,
        return_pos: bool = True,
    ):
        
        # Disable default aggregation because feature and coordinate aggregation is handled separately
        super().__init__(aggr=None)

        self.node_aggr = node_aggr
        self.coord_aggr = coord_aggr
        self.residual = residual
        self.update_coords = update_coords
        self.act = act
        self.initialization_gain = initialization_gain
        self.return_pos = return_pos

        # Residual connections require matching feature dimensions
        if (node_features != out_features) and residual:
            raise ValueError(
                "Residual connections are only compatible with the same input and output dimensions."
            )

        # Message network
        self.message_net = nn.Sequential(
            nn.Linear(2 * node_features + edge_features, hidden_features),
            dropout,
            act,
            nn.Linear(hidden_features, hidden_features),
        )

        # Feature update network
        self.update_net = nn.Sequential(
            nn.Linear(node_features + hidden_features, hidden_features),
            dropout,
            act, 
            nn.Linear(hidden_features, out_features),
        )

        # Coordinate update network
        self.pos_net = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            dropout,
            act,
            nn.Linear(hidden_features, 1),
        )

        # Optional feature normalization
        self.node_norm = (
            LayerNorm(node_features) if norm_feats else nn.Identity()
        )

        # Optional coordinate normalization
        self.coords_norm = (
            CoordsNorm(scale_init=norm_coords_scale_init) if norm_coords else nn.Identity()
        )

        # Initialize weights
        self.apply(self.init_)


    def init_(
        self,
        module: nn.Module
    ):
        """
        Xavier initialization for linear layers, with special handling for SELU activation.
        """

        if type(module) in {nn.Linear}:
            if (type(self.act) is nn.SELU):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear", mode="fan_in")
                nn.init.zeros_(module.bias)
            else:
                nn.init.xavier_normal_(module.weight, gain=self.initialization_gain)
                nn.init.zeros_(module.bias)


    def message(
        self, 
        x_i: Tensor, 
        x_j: Tensor, 
        pos_i: Tensor, 
        pos_j: Tensor, 
        edge_weight: OptTensor = None
    ):
        """
        Compute messages for each edge.

        Parameters
        ----------
        x_i: 
            target node features
        x_j: 
            source node features
        pos_i:
            target node coordinates
        pos_j:
            source node coordinates
        edge_weight:
            Optional edge features

        Returns
        -------
        Tensor            
            Messages to be aggregated for each target node.
        """

        # Compute relative position and distance between source and target nodes
        pos_dir = pos_i - pos_j
        dist = torch.norm(pos_dir, dim=-1, keepdim=True)

        # Normalize node features before message computation
        input = [self.node_norm(x_i), self.node_norm(x_j), dist]
        input = torch.cat(input, dim=-1)

        # Pass through message MLP
        node_message = self.message_net(input)
        pos_message = self.coords_norm(pos_dir) * self.pos_net(node_message)

        # Optionally weight messages by edge weights
        if edge_weight is not None:
            node_message = node_message * edge_weight.unsqueeze(-1)
            pos_message = pos_message * edge_weight.unsqueeze(-1)

        return node_message, pos_message
    

    def aggregate(
        self,
        inputs: Tuple[Tensor, Tensor],
        index: Tensor,
        ptr: Tensor = None,
        dim_size: int = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Aggregate messages from neighbors for nodes and coordinates separately.

        Parameters
        ----------
        inputs: tuple of torch.Tensor
            Tuple containing:
                - node_message: Tensor of shape [num_edges, node_feat_dim], messages for node features
                - pos_message: Tensor of shape [num_edges, coord_dim], messages for coordinates
        index: torch.Tensor
            Target node indices for each edge; used for aggregation.
        ptr: torch.Tensor, optional
            Optional pointer for segment aggregation (used internally by PyG).
        dim_size: int, optional
            Optional number of nodes for the output size.

        Returns
        -------
        tuple of torch.Tensor
            - agg_node_message: Aggregated messages for node features [num_nodes, node_feat_dim]
            - agg_pos_message: Aggregated messages for coordinates [num_nodes, coord_dim]
        """

        # Unpack node and coordinate messages
        node_message, pos_message = inputs

        # Aggregate node messages using chosen aggregation function (sum, mean, etc.)
        agg_node_message = self.node_aggr(node_message, index, ptr, dim_size)

        # Aggregate coordinate messages separately (usually mean aggregation)
        agg_pos_message = self.coord_aggr(pos_message, index, ptr, dim_size)

        return agg_node_message, agg_pos_message


    def update(
        self,
        message: Tuple[Tensor, Tensor],
        x: Union[Tensor, OptPairTensor],
        pos: Union[Tensor, OptPairTensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        Update node features and coordinates using aggregated messages.

        Parameters
        ----------
        message : tuple of torch.Tensor
            Tuple containing:
                - node_message: Aggregated node feature messages [num_nodes, hidden_dim]
                - pos_message: Aggregated coordinate messages [num_nodes, coord_dim]
        x : torch.Tensor or pair of tensors
            Original node features. If bipartite graph, x[1] corresponds to target nodes.
        pos : torch.Tensor or pair of tensors
            Original coordinates. If bipartite, pos[1] corresponds to target nodes.

        Returns
        -------
        tuple of torch.Tensor
            - x_new: Updated node features [num_nodes, out_features]
            - pos_new: Updated coordinates [num_nodes, coord_dim] (may remain unchanged if update_coords=False)
        """
    
        # Unpack messages
        node_message, pos_message = message

        # Select target node features/coordinates (for bipartite graphs use x[1], pos[1])
        x_, pos_ = (x, pos) if isinstance(x, Tensor) else (x[1], pos[1])

        # Concatenate original node features with aggregated node messages
        input = torch.cat((x_, node_message), dim=-1)

        # Update node features using MLP
        x_new = self.update_net(input)

        # Update coordinates if enabled; otherwise keep them unchanged
        pos_new = pos_ + pos_message if self.update_coords else pos_

        return x_new, pos_new


    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        pos: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptPairTensor = None,
    ):
        """
        Forward pass of the EGNN layer.

        Parameters
        ----------
        x : torch.Tensor or pair of tensors
            Node features. If bipartite graph, passed as (x_src, x_dst).
        pos : torch.Tensor or pair of tensors
            Node coordinates. If bipartite graph, passed as (pos_src, pos_dst).
        edge_index : torch.Tensor
            Edge index in COO format.
        edge_weight : torch.Tensor, optional
            Edge weights for message weighting.
        edge_attr : torch.Tensor, optional
            Edge features.

        Returns
        -------
        torch.Tensor
            Updated node features.
        """

        # Save residual (only target node features in bipartite case)    
        if self.residual:
            residual = x if isinstance(x, Tensor) else x[1]

        # Perform message passing (calls message + aggregate + update)
        x_dest, pos = self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr, edge_weight=edge_weight)

        # Add residual connection if enabled
        if self.residual:
            x_dest = x_dest + residual

        out = (x_dest, pos) if self.return_pos else x_dest
        
        return out