import torch.nn as nn
import torch
import random
import pytorch_lightning as pl
from conglude.modules.utils.gnn_layers import EGNNLayer, GNNLayer
from conglude.modules.utils.mappings import activation_mapping, aggregation_mapping, dropout_mapping


class VNEGNN(pl.LightningModule):
    """
    Virtual Node Equivariant Graph Neural Network (VN-EGNN).

    Parameters
    ----------
    input_features: int
        Dimensionality of input node features.
    node_features: int
        Internal node embedding dimension.
    edge_features: int
        Dimensionality of edge features.
    hidden_features: int
        Hidden dimension inside message passing networks.
    out_features: int
        Output node embedding dimension after each layer.
    num_layers: int
        Number of message passing iterations.
    act: str
        Activation function key (see `activation_mapping`).
    dropout: float
        Dropout probability.
    node_aggr: str
        Aggregation method for node feature messages.
    coord_aggr: str
        Aggregation method for coordinate updates.
    residual: bool
        Whether to use residual connections.
    norm_coords: bool
        Whether to normalize coordinate updates.
    norm_coords_scale_init: float
        Initial scaling factor for coordinate normalization.
    norm_feats: bool
        Whether to normalize node features.
    update_coords: bool
        Whether to update coordinates during message passing.
    initialization_gain: float
        Gain factor for weight initialization.
    scaling_factor: float
        Optional factor to rescale input coordinates before message passing.
    weight_share: bool
        If True, reuse the same layer across all message passing steps.
    protein_node: bool
        If True, include a global protein node and its interactions.
    """

    def __init__(
        self,
        input_features: int = 1280,
        node_features: int = 100,
        edge_features: int = 1, 
        hidden_features: int = 100,
        out_features: int = 100,
        num_layers: int = 5,
        act: str = "silu",
        dropout: float = 0.1,
        node_aggr: str = "mean",
        coord_aggr: str = "mean",
        residual: bool = True,
        norm_coords: bool = True,
        norm_coords_scale_init: float = 0.01,
        norm_feats: bool = True,
        update_coords: bool = False,
        initialization_gain: float = 1,
        scaling_factor: float = 5,
        weight_share: bool = False,
        protein_node: bool = True,
    ):
        self.save_hyperparameters(
            logger=False,
        )

        super().__init__()

        self.input_features = input_features
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.num_layers = num_layers
        self.act = act
        self.dropout = dropout

        self.node_aggr = node_aggr
        self.coord_aggr = coord_aggr
        self.residual = residual
        self.norm_coords = norm_coords
        self.norm_coords_scale_init = norm_coords_scale_init
        self.norm_feats = norm_feats
        self.update_coords = update_coords
        self.initialization_gain = initialization_gain
        self.scaling_factor = scaling_factor
        self.weight_share = weight_share
        self.protein_node = protein_node
        
        # Build all learnable modules
        self.initialize_layers()

         # Special initialization for SELU-based activation
        if act == "selu":
            self.apply(self.init_)


    def init_(
        self, 
        module: nn.Module
    ):
        """
        Custom weight initialization for SELU activation.
        """
        if type(module) in {nn.Linear}:
            nn.init.kaiming_normal_(module.weight, nonlinearity="linear", mode="fan_in")
            nn.init.zeros_(module.bias)


    def create_vnegnn_layer(
            self,
        ):
        """
        Create a single VN-EGNN message passing block consisting of multiple (E)GNN modules:
            1. Residue-to-residue EGNN
            2. Residue-to-pocket EGNN
            3. Pocket-to-residue EGNN
            4. (Optional) Residue-to-protein GNN
            5. (Optional) Protein-to-residue GNN

        Returns
        -------
        nn.ModuleDict
            Dictionary of interaction modules for this layer.
        """
        module_dict = nn.ModuleDict(
            {
                "residue_to_residue": EGNNLayer(
                    node_features=self.node_features,
                    edge_features=self.edge_features,
                    hidden_features=self.hidden_features,
                    out_features=self.out_features,
                    act=activation_mapping[self.act],
                    dropout=dropout_mapping[self.act](self.dropout),
                    node_aggr=aggregation_mapping[self.node_aggr],
                    coord_aggr=aggregation_mapping[self.coord_aggr],
                    residual=self.residual,
                    update_coords=self.update_coords,
                    norm_coords=self.norm_coords,
                    norm_coords_scale_init=self.norm_coords_scale_init,
                    norm_feats=self.norm_feats,
                    initialization_gain=self.initialization_gain,
                ),
                "residue_to_pocket": EGNNLayer(
                    node_features=self.node_features,
                    edge_features=self.edge_features,
                    hidden_features=self.hidden_features,
                    out_features=self.out_features,
                    act=activation_mapping[self.act],
                    dropout=dropout_mapping[self.act](self.dropout),
                    node_aggr=aggregation_mapping[self.node_aggr],
                    coord_aggr=aggregation_mapping[self.coord_aggr],
                    residual=self.residual,
                    update_coords=True, # always update pocket positions based on residue-to-pocket messages
                    norm_coords=self.norm_coords,
                    norm_coords_scale_init=self.norm_coords_scale_init,
                    norm_feats=self.norm_feats,
                    initialization_gain=self.initialization_gain,
                ),
                "pocket_to_residue": EGNNLayer(
                    node_features=self.node_features,
                    edge_features=self.edge_features,
                    hidden_features=self.hidden_features,
                    out_features=self.out_features,
                    act=activation_mapping[self.act],
                    dropout=dropout_mapping[self.act](self.dropout),
                    node_aggr=aggregation_mapping[self.node_aggr],
                    coord_aggr=aggregation_mapping[self.coord_aggr],
                    residual=self.residual,
                    update_coords=self.update_coords,
                    norm_coords=self.norm_coords,
                    norm_coords_scale_init=self.norm_coords_scale_init,
                    norm_feats=self.norm_feats,
                    initialization_gain=self.initialization_gain,
                ),
            }
        )

         # Optional global protein node interactions
        if self.protein_node:
            
            module_dict["residue_to_protein"] = GNNLayer(
                node_features=self.node_features,
                hidden_features=self.hidden_features,
                out_features=self.out_features,
                act=activation_mapping[self.act],
                dropout=dropout_mapping[self.act](self.dropout),
                node_aggr=aggregation_mapping[self.node_aggr],
                residual=self.residual,
                norm_feats=self.norm_feats,
                initialization_gain=self.initialization_gain,
            )
            
            module_dict["protein_to_residue"] = GNNLayer(
                node_features=self.node_features,
                hidden_features=self.hidden_features,
                out_features=self.out_features,
                act=activation_mapping[self.act],
                dropout=dropout_mapping[self.act](self.dropout),
                node_aggr=aggregation_mapping[self.node_aggr],
                residual=self.residual,
                norm_feats=self.norm_feats,
                initialization_gain=self.initialization_gain,
            )
    
        return module_dict


    def initialize_layers(
        self
    ):
        """
        Initialize input projection, message passing layers, prediction head, and confidence head.
        """
        
        # Map raw input features into internal node embedding space
        self.input_mapping = nn.Linear(self.input_features, self.node_features)

        if self.weight_share:
            # Use a single layer that will be shared across all iterations
            self.shared_layer = self.create_vnegnn_layer()

        else:
            # Independent layer per message passing iteration
            self.layers = nn.ModuleList([self.create_vnegnn_layer() for _ in range(self.num_layers)])

        # Prediction head for residue segmentation
        self.head = nn.Sequential(
            nn.Linear(self.out_features, self.out_features),
            dropout_mapping[self.act](self.dropout),
            activation_mapping[self.act],
            nn.Linear(self.out_features, 1),
        )

        # Confidence prediction head
        self.confidence_mlp = nn.Sequential(
            nn.Linear(self.out_features, self.out_features),
            dropout_mapping[self.act](self.dropout),
            activation_mapping[self.act],
            nn.Linear(self.out_features, 1),
        )


    def forward(
        self, 
        batch, 
    ) -> tuple:
        """
        Forward pass of the VN-EGNN model.

        Parameters
        ----------
        batch : torch_geometric.data.HeteroData
            Batched heterogeneous graph containing with node types "residue", "pocket", and optionally "protein". Each node type should have:
                - Node features in `batch.x_dict`
                - Node coordinates in `batch.pos_dict`
                - Edge indices in `batch.edge_index_dict`

        Returns
        -------
        tuple
            (
                pocket_feats: torch.Tensor
                    Final pocket node embeddings.
                pocket_pos: torch.Tensor
                    Final pocket coordinates.
                protein_feats: torch.Tensor or None
                    Final protein node embeddings (if enabled).
                residue_feats: torch.Tensor
                    Final residue node embeddings.
                residue_pos: torch.Tensor
                    Final residue coordinates.
                residue_segm: torch.Tensor
                    Residue-level prediction logits (segmentation head).
                confidence: torch.Tensor
                    Confidence scores predicted from pocket embeddings.
                edge_index_residue_residue: torch.Tensor
                edge_index_residue_pocket: torch.Tensor
                edge_index_pocket_residue: torch.Tensor
            )
        """

        # Extract node features and coordinates from heterogeneous batch
        residue_feats = batch.x_dict["residue"]
        residue_pos = batch.pos_dict["residue"]

        pocket_feats = batch.x_dict["pocket"]
        pocket_pos = batch.pos_dict["pocket"]

        # Extract edge connectivity
        edge_index_residue_residue = batch.edge_index_dict[("residue", "to", "residue")]
        edge_index_residue_pocket = batch.edge_index_dict[("residue", "to", "pocket")]
        edge_index_pocket_residue = batch.edge_index_dict[("pocket", "to", "residue")]

        if self.protein_node:
            protein_feats = batch.x_dict["protein"]
            edge_index_residue_protein = batch.edge_index_dict[("residue", "to", "protein")]
            edge_index_protein_residue = batch.edge_index_dict[("protein", "to", "residue")]

        # Coordinate scaling
        if self.scaling_factor != 1:
            residue_pos = residue_pos / self.scaling_factor
            pocket_pos = pocket_pos / self.scaling_factor

        # Map input features into latent node feature space
        residue_feats = self.input_mapping(residue_feats)
        pocket_feats = self.input_mapping(pocket_feats)
        protein_feats = self.input_mapping(protein_feats)

        # Iterative message passing
        for i in range(self.num_layers):
            # Select shared layer (weight tying) or layer-specific module
            layer = self.shared_layer if self.weight_share else self.layers[i]

            # Residue ↔ Residue message passing (updates residue feats + coords)
            residue_feats, residue_pos = layer["residue_to_residue"](
                x=(residue_feats, residue_feats),
                edge_index=edge_index_residue_residue,
                pos=(residue_pos, residue_pos),
            )

            # Residue → Pocket message passing (updates pocket feats + coords)
            pocket_feats, pocket_pos = layer["residue_to_pocket"](
                x=(residue_feats, pocket_feats),
                edge_index=edge_index_residue_pocket,
                pos=(residue_pos, pocket_pos),
            )
            
            # Pocket → Residue message passing (updates residue feats + coords)
            residue_feats, residue_pos = layer["pocket_to_residue"](
                x=(pocket_feats, residue_feats),
                edge_index=edge_index_pocket_residue,
                pos=(pocket_pos, residue_pos),
            )

            # Optional global protein node interaction
            if self.protein_node:
                # Residue → Protein aggregation (updates protein feats)
                protein_feats = layer["residue_to_protein"](
                    x=(residue_feats, protein_feats),
                    edge_index=edge_index_residue_protein,
                )

                # Protein → Residue broadcast
                residue_feats = layer["protein_to_residue"](
                    x=(protein_feats, residue_feats),
                    edge_index=edge_index_protein_residue,
                )

            else: 
                protein_feats = None

         # Residue-level prediction (segmentation head)
        residue_segm = self.head(residue_feats)

        # Confidence prediction from pocket embeddings
        confidence = self.confidence_mlp(pocket_feats)

        # Restore original coordinate scale
        if self.scaling_factor != 1:
            residue_pos = residue_pos * self.scaling_factor
            pocket_pos = pocket_pos * self.scaling_factor

        return (
            pocket_feats, 
            pocket_pos, 
            protein_feats,
            residue_feats, 
            residue_pos, 
            residue_segm,
            confidence,
            edge_index_residue_residue,
            edge_index_residue_pocket,
            edge_index_pocket_residue,
        )