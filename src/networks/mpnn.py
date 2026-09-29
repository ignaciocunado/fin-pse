import os.path as osp

import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.graphgym import cfg
from torch_geometric.nn import GINEConv, BatchNorm, PNAConv, HeteroConv
import torch.nn.functional as F
import torch

from torch_geometric.graphgym.register import register_network, head_dict

from src.data.hetero import EDGE, REV_EDGE, node_store, edge_store
from src.networks.fin_pse import FinPSEEncoder


@register_network("MPNN")
class MPNN(torch.nn.Module):
    def __init__(self, dim_in=0, dim_out=0):
        super().__init__()
        self.n_hidden = cfg.gnn.dim_inner
        self.final_dropout = cfg.gnn.dropout
        self.reverse_mp = cfg.gnn.reverse_mp

        emb_dim = cfg.gnn.dim_inner // 2 if cfg.gnn.add_encodings else cfg.gnn.dim_inner
        node_dim = cfg.gnn.dim_in + (1 if cfg.gnn.ego else 0)
        edge_dim = cfg.gnn.edge_dim + (2 if cfg.dataset.ports else 0)

        self.node_emb = nn.Linear(node_dim, emb_dim)
        self.edge_emb = nn.Linear(edge_dim, emb_dim)
        if self.reverse_mp:
            self.edge_emb_rev = nn.Linear(edge_dim, emb_dim)

        if cfg.gnn.add_encodings:
            if cfg.gnn.encodings_ports and not cfg.dataset.ports:
                raise ValueError("gnn.encodings_ports requires dataset.ports")

            self.encodings = FinPSEEncoder(
                cfg.gnn.dim_in,
                cfg.gnn.encodings_edge_dim + (2 if cfg.gnn.encodings_ports else 0),
                cfg.gnn.encodings_dim,
                num_layers=cfg.gnn.encodings_layers if cfg.gnn.encodings_layers > 0 else None,
                reverse_mp=cfg.gnn.encodings_reverse_mp,
            )
            self.encodings.load_state_dict(
                torch.load(
                    osp.join(cfg.checkpoint_dir, cfg.gnn.encodings_file),
                    weights_only=True,
                    map_location=cfg.accelerator,
                )["model_state_dict"]
            )
            for param in self.encodings.parameters():
                param.requires_grad = False

            self.project_node_encodings = nn.Sequential(
                nn.Linear(cfg.gnn.encodings_dim, cfg.gnn.dim_inner),
                nn.ReLU(),
                nn.BatchNorm1d(cfg.gnn.dim_inner),
                nn.Linear(cfg.gnn.dim_inner, emb_dim),
            )
            self.project_edge_encodings = nn.Sequential(
                nn.Linear(cfg.gnn.encodings_dim, cfg.gnn.dim_inner),
                nn.ReLU(),
                nn.BatchNorm1d(cfg.gnn.dim_inner),
                nn.Linear(cfg.gnn.dim_inner, emb_dim),
            )

        self.gnn = GnnHelper(
            num_gnn_layers=cfg.gnn.layers_mp,
            n_hidden=cfg.gnn.dim_inner,
            edge_updates=cfg.gnn.emlps,
            final_dropout=cfg.gnn.dropout,
            deg=torch.tensor(cfg.gnn.pna_deg, dtype=torch.float) if cfg.gnn.layer_type == "pna" else None,
            reverse_mp=self.reverse_mp,
        )

        self.head = head_dict[cfg.gnn.head](cfg.gnn.dim_inner, cfg.gnn.dim_out)

    def forward(self, data):
        if isinstance(data, HeteroData):
            x_raw, edge_index, edge_attr_raw = data["node"].x, data.edge_index_dict, data.edge_attr_dict
        else:
            x_raw, edge_index, edge_attr_raw = data.x, {EDGE: data.edge_index}, {EDGE: data.edge_attr}

        # Initial Embedding Layers
        x = self.node_emb(x_raw)
        edge_attr = {EDGE: self.edge_emb(edge_attr_raw[EDGE])}
        if self.reverse_mp:
            edge_attr[REV_EDGE] = self.edge_emb_rev(edge_attr_raw[REV_EDGE])

        if cfg.gnn.add_encodings:
            node_encodings, edge_encodings = self.compute_encodings(x_raw, edge_index, edge_attr_raw)
            x = torch.cat([x, self.project_node_encodings(node_encodings)], dim=1)
            edge_attr = {
                edge_type: torch.cat([attr, self.project_edge_encodings(edge_encodings[edge_type])], dim=1)
                for edge_type, attr in edge_attr.items()
            }

        # Message Passing Layers
        if self.reverse_mp:
            x, edge_attr = self.gnn(x, edge_index, edge_attr)
            edge_attr = edge_attr[EDGE]
        else:
            x, edge_attr = self.gnn(x, edge_index[EDGE], edge_attr[EDGE])

        node_store(data).x = x
        edge_store(data).edge_attr = edge_attr

        return self.head(data)

    def compute_encodings(self, x, edge_index, edge_attr):
        """Runs the frozen pretrained encoder.

        Args:
            x: Raw node features
            edge_index: Dictionary with the edge index of every edge type
            edge_attr: Dictionary with the raw edge features of every edge type

        Returns:
            Node encodings and a dictionary with the encodings of every edge type in edge_attr
        """
        x = x[:, : cfg.gnn.dim_in]  # Drop the ego IDs, the encoder is pretrained without them
        if cfg.gnn.encodings_random_feats:
            x = torch.randn(x.shape, device=x.device)
        edge_attr = {edge_type: self.encoder_edge_features(attr) for edge_type, attr in edge_attr.items()}

        with torch.no_grad():
            if cfg.gnn.encodings_reverse_mp:
                return self.encodings.encode(x, edge_index, edge_attr)

            node_encodings, edge_encodings = self.encodings.encode(x, edge_index[EDGE], edge_attr[EDGE])
            # The homogeneous encoder embeds the raw features of each edge, so reverse edges get the same embedding
            edge_encodings = {EDGE: edge_encodings} | {
                edge_type: self.encodings.edge_emb(attr) for edge_type, attr in edge_attr.items() if edge_type != EDGE
            }

        return node_encodings, edge_encodings

    @staticmethod
    def encoder_edge_features(edge_attr):
        """Edge features the encoder was pretrained on: the leading ones (Timestamp, Amount) and optionally the ports"""
        features = edge_attr[:, : cfg.gnn.encodings_edge_dim]
        if cfg.gnn.encodings_ports:
            features = torch.cat([features, edge_attr[:, -2:]], dim=1)
        return features


class GnnHelper(torch.nn.Module):
    def __init__(self, num_gnn_layers, n_hidden=100, edge_updates=False, final_dropout=0.5, deg=None, reverse_mp=False):
        super().__init__()

        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.final_dropout = final_dropout
        self.edge_updates = edge_updates
        self.reverse_mp = reverse_mp

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            conv = self._make_conv(deg)
            if self.reverse_mp:
                # Separate weights per edge direction, averaged per node like to_hetero(..., aggr="mean") in MEGA-GNN
                conv = HeteroConv({EDGE: conv, REV_EDGE: self._make_conv(deg)}, aggr="mean")

            if self.edge_updates:
                emlp = self._make_emlp()
                if self.reverse_mp:
                    emlp = nn.ModuleDict({"__".join(EDGE): emlp, "__".join(REV_EDGE): self._make_emlp()})
                self.emlps.append(emlp)

            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

    def _make_conv(self, deg):
        if cfg.gnn.layer_type == "gin":
            return GINEConv(
                nn.Sequential(
                    nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(), nn.Linear(self.n_hidden, self.n_hidden)
                ),
                edge_dim=self.n_hidden,
            )
        elif cfg.gnn.layer_type == "pna":
            aggregators = ["mean", "min", "max", "std"]
            scalers = ["identity", "amplification", "attenuation"]
            return PNAConv(
                in_channels=self.n_hidden,
                out_channels=self.n_hidden,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                edge_dim=self.n_hidden,
                towers=5,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
        raise ValueError(f"Unknown gnn.layer_type {cfg.gnn.layer_type}")

    def _make_emlp(self):
        return nn.Sequential(
            nn.Linear(3 * self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
        )

    @staticmethod
    def _update_edges(emlp, x, edge_index, edge_attr):
        src, dst = edge_index
        return edge_attr + emlp(torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2

    def forward(self, x, edge_index, edge_attr):
        """With reverse message passing, edge_index and edge_attr are dictionaries keyed by edge type"""
        for i in range(self.num_gnn_layers):
            if self.reverse_mp:
                n_new = self.convs[i]({"node": x}, edge_index, edge_attr_dict=edge_attr)["node"]
            else:
                n_new = self.convs[i](x, edge_index, edge_attr)
            x = (x + F.relu(self.batch_norms[i](n_new))) / 2

            if self.edge_updates and self.reverse_mp:
                edge_attr = {
                    edge_type: self._update_edges(self.emlps[i]["__".join(edge_type)], x, edge_index[edge_type], attr)
                    for edge_type, attr in edge_attr.items()
                }
            elif self.edge_updates:
                edge_attr = self._update_edges(self.emlps[i], x, edge_index, edge_attr)

        return x, edge_attr
