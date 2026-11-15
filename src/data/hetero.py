import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.graphgym import cfg
from torch_geometric.transforms import BaseTransform

# Edge types used for reverse message passing, following MEGA-GNN / Multi-GNN
EDGE = ("node", "to", "node")
REV_EDGE = ("node", "rev_to", "node")


def uses_hetero_data() -> bool:
    """Whether the graphs must be heterogeneous, i.e. whether any part of the model uses reverse message passing.

    The reverse edges are also needed when only the frozen encoder uses them, as the neighbour loaders only sample
    along the edge types in the graph (a homogeneous loader never samples the outgoing edges of a node).
    """
    return cfg.gnn.reverse_mp or (cfg.gnn.add_encodings and cfg.gnn.encodings_reverse_mp)


def swap_port_columns(edge_attr: torch.Tensor) -> torch.Tensor:
    """The in-port of an edge is the out-port of its reverse edge and vice versa (ports are the last two columns)"""
    edge_attr = edge_attr.clone()
    edge_attr[:, [-2, -1]] = edge_attr[:, [-1, -2]]
    return edge_attr


def to_hetero_data(data: Data, target: str = "edge", ports: bool = None) -> HeteroData:
    """Turns a homogeneous graph into a heterogeneous graph with forward and reverse edges.

    Args:
        data: Homogeneous graph
        target: "edge" if the labels are edge labels (AML), "node" if they are node labels (ETH, SSL)
        ports: Whether the last two edge features are port numbers (default: dataset.ports)
    """
    ports = cfg.dataset.ports if ports is None else ports

    hetero = HeteroData()
    hetero["node"].x = data.x
    hetero["node"].num_nodes = data.num_nodes

    hetero[EDGE].edge_index = data.edge_index
    hetero[EDGE].edge_attr = data.edge_attr
    hetero[REV_EDGE].edge_index = data.edge_index.flipud()
    hetero[REV_EDGE].edge_attr = swap_port_columns(data.edge_attr) if ports else data.edge_attr

    timestamps = getattr(data, "timestamps", None)
    if timestamps is not None:
        hetero[EDGE].timestamps = timestamps
        hetero[REV_EDGE].timestamps = timestamps

    if target == "edge":
        hetero[EDGE].y = data.y
    else:
        hetero["node"].y = data.y

    for key in ("train_mask", "val_mask", "test_mask", "node_encodings"):
        if key in data:
            hetero["node"][key] = data[key]

    return hetero


def node_store(data):
    """Node attributes of a homogeneous or heterogeneous graph"""
    return data["node"] if isinstance(data, HeteroData) else data


def edge_store(data):
    """Attributes of the (forward) transaction edges of a homogeneous or heterogeneous graph"""
    return data[EDGE] if isinstance(data, HeteroData) else data


def homogeneous_view(data):
    """Node features, edge index and edge features of the forward edges"""
    edges = edge_store(data)
    return node_store(data).x, edges.edge_index, edges.edge_attr


def strip_edge_ids(data):
    """Removes the unique edge ids added by add_arange_ids from all edge types"""
    for store in data.edge_stores:
        store.edge_attr = store.edge_attr[:, 1:]


class AddEgoIds(BaseTransform):
    """Flags the seed nodes of a batch with an extra node feature (ego IDs from Multi-GNN / MEGA-GNN).

    The seed nodes are the endpoints of the seed edges for link loaders and the first batch_size nodes for node loaders.
    """

    def forward(self, data):
        nodes, edges = node_store(data), edge_store(data)

        ids = torch.zeros((nodes.x.shape[0], 1), device=nodes.x.device)
        if "edge_label_index" in edges:
            ids[edges.edge_label_index.reshape(-1)] = 1
        else:
            ids[: nodes.batch_size] = 1

        nodes.x = torch.cat([nodes.x, ids], dim=1)
        return data
