from torch_geometric.data import Data
from torch_geometric.typing import OptTensor


class GraphData(Data):
    """This is the homogenous graph object we use for GNN training if reverse MP is not enabled"""

    def __init__(
            self,
            x: OptTensor = None,
            edge_index: OptTensor = None,
            edge_attr: OptTensor = None,
            y: OptTensor = None,
            pos: OptTensor = None,
            readout: str = "edge",
            num_nodes: int = None,
            timestamps: OptTensor = None,
            node_timestamps: OptTensor = None,
            **kwargs,
    ):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        self.readout = readout
        if num_nodes is not None:
            self.num_nodes = int(num_nodes)
        elif self.x is not None:
            self.num_nodes = int(self.x.size(0))
            # else: leave num_nodes unset; PyG can infer later in many cases

        if timestamps is not None:
            self.timestamps = timestamps
        elif self.edge_attr is not None and self.edge_attr.size(-1) > 0:
            self.timestamps = self.edge_attr[:, 0].clone()
        else:
            self.timestamps = None