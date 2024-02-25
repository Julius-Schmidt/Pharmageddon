import torch
from itertools import chain
from .graph_functions import k_hop_graph


class HyperNodeLoader:
    def __init__(
        self,
        graph,
        node_class,
        hypernode_class,
        hyperedge_class,
        n_hops,
        batch_size,
        shuffle=True,
        drop_fraction=0,
    ):
        self.graph = graph
        self.node_class = node_class
        self.hypernode_class = hypernode_class
        self.hyperedge_class = hyperedge_class
        self.batch_size = batch_size
        self.batch_size = batch_size
        self.n_hops = n_hops
        self.shuffle = shuffle
        self.drop_fraction = drop_fraction

        self.all_effect_ids = self.graph["hypernodes"].effect_ids
        self.node_id_list = torch.tensor(
            list(
                chain.from_iterable(
                    [[x] * len(y) for x, y in enumerate(self.all_effect_ids)]
                )
            )
        )
        self.all_effect_ids = torch.tensor(
            list(chain.from_iterable(self.all_effect_ids))
        )

        self.total = len(self.node_id_list)

        idx = torch.arange(0, self.total)
        if self.shuffle:
            indexes = torch.randperm(idx.shape[0])
            idx = idx[indexes]

        self.idx = torch.split(idx, self.batch_size)

    def __iter__(self):
        idx_iter = iter(range(len(self.idx)))
        for i in idx_iter:
            effect_ids = self.all_effect_ids[self.idx[i]]
            x_nodes = self.get_x_nodes(
                self.node_id_list[self.idx[i]],
                (self.hypernode_class, self.hyperedge_class, self.node_class),
                self.graph.edge_index_dict,
            )
            x_nodes_flat = torch.unique(torch.cat(x_nodes))
            mask = k_hop_graph(
                self.n_hops,
                self.node_class,
                x_nodes_flat,
                self.graph.edge_index_dict,
                self.graph,
                self.drop_fraction,
            )
            yield mask, x_nodes, effect_ids, self.node_id_list[self.idx[i]]

    def __len__(self):
        return len(self.idx)

    def get_x_nodes(self, hypernode_idx, edge_type, edge_index_dict):
        edges = edge_index_dict[edge_type]
        expanded_edge_origin = torch.Tensor.expand(
            edges[0], (len(hypernode_idx), len(edges[0]))
        )
        expanded_hypernodes = torch.Tensor.expand(
            hypernode_idx, (len(edges[0]), len(hypernode_idx))
        ).T
        eqs = expanded_edge_origin == expanded_hypernodes
        return [edges[1][eqs[i]] for i in range(len(eqs))]
