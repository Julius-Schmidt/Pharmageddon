import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphLayer, SAGEConvWeight


class Encoder(nn.Module):
    def __init__(self, emb_dim, n_effects, n_hops):
        super(Encoder, self).__init__()
        self.effect_emb = nn.Embedding(n_effects, emb_dim)
        nn.init.xavier_uniform_(self.effect_emb.weight.data)

        self.layers = nn.ModuleList(
            GraphLayer(SAGEConvWeight, emb_dim) for _ in range(n_hops)
        )

        self.act = F.relu

    def forward(self, graph_x, edge_index, x_nodes, effect_ids, chemical_similarity):
        for layer in self.layers:
            graph_x = layer(graph_x, edge_index, chemical_similarity)
            graph_x = {key: self.act(x) for key, x in graph_x.items()}

        embs = [
            graph_x["drugs"][nodes] + self.effect_emb(effect)
            for effect, nodes in zip(effect_ids, x_nodes)
        ]
        return embs


class Decoder(nn.Module):
    def __init__(self, emb_dim):
        super(Decoder, self).__init__()
        self.Sigmoid = nn.Sigmoid()
        self.Linear = nn.Linear(emb_dim, 1)

    def forward(self, embs):
        embs = torch.stack([torch.prod(x, dim=0) for x in embs])
        return self.Sigmoid(self.Linear(embs))


class PHARMAGEDDON(nn.Module):
    def __init__(self, emb_dim, n_effects, n_hops):
        super(PHARMAGEDDON, self).__init__()
        self.encoder = Encoder(emb_dim, n_effects, n_hops)
        self.decoder = Decoder(emb_dim)

    def forward(self, graph, x_nodes, effect_ids):
        graph_x = graph.x_dict
        edge_index = graph.edge_index_dict
        return self.decoder(
            self.encoder(
                graph_x,
                edge_index,
                x_nodes,
                effect_ids,
                graph["drugs", "chemical_similarity", "drugs"].edge_weight,
            )
        )
