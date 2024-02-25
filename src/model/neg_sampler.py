import numpy as np
import torch


def hyperedge_size_distribution(graph):
    out_edges = graph.edge_index_dict[("hypernodes", "hyperedge_part_2", "drugs")][0]
    counts = torch.bincount(torch.bincount(out_edges))
    fractions = counts / sum(counts)
    distribution = {i: float(fractions[i]) for i in range(len(fractions))}
    return distribution


def random_neg_sample(graph, batch_size, n_effects, graph_effect_ids):
    # TODO: Maybe replace with a mixture of SNS, CNS and MNS (https://github.com/HyunjinHwn/SIGIR22-AHP/blob/main/sampler.py)
    distribution = hyperedge_size_distribution(graph)
    num_nodes = np.random.choice(
        list(distribution.keys()), p=list(distribution.values())
    )
    nodes = [
        torch.tensor(
            np.random.choice(len(graph["drugs"].x), size=num_nodes, replace=False)
        )
        for _ in range(batch_size)
    ]

    effects = graph_effect_ids[
        np.random.choice(len(graph_effect_ids), batch_size, replace=True)
    ]

    # effects = [torch.tensor(np.random.choice(n_effects)) for _ in range(batch_size)]

    return nodes, effects
