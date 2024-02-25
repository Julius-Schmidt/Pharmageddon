import os
import torch
import copy
import torch_geometric
import pandas as pd
import polars as pl
import numpy as np
from collections import defaultdict
from scipy.sparse import coo_matrix
from torch_geometric.data import HeteroData
from torch_geometric.utils import bipartite_subgraph


def build_graph(path: str, similarity_threshold=0.05) -> HeteroData:
    SIDE_EFFECT_PATH = os.path.join(path, "side_effects.csv")
    DRUGS_PATH = os.path.join(path, "drugs.csv")
    POLYPHARMACY_PATH = os.path.join(path, "polypharmacy.csv")

    def name_to_index(s, n):
        s = pd.DataFrame({"name": s.label})
        s["id"] = s.index
        n = pd.DataFrame({"name": n})
        return pd.merge(n, s, on="name", how="left")["id"]

    side_effects = pd.read_csv(SIDE_EFFECT_PATH, skiprows=1, header=None)
    drugs = pd.read_csv(DRUGS_PATH, skiprows=1, header=None)
    drugs["features"] = drugs[drugs.columns[2:]].values.tolist()
    drugs = drugs[[0, "features"]]

    graph = HeteroData(
        {
            "drugs": {
                "x": torch.tensor(drugs["features"]).to(torch.float32),
                "label": drugs[0],
            },  # .to(torch.float64)
        }
    )

    graph["effects"] = side_effects[0]

    graph["drugs"].num_nodes = len(drugs)
    polypharmacy = pl.read_csv(POLYPHARMACY_PATH).to_pandas()
    polypharmacy_drugs = polypharmacy.iloc[:, 0].str.split("|", expand=True)
    polypharmacy.columns = ["drug_combination", "side_effect"]
    max_side_effect = side_effects[1].max()

    for i in range(len(polypharmacy_drugs.columns)):
        polypharmacy_drugs[i] = name_to_index(graph["drugs"], polypharmacy_drugs[i])

    polypharmacy_drugs_np = polypharmacy_drugs.to_numpy()
    polypharmacy[0] = [row[~np.isnan(row)].tolist() for row in polypharmacy_drugs_np]
    polypharmacy["drug_combination"] = [
        "|".join(map(str, sorted(filter(pd.notna, row))))
        for row in polypharmacy_drugs.values
    ]

    polypharmacy_grouped = (
        polypharmacy.groupby("drug_combination")["side_effect"]
        .apply(list)
        .reset_index()
    )
    polypharmacy_grouped["side_effect"] = np.sort(
        polypharmacy_grouped["side_effect"].to_numpy()
    )

    rows = []
    cols = []
    for i, side_effects in enumerate(polypharmacy_grouped["side_effect"]):
        rows.extend([i] * len(side_effects))
        cols.extend(side_effects)

    data = np.ones(len(rows))
    side_effect_matrix = coo_matrix(
        (data, (rows, cols)), shape=(len(polypharmacy_grouped), max_side_effect + 1)
    )

    i = torch.LongTensor(np.vstack((side_effect_matrix.row, side_effect_matrix.col)))
    v = torch.FloatTensor(side_effect_matrix.data)
    shape = torch.Size(side_effect_matrix.shape)

    side_effect_tensor = torch.sparse.FloatTensor(i, v, shape)

    graph["hypernodes"].x = side_effect_tensor.to(torch.float32)
    graph["hypernodes"].effect_ids = [
        torch.tensor(item) for item in polypharmacy_grouped["side_effect"]
    ]

    polypharmacy_grouped["drug_combination"] = polypharmacy_grouped[
        "drug_combination"
    ].str.split("|")

    graph["drugs", "hyperedge_part_1", "hypernodes"].edge_index = (
        torch.cat(
            (
                torch.from_numpy(
                    np.array(
                        polypharmacy_grouped["drug_combination"]
                        .explode()
                        .astype(float)
                        .astype(np.int64),
                        dtype=np.int64,
                    )
                ),
                torch.tensor(polypharmacy_grouped["drug_combination"].explode().index),
            )
        )
        .reshape(-1, len(polypharmacy_grouped["drug_combination"].explode()))
        .to(torch.int64)
    )

    graph["hypernodes", "hyperedge_part_2", "drugs"].edge_index = (
        torch.cat(
            (
                torch.tensor(polypharmacy_grouped["drug_combination"].explode().index),
                torch.from_numpy(
                    np.array(
                        polypharmacy_grouped["drug_combination"]
                        .explode()
                        .astype(float)
                        .astype(np.int64),
                        dtype=np.int64,
                    )
                ),
            )
        )
        .reshape(-1, len(polypharmacy_grouped["drug_combination"].explode()))
        .to(torch.int64)
    )

    drug_degree = torch_geometric.utils.degree(
        graph["hypernodes", "hyperedge_part_2", "drugs"].edge_index[1],
        num_nodes=graph["drugs"].num_nodes,
    )
    drug_degree = torch.repeat_interleave(drug_degree, len(drug_degree))

    flattened_tensors = graph["drugs"].x.view(graph["drugs"].x.size(0), -1)
    flattened_tensors = flattened_tensors.unsqueeze(1)
    differences = flattened_tensors - flattened_tensors.permute(1, 0, 2)
    euclidean_distances = torch.norm(differences, dim=2)
    similarity = 1 / (1 + euclidean_distances)
    drug_edges = torch_geometric.utils.to_undirected(
        torch.combinations(torch.arange(graph["drugs"].num_nodes), 2, True).T
    )
    drug_edges, edge_weights = torch_geometric.utils.remove_self_loops(
        drug_edges, similarity.view(-1) * (1 / (1 + drug_degree))
    )

    mask = edge_weights >= similarity_threshold
    drug_edges = drug_edges[:, mask]
    edge_weights = edge_weights[mask]

    graph["drugs", "chemical_similarity", "drugs"].edge_index = drug_edges
    graph["drugs", "chemical_similarity", "drugs"].edge_weight = edge_weights

    return graph


def k_hop_graph(k, node_type, node_idx, edge_index_dict, graph, drop_fraction=0):
    mask = {}
    mask = defaultdict(lambda: torch.tensor([], dtype=torch.int64))
    mask[node_type] = node_idx

    all_new_nodes = defaultdict(lambda: torch.tensor([], dtype=torch.int64))
    for _ in range(k):
        for edge_type in edge_index_dict:
            f = edge_type[0]
            t = edge_type[2]
            edge = edge_index_dict[edge_type]
            from_nodes = mask[f]
            if len(from_nodes) == 0:
                continue

            indices = torch.tensor(
                np.where(
                    pd.Index(pd.unique(from_nodes.numpy())).get_indexer(edge[0].numpy())
                    >= 0
                )[0]
            )
            indices = (
                indices[torch.randperm(indices.shape[0])]
                .narrow(0, 0, int(indices.shape[0] * (1 - drop_fraction)))
                .sort()
                .values
            )

            new_nodes = edge[1][indices].unique()
            all_new_nodes[t] = torch.concat((all_new_nodes[t], new_nodes)).unique()

            if t == "hypernodes":
                from_nodes = new_nodes
                for edge_type in edge_index_dict:
                    if edge_type[0] != "hypernodes" or len(from_nodes) == 0:
                        continue

                    f = "hypernodes"
                    t = edge_type[2]
                    edge = edge_index_dict[edge_type]
                    indices = torch.tensor(
                        np.where(
                            pd.Index(pd.unique(from_nodes.numpy())).get_indexer(
                                edge[0].numpy()
                            )
                            >= 0
                        )[0]
                    )

                    new_nodes = edge[1][indices].unique()

                    all_new_nodes[t] = torch.concat(
                        (all_new_nodes[t], new_nodes)
                    ).unique()

        for t in all_new_nodes:
            mask[t] = torch.concat((mask[t], all_new_nodes[t])).unique()

    return mask


def get_subgraph(graph, subset_dict) -> "HeteroData":
    r"""Returns the induced subgraph containing the node types and
    corresponding nodes in :obj:`subset_dict`.

    If a node type is not a key in :obj:`subset_dict` then all nodes of
    that type remain in the graph.

    .. code-block:: python

        data = HeteroData()
        data['paper'].x = ...
        data['author'].x = ...
        data['conference'].x = ...
        data['paper', 'cites', 'paper'].edge_index = ...
        data['author', 'paper'].edge_index = ...
        data['paper', 'conference'].edge_index = ...
        print(data)
        HeteroData(
            paper={ x=[10, 16] },
            author={ x=[5, 32] },
            conference={ x=[5, 8] },
            (paper, cites, paper)={ edge_index=[2, 50] },
            (author, to, paper)={ edge_index=[2, 30] },
            (paper, to, conference)={ edge_index=[2, 25] }
        )

        subset_dict = {
            'paper': torch.tensor([3, 4, 5, 6]),
            'author': torch.tensor([0, 2]),
        }

        print(data.subgraph(subset_dict))
        HeteroData(
            paper={ x=[4, 16] },
            author={ x=[2, 32] },
            conference={ x=[5, 8] },
            (paper, cites, paper)={ edge_index=[2, 24] },
            (author, to, paper)={ edge_index=[2, 5] },
            (paper, to, conference)={ edge_index=[2, 10] }
        )

    Args:
        subset_dict (Dict[str, LongTensor or BoolTensor]): A dictionary
            holding the nodes to keep for each node type.
    """
    data = copy.copy(graph)
    subset_dict = copy.copy(subset_dict)

    for node_type, subset in subset_dict.items():
        for key, value in graph[node_type].items():
            if key == "num_nodes":
                if subset.dtype == torch.bool:
                    data[node_type].num_nodes = int(subset.sum())
                else:
                    data[node_type].num_nodes = subset.size(0)
            elif graph[node_type].is_node_attr(key):
                if type(data[node_type][key]) == torch.Tensor:
                    dense = data[node_type][key].to_dense()
                    dense = dense[subset]
                    sparse = coo_matrix(dense)
                    i = torch.LongTensor(np.vstack((sparse.row, sparse.col)))
                    v = torch.FloatTensor(sparse.data)
                    shape = torch.Size(sparse.shape)
                    data[node_type][key] = torch.sparse.FloatTensor(i, v, shape)
                else:
                    data[node_type][key] = [value[i] for i in subset]
            else:
                data[node_type][key] = value

    for edge_type in graph.edge_types:
        src, _, dst = edge_type

        src_subset = subset_dict.get(src)
        if src_subset is None:
            src_subset = torch.arange(data[src].num_nodes)
        dst_subset = subset_dict.get(dst)
        if dst_subset is None:
            dst_subset = torch.arange(data[dst].num_nodes)

        edge_index, _, edge_mask = bipartite_subgraph(
            (src_subset, dst_subset),
            graph[edge_type].edge_index,
            relabel_nodes=True,
            size=(graph[src].num_nodes, graph[dst].num_nodes),
            return_edge_mask=True,
        )

        for key, value in graph[edge_type].items():
            if key == "edge_index":
                data[edge_type].edge_index = edge_index
            elif graph[edge_type].is_edge_attr(key):
                data[edge_type][key] = value[edge_mask]
            else:
                data[edge_type][key] = value

    return data
