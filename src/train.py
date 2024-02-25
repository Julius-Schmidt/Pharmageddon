import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from .graph.graph_functions import build_graph, get_subgraph, k_hop_graph
from .graph.hypernode_loader import HyperNodeLoader
from .model.model import PHARMAGEDDON
from .model.neg_sampler import random_neg_sample


class Train:
    def __init__(
        self,
        train_path: Path,
        test_path: Path,
        out: Path,
        config_path: Path = None,
        checkpoint_path: Path = None,
    ) -> None:
        self.out = out
        os.makedirs(out / "roc", exist_ok=True)
        os.makedirs(out / "model", exist_ok=True)

        if config_path is None:
            config_path = Path(os.path.dirname(__file__)) / "config" / "train.conf"

        with open(config_path, "r", encoding='UTF-8') as f:
            self.train_config = yaml.safe_load(f)

        print("Building train graph...")
        self.train_graph = build_graph(train_path)

        print("Building test graph...")
        self.test_graph = build_graph(test_path)

        with open(self.out / "graph.pk", "wb") as f:
            torch.save(self.train_graph, f)

        print("Creating train loader ...")
        self.train_loader = HyperNodeLoader(
            self.train_graph,
            "drugs",
            "hypernodes",
            "hyperedge_part_2",
            self.train_config["n_hops"],
            self.train_config["batch_size"],
            self.train_config["shuffle"],
        )

        print("Creating test loader ...")
        self.test_loader = HyperNodeLoader(
            self.test_graph,
            "drugs",
            "hypernodes",
            "hyperedge_part_2",
            self.train_config["n_hops"],
            self.train_config["batch_size"],
            self.train_config["shuffle"],
        )

        if checkpoint_path != None:
            self.model = torch.load(checkpoint_path)
        else:
            self.model = PHARMAGEDDON(
                self.train_config["embedding_dim"],
                self.train_graph["hypernodes"].x.shape[1],
                self.train_config["n_hops"],
            )

        self.pharmageddon_optimiser = getattr(torch.optim, self.train_config["optim"])(
            self.model.parameters(), lr=self.train_config["lr"]
        )

        self.train_effects, _ = torch.hstack(
            [t.flatten() for t in self.train_graph["hypernodes"].effect_ids]
        ).sort()
        self.test_effects, _ = (
            self.train_graph["hypernodes"].x.coalesce().indices()[1].sort()
        )

    def train(self):
        for epoch in range(self.train_config["epochs"]):
            i = 0
            for pos_mask, pos_nodes, pos_effect_ids, hypernode_ids in tqdm(
                self.train_loader, desc=f"Training (Epoch {epoch}): "
            ):
                train_loss_pharmageddon = 0
                train_preds = torch.tensor([])
                train_labels = torch.tensor([])

                i += 1
                self.model.train()
                self.pharmageddon_optimiser.zero_grad()

                loss_pharmageddon, preds, labels = self.step(
                    self.train_graph,
                    pos_mask,
                    pos_nodes,
                    pos_effect_ids,
                    hypernode_ids,
                    self.train_effects,
                )

                train_loss_pharmageddon += loss_pharmageddon.detach()
                train_preds = torch.cat((train_preds, preds.detach()))
                train_labels = torch.cat((train_labels, labels.detach()))

                if i % 100 == 0:  # self.train_config["save_steps"]
                    self.model.eval()

                    sum_test_loss_pharmageddon = 0
                    test_preds = torch.tensor([])
                    test_labels = torch.tensor([])
                    with torch.no_grad():
                        for (
                            pos_mask,
                            pos_nodes,
                            pos_effect_ids,
                            hypernode_ids,
                        ) in self.test_loader:
                            loss_pharmageddon, preds, labels = self.step(
                                self.test_graph,
                                pos_mask,
                                pos_nodes,
                                pos_effect_ids,
                                hypernode_ids,
                                self.test_effects,
                            )

                            sum_test_loss_pharmageddon += loss_pharmageddon
                            test_preds = torch.cat((test_preds, preds))
                            test_labels = torch.cat((test_labels, labels))

                    self.evaluate(
                        epoch,
                        i,
                        train_loss_pharmageddon,
                        train_preds,
                        train_labels,
                        sum_test_loss_pharmageddon,
                        test_preds,
                        test_labels,
                    )
                    self.save(epoch, i)

    def step(
        self,
        graph,
        pos_mask,
        pos_nodes,
        pos_effect_ids,
        hypernode_ids,
        graph_effect_ids,
    ):
        neg_nodes, neg_effect_ids = random_neg_sample(
            graph,
            self.train_config["batch_size"],
            len(graph["hypernodes"].x[0]),
            graph_effect_ids,
        )

        neg_mask = k_hop_graph(
            self.train_config["n_hops"],
            "drugs",
            torch.cat(neg_nodes).unique(),
            graph.edge_index_dict,
            graph,
            self.train_config["drop_fraction"],
        )

        mask = {
            key: torch.cat(
                (
                    pos_mask.get(key, torch.tensor([], dtype=torch.int64)),
                    neg_mask.get(key, torch.tensor([], dtype=torch.int64)),
                )
            ).unique()
            for key in set(pos_mask.keys()) | set(neg_mask.keys())
        }

        # Remove hypernodes that should be predicted
        mask["hypernodes"] = mask["hypernodes"][
            ~torch.isin(mask["hypernodes"], hypernode_ids)
        ]

        pos_nodes = [
            torch.tensor([(mask["drugs"] == y).nonzero().flatten() for y in x])
            for x in pos_nodes
        ]
        neg_nodes = [
            torch.tensor([(mask["drugs"] == y).nonzero().flatten() for y in x])
            for x in neg_nodes
        ]

        x = graph["hypernodes"].x.coalesce()
        graph["hypernodes"].x = torch.ones(x.shape[0]).reshape(x.shape[0], -1)
        effect_ids = graph["hypernodes"]["effect_ids"]
        graph["hypernodes"]["effect_ids"] = None
        subgraph = graph.subgraph(mask)
        graph["hypernodes"]["effect_ids"] = effect_ids

        graph["hypernodes"].x = x
        mask = torch.isin(x.indices()[0], mask["hypernodes"])
        mask = torch.concat((mask, mask)).view(2, -1)
        indices = x.indices()[mask].view(2, -1)
        unique_values, inverse_indices = torch.unique(indices[0], return_inverse=True)
        indices[0] = torch.arange(len(unique_values))[inverse_indices]
        subgraph["hypernodes"].x = torch.sparse_coo_tensor(
            indices, torch.ones(indices.shape[1]), x.shape
        )

        pos_preds = self.model(subgraph, pos_nodes, pos_effect_ids)
        neg_preds = self.model(subgraph, neg_nodes, neg_effect_ids)
        preds = torch.cat((pos_preds, neg_preds))

        labels = torch.cat((torch.ones(len(pos_nodes)), torch.zeros(len(neg_nodes))))

        loss_pharmageddon = torch.nn.functional.binary_cross_entropy(preds.squeeze(), labels)

        if self.model.training:
            loss_pharmageddon.backward()
            self.pharmageddon_optimiser.step()

        return loss_pharmageddon.detach(), preds.detach(), labels.detach()

    def evaluate(
        self,
        epoch,
        batch,
        train_loss,
        train_preds,
        train_labels,
        test_loss,
        test_preds,
        test_labels,
    ):
        fpr, tpr, _ = roc_curve(test_labels, test_preds)
        plt.plot(fpr, tpr, label="test")
        fpr, tpr, _ = roc_curve(train_labels, train_preds)
        plt.plot(fpr, tpr, label="train")

        plt.legend(loc="lower right")

        plt.savefig(self.out / "roc" / f"{epoch}-{batch}.png")
        plt.clf()

        tqdm.write(
            f"Epoch {epoch} - {batch} | Train Loss: {round(float(train_loss)/len(self.train_loader), 3)}, Train AUC: {round(roc_auc_score(train_labels, train_preds),3)} | Test Loss: {round(float(test_loss)/len(self.test_loader), 3)}, Test AUC: {round(roc_auc_score(test_labels, test_preds),3)}",
            end="\n",
        )

    def save(self, epoch, batch):
        torch.save(self.model, self.out / "model" / f"model_{epoch}_{batch}.pt")
