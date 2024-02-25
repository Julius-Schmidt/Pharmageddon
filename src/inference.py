import os
import torch
from pathlib import Path
from .model.model import PHARMAGEDDON


class Inference:
    def __init__(self, checkpoint_path, graph_path) -> None:
        if graph_path is None:
            graph_path = Path(os.path.dirname(__file__)) / "data" / "graph.pk"

        if checkpoint_path is None:
            checkpoint_path = Path(os.path.dirname(__file__)) / "data" / "checkpoint.pk"

        with open(graph_path, "rb") as f:
            self.graph = torch.load(f)

        self.model = torch.load(checkpoint_path)

    def predict(self, drugs, effects=None):
        if len(effects) == 0:
            effects = self.graph["effects"].values.tolist()
            effect_ids = torch.tensor(self.graph["effects"].index)
        else:
            effect_ids = torch.tensor(
                self.graph["effects"][self.graph["effects"].isin(effects)].index
            )

        drug_ids = torch.tensor(
            self.graph["drugs"]["label"][self.graph["drugs"]["label"].isin(drugs)].index
        )
        drug_ids = [drug_ids for _ in range(len(effect_ids))]
        with torch.no_grad():
            res = self.model(self.graph, drug_ids, effect_ids)
            print("Drugs\tEffect\tProbability")
            for r, e in zip(res, effects):
                print(f"{' '.join(drugs)}\t{e}\t{round(float(r), 4)}")
