import argparse
from pathlib import Path
from src.train import Train
from src.inference import Inference


def main():
    parser = argparse.ArgumentParser(description="Train or predict using a model.")

    subparsers = parser.add_subparsers(dest="mode")

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--train", required=True, type=Path, help="Path to training data"
    )
    train_parser.add_argument(
        "--test", required=True, type=Path, help="Path to testing data"
    )
    train_parser.add_argument(
        "--out", required=True, type=Path, help="Path to output directory"
    )
    train_parser.add_argument("--config", type=Path, help="Path to configuration file")
    train_parser.add_argument(
        "--checkpoint", type=Path, help="Path to a custom trained model"
    )
 
    predict_parser = subparsers.add_parser("predict", help="Predict using a model")
    predict_parser.add_argument(
        "--checkpoint", type=Path, help="Path to a custom trained model"
    )
    predict_parser.add_argument(
        "--graph", type=Path, help="Path to a polypharmacy graph"
    )

    predict_group = predict_parser.add_argument_group("Predict specific options")
    predict_group.add_argument(
        "--drugs",
        nargs="*",
        help="List of drugs in interaction",
        default=[],
        metavar="drug",
    )
    predict_group.add_argument(
        "--effects",
        nargs="*",
        help="List of potential effects",
        default=[],
        metavar="effect",
    )

    args = parser.parse_args()
    if args.mode == "train":
        train = Train(args.train, args.test, args.out, args.config, args.checkpoint)
        train.train()

    elif args.mode == "predict":
        if len(args.drugs) < 2:
            print("Error: At least two drugs are required for prediction.")
            exit(1)

        inference = Inference(args.checkpoint, args.graph)
        inference.predict(args.drugs, args.effects)
