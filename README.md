
![Screenshot](./design/header.png)

---

# Pharmageddon

## About

## Demo Video

## Interface

## Installation

Dependencies: `poetry`

run `make venv`
activate venv `source .venv/bin/activate`
install torch-geometric `pip install torch-geometric==2.3.1`
run streamlit: `streamlit run src/webapp.py`

1. Install the `pharmageddon-polypharmacy` package from PyPI.

    ```shell
    pip install pharmageddon-polypharmacy
    ```

2. Train the model using your own polypharmacy datasets

    ```shell
    pharmageddon train --train <path to training set> --test <path to test set> --out <path to output directory> [--config <path to config file> --checkpoint <path to trained model>]
    ```

    OR
    Download a checkpoint file from a pretrained model.
    Checkpoint files will be made available. A default checkpoint file is also included in the PyPI version.

3. Use the trained model to predict side effects for your chemical compound of interest. The `--checkpoint` parameter can be omitted - the default model will then be used.  `--effects` can also be omitted, probabilities for all available effects are predicted then.

    ```shell
    pharmageddon predict --graph <path to polypharmacy graph> --drugs <drug1> <drug2>  [--checkpoint <path_to_model> --effects <effect1> <effect2>]
    ```