
![Screenshot](./design/header.png)

---

# Pharmageddon

1. Install Dependencies: `pip install poetry`

2. Download data and models here: https://drive.google.com/file/d/1hinxrf1u1hMRsR6_g1fP84wumLJKvKce/view?usp=sharing\
The folder structure should look like this
```
├── data <-- You added this from the google drive link
│   └── ...
├── design
│   └── ...
├── src
│   └── ...
├── Dockerfile
...
└── README.me
```

1. Run `make venv` to create virtual environment

2.  activate venv `source .venv/bin/activate`

3. install torch-geometric `pip install torch-geometric==2.3.1`\

4. Run streamlit: `streamlit run src/webapp.py`\

5. (Optional) To train the model using your own polypharmacy datasets rn:
    ```shell
    pharmageddon train --train <path to training set> --test <path to test set> --out <path to output directory> [--config <path to config file> --checkpoint <path to trained model>]
    ```

6. (Optional) Use the trained model to predict side effects for your chemical compound of interest. The `--checkpoint` parameter can be omitted - the default model will then be used.  `--effects` can also be omitted, probabilities for all available effects are predicted then.
    ```shell
    pharmageddon predict --graph <path to polypharmacy graph> --drugs <drug1> <drug2>  [--checkpoint <path_to_model> --effects <effect1> <effect2>]
    ```