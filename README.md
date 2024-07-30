# vEXP: A virtual enhanced cross screen panel for off-target pharmacology alerts

* James A. Lumley, David Fallon, Ryan Whatling, Damien Coupry, Andrew Brown
* _Computational Toxicology, Volume 31, 2024, 100324, ISSN 2468-1113_
* [https://doi.org/10.1016/j.comtox.2024.100324.](https://doi.org/10.1016/j.comtox.2024.100324)


This repository contains the publically available ChEMBL datasets along with RDKit + sklearn notebooks and scripts to reproduce the vEXP ChEMBL models.  A summary of the models with their dataset statistics and model performance metrics is found in the [models folder README file](./models/README.md).


## Installation

In order to install the conda environment you need to follow the instructions below:

1) Have a valid installation of anaconda or miniconda
2) git clone https://github.com/gskcheminformatics/vEXP_ChEMBL.git
3) cd Chembl_vEXP_Models
4) When you are in the parent directory and your conda module is activated use the following command line

```bash
conda env create -f environment.yml
```

5) Activate the environment using the following command line

```bash
conda activate vexp_chembl_env
```

6) Create a kernel for use in jupyter lab using the following command line 

```bash
python3 -m ipykernel install --user --name vexp_chembl_env --display-name "vexp_chembl_env"
```

## Rebuilding a model

Copy the model_build_notebook.ipynb file into one of the target model directories and run the notebook.  You should see a drop down with the vexp_chembl_env kernel.  Edit the first cell that contains the model and model file names. You can now reproduce the paper results by simply running through the cells in the notebook.

## Making Temporal predictions with the predictor notebook

The predictor notebook allows an end to end pipeline to be run for each of the models. When in the notebook the first cell should be the model and smiles files to use.  You can now run predicitons by simply running through the cells in the notebook.
