# ECE-143-Project
UCSD ECE Group 19 Project: Prediction and Analysis of Heart Failure

## Installation

Requires:
- conda
- python 3.9.5
- pandas
- numpy
- jupyter notebook
- sklearn
- seaborn
- plotly

Clone the repository using
```
git clone https://github.com/rskpdev/ECE-143-Project.git
```

Create a conda environment from the environment.yml file. The first line of the .yml file sets the new environment's name
```
conda env create -f environment.yml
```
Activate the conda environment
```
conda activate 143_env
```

Deactivate when done making changes
```
conda deactivate
```

## Usage

The data we used is stored in [data](https://github.com/rskpdev/ECE-143-Project/tree/main/data) folder, and the Machine Learning model scripts are stored in [model](https://github.com/rskpdev/ECE-143-Project/tree/main/model) folder.

### EDA

EDA of the features from dataset are stored in the notebook in [notebooks](https://github.com/rskpdev/ECE-143-Project/tree/main/notebooks) folder.<br>
View notebook here [jupyter nbviewer](https://nbviewer.jupyter.org/github/rskpdev/ECE-143-Project/blob/main/notebooks/ece143project_final.ipynb)

### Machine Learning

Machine Learning of the features from dataset are stored in the notebook in [notebooks](hhttps://github.com/rskpdev/ECE-143-Project/blob/main/model) folder.<br>
View notebook here [jupyter nbviewer](https://github.com/rskpdev/ECE-143-Project/blob/main/model/prediction.ipynb)

PCA.py 
This file is used for performing PCA to extract the top 2 features for visualization.
model.py 
This file consists of all the models used in prediction, plotting confusion matrix and calculating the metric scores.
encoder.py 
this file encodes the categorical features in dataset.
split_dataset.py 
This file splits the dataset into testing and training sets
