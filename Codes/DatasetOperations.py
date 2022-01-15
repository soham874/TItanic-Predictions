from numpy import loadtxt,savetxt

import os
import pandas as pd

# Generic Paths
MODEL_PATH = os.path.join("Models")
DATASETS = os.path.join("Datasets")

if not os.path.isdir(MODEL_PATH):
    os.makedirs(MODEL_PATH)
if not os.path.isdir(DATASETS):
    os.makedirs(DATASETS)

# load the training set as a pandas dataframe and split it into features and labels
def check_and_load_data():

    if not os.path.isfile(os.path.join(DATASETS,'train.csv')):
        print("Datsets not found. Please download them from https://www.kaggle.com/c/titanic/data and move them to Datasets directory for this project before proceeding")
        return [],[]
    
    print("Datasets found, loading...")

    titanic_data = pd.read_csv(os.path.join(DATASETS,'train.csv'))
    print("Information about loaded training set ->")
    print(titanic_data.info())
    X_train = titanic_data.drop("Survived",axis=1)
    y_train = titanic_data["Survived"].copy()

    return X_train,y_train
