from numpy import loadtxt,savetxt

import os
import matplotlib.pyplot as plt
import pandas as pd


# Generic Paths
MODEL_PATH = os.path.join("Models")
DATASETS = os.path.join("Datasets")
IMAGE_PATH = os.path.join("Images")

if not os.path.isdir(IMAGE_PATH):
    os.makedirs(IMAGE_PATH)
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
    # print(titanic_data.info())

    # splitting data into features and labels
    # X_train = titanic_data.drop("Survived",axis=1)
    # y_train = titanic_data["Survived"].copy()

    # return X_train,y_train
    return titanic_data

def process_data(titanic_data):

    # Coverting Embarked data into integer type
    # 1 -> Cherbourg(C)
    # 2 -> Queenstown(Q)
    # 3 -> Southampton(S)
    titanic_data.loc[titanic_data["Embarked"] == 'C', "Embarked" ] = 1
    titanic_data.loc[titanic_data["Embarked"] == 'Q', "Embarked" ] = 2
    titanic_data.loc[titanic_data["Embarked"] == 'S', "Embarked" ] = 3
    titanic_data["Embarked"] = pd.to_numeric(titanic_data["Embarked"])

    # Dropping cabin, name and ticket number column
    titanic_data.drop("Name",axis=1,inplace = True)
    titanic_data.drop("Ticket",axis=1,inplace = True)
    titanic_data.drop("Cabin",axis=1,inplace = True)

    # return
    # plotting the available information for each column in histograms and saving it
    print(titanic_data.info())
    titanic_data.hist(bins=50, figsize=(20,15))
    plt.savefig(os.path.join(IMAGE_PATH,"column_information.png"))
    plt.show()    