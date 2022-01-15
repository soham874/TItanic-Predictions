from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer

import os
import matplotlib.pyplot as plt
import pandas as pd

# Generic Paths
DATASETS = os.path.join("Datasets")
IMAGE_PATH = os.path.join("Images")

if not os.path.isdir(IMAGE_PATH):
    os.makedirs(IMAGE_PATH)
if not os.path.isdir(DATASETS):
    os.makedirs(DATASETS)

# load the training set as a pandas dataframe and split it into features and labels
def check_and_load_data():

    if not os.path.isfile(os.path.join(DATASETS,'train.csv')):
        print("Datsets not found. Please download them from https://www.kaggle.com/c/titanic/data and move them to Datasets directory for this project before proceeding")
        return []
    
    print("<< Datasets found, loading...")

    titanic_data = pd.read_csv(os.path.join(DATASETS,'train.csv'))
    print("Information about loaded training set ->")
    print(titanic_data.info())

    return titanic_data

# create the test and train set
def train_test_split(titanic_data):

    # Splitting the set into train and test set
    print("<< Splitting data into train and test sets ->")

    stratfold = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in stratfold.split(titanic_data, titanic_data["Survived"]):
        strat_train_set = titanic_data.loc[train_index]
        strat_test_set = titanic_data.loc[test_index]

    print("Size of train and test set")
    print(strat_train_set.shape)
    print(strat_test_set.shape)
    print("Target stats for the created set ->")
    print(strat_train_set["Survived"].value_counts())
    print(strat_test_set["Survived"].value_counts()) 

    # splitting data into features and labels
    X_train = strat_train_set.drop("Survived",axis=1)
    y_train = strat_train_set["Survived"].copy()

    strat_test_set.to_csv(os.path.join(DATASETS,'test_from_train.csv')) 

    return X_train,y_train

# Pre-process data
def process_data(titanic_data):

    # Coverting Embarked data into integer type
    # 1 -> Cherbourg(C)
    # 2 -> Queenstown(Q)
    # 3 -> Southampton(S)
    titanic_data.loc[titanic_data["Embarked"] == 'C', "Embarked" ] = 1
    titanic_data.loc[titanic_data["Embarked"] == 'Q', "Embarked" ] = 2
    titanic_data.loc[titanic_data["Embarked"] == 'S', "Embarked" ] = 3
    titanic_data["Embarked"] = pd.to_numeric(titanic_data["Embarked"])

    # Coverting Gender data into integer type
    # 1 -> male
    # 2 -> female
    titanic_data.loc[titanic_data["Sex"] == "male", "Sex" ] = 1
    titanic_data.loc[titanic_data["Sex"] == "female", "Sex" ] = 2
    titanic_data["Sex"] = pd.to_numeric(titanic_data["Sex"])
    
    # Dropping cabin, name and ticket number column
    titanic_data.drop("Name",axis=1,inplace = True)
    titanic_data.drop("Ticket",axis=1,inplace = True)
    titanic_data.drop("Cabin",axis=1,inplace = True)

    # Filling in missing "age" and "emabrked" column using median impute startegy
    imputer = SimpleImputer(strategy="most_frequent")
    imputer.fit(titanic_data)
    titanic_data = pd.DataFrame(imputer.transform(titanic_data),columns=titanic_data.columns)

    # plotting the available information for each column in histograms and saving it
    print("Information about final processed training dataset ->")
    print(titanic_data.info())
    titanic_data.hist(bins=50, figsize=(20,15))
    # plt.savefig(os.path.join(IMAGE_PATH,"column_information_prepared.png"))
    # plt.show()

    return titanic_data