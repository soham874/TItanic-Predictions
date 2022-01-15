import os
import joblib
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from DatasetOperations import *

MODEL_PATH = os.path.join("Models")
if not os.path.isdir(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# function to save confusion and its error matrix
def conf_err(conf_mat,name):

    if not os.path.isdir(os.path.join(IMAGE_PATH,name)):
        os.makedirs(os.path.join(IMAGE_PATH,name))

    # confusion matix
    plt.matshow(conf_mat, cmap=plt.cm.gray)
    plt.savefig(os.path.join(IMAGE_PATH,name,"Confusion_matrix.png"))

    row_sums = conf_mat.sum(axis=1, keepdims=True)
    norm_conf_mat = conf_mat / row_sums

    np.fill_diagonal(norm_conf_mat, 0)
    plt.matshow(norm_conf_mat, cmap=plt.cm.gray)
    plt.savefig(os.path.join(IMAGE_PATH,name,"Error_matrix.png"))

# Evaluate a model
def evaluate_model(model,name):

    file_object = open('Model_Performances.txt', 'a')
    file_object.write("\nEvaluation for model "+name+"\n")

    # Making predictions
    print("~~~~~~~~~~~~~~~~~~~~~~ Model Evaluation ~~~~~~~~~~~~~~~~~~~")
    print("Loading test set...")
    test_set = pd.read_csv(os.path.join(DATASETS,'test_from_train.csv'))
    test_set = process_data(test_set)
    X_test = test_set.drop("Survived",axis=1)
    X_test = X_test.drop(X_test.columns[[0]],axis=1)
    y_test = test_set["Survived"].copy()
    
    print("Predicting labels using model....")
    y_pred = model.predict(X_test)

    # Confusion Matrix
    print("Evaluating confusion matrix..")
    conf_mat = confusion_matrix(y_test,y_pred)
    file_object.write("\nConfusion matrix -> \n")
    file_object.write(str(np.array(conf_mat)))

    conf_err(conf_mat,name)

    # Precision, Recall, F1
    print("Evaluating Precision, Recall, F1..")
    file_object.write("\nPrecision -> "+str(precision_score(y_test,y_pred,average="macro")))
    file_object.write("\nRecall -> "+str(recall_score(y_test,y_pred,average="macro")))
    file_object.write("\nF1 score -> "+str(f1_score(y_test,y_pred,average="macro")))
    
    print("Report stored in Model_performances.txt")
    file_object.close()