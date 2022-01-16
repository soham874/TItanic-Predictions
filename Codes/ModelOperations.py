import os
import joblib
import numpy as np

from sklearn.model_selection import GridSearchCV,cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score, roc_curve

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

# Using this scores, we can plot the precision recall curve for the calssifier
def precision_recall_vs_threshold(precisions, recalls, thresholds, name):
    plt.figure(figsize=(16, 9), dpi=150)
    plt.rcParams.update({'font.size': 22})
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel('Threshold')
    plt.grid(True)
    plt.legend(['precision','recall'])
    plt.ylim(0,1.1)
    plt.savefig(os.path.join(IMAGE_PATH,name,"precision_recall_vs_threshold.png"))

# Another form of measurement is to plot the precision vs recall curve.
def precision_vs_recall(precisions, recalls, name):
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(16, 9), dpi=150)
    plt.plot(recalls[:-1], precisions[:-1], "b-")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    #plt.xlim(-4000,4000)
    plt.savefig(os.path.join(IMAGE_PATH,name,"precision_vs_recall.png"))

# Plot ROC curve
def plot_roc_curve(fpr, tpr, name):
    plt.figure(figsize=(16, 9), dpi=150)
    plt.rcParams.update({'font.size': 22})
    plt.plot(fpr, tpr, "b-", label="Precision")
    plt.plot(np.linspace(0,1,100),np.linspace(0,1,100),"g--")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.grid(True)
    plt.savefig(os.path.join(IMAGE_PATH,name,"ROC_curve.png"))

# Evaluate a model
def evaluate_model(model,name):

    file_object = open('Model_Performances.txt', 'a')
    file_object.write("\n\nEvaluation for model "+name+"\n")

    # Making predictions
    print("~~~~~~~~~~~~~~~~~~~~~~ Model Evaluation ~~~~~~~~~~~~~~~~~~~")
    print("<< Loading test set...")
    test_set = pd.read_csv(os.path.join(DATASETS,'test_from_train.csv'))
    test_set = process_data(test_set)
    X_test = test_set.drop("Survived",axis=1)
    X_test = X_test.drop(X_test.columns[[0]],axis=1)
    y_test = test_set["Survived"].copy()
    
    print("<< Predicting labels using model....")
    y_pred = model.predict(X_test)

    # Confusion Matrix
    print("<< Evaluating confusion matrix..")
    conf_mat = confusion_matrix(y_test,y_pred)
    file_object.write("\nConfusion matrix -> \n")
    file_object.write(str(np.array(conf_mat)))

    conf_err(conf_mat,name)

    # Precision, Recall, F1
    print("<< Evaluating Precision, Recall, F1, AUC..")
    file_object.write("\nPrecision -> "+str(precision_score(y_test,y_pred,average="macro")))
    file_object.write("\nRecall -> "+str(recall_score(y_test,y_pred,average="macro")))
    file_object.write("\nF1 score -> "+str(f1_score(y_test,y_pred,average="macro")))
    
    # evaluating roc/auc
    y_scores = cross_val_predict(model, X_test, y_test, cv=3)
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
    precision_recall_vs_threshold(precisions, recalls, thresholds, name)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    plot_roc_curve(fpr, tpr, name)

    file_object.write("\nAUC -> "+str(roc_auc_score(y_test,y_scores)))

    print("Report stored in Model_performances.txt")
    file_object.close()