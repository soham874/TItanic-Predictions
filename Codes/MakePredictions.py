from ModelOperations import *
from DatasetOperations import *

from numpy import savetxt

modelname = "BestSVC.pkl"
model = joblib.load(os.path.join(MODEL_PATH,modelname))
print(model)

print("<< Loading test set...")
test_set = pd.read_csv(os.path.join(DATASETS,'test.csv'))
test_set = process_data(test_set)

#X_test = test_set.drop(test_set.columns[[0]],axis=1)

print("<< Predicting labels using model....")
y_pred = model.predict(test_set)

savetxt(os.path.join(DATASETS,'y_predictions.csv'),y_pred)