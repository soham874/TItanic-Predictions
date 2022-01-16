from DatasetOperations  import *
from ModelOperations import *

from sklearn.ensemble import RandomForestClassifier

titanic_data = check_and_load_data()
print("Size of loaded training dataset->")
print(titanic_data.shape)

X_train,y_train = train_test_split(titanic_data)
X_train = process_data(X_train)
print(X_train.shape)

modelname = "BestRandomForest.pkl"
# Loading the best model using specified parameters
if os.path.isfile(os.path.join(MODEL_PATH,modelname)):
    best_model = joblib.load(os.path.join(MODEL_PATH,modelname))
else:

    param_grid = {
        'bootstrap':[True,False],
        'n_estimators':[10,40,50,100]
    }
    model = RandomForestClassifier()

    grid_seach_result = GridSearchCV(model , param_grid , cv=5, verbose=20)
    grid_seach_result.fit(X_train, y_train)

    print(grid_seach_result.best_params_)
    print(grid_seach_result.best_score_)
    joblib.dump(grid_seach_result.best_estimator_,os.path.join(MODEL_PATH,modelname))
    best_model = grid_seach_result.best_estimator_

print("Model found with parameters ->")
print(best_model)

# Making predictions and evaluating the model, saving the paramters
evaluate_model(best_model,"RandomForest")