from DatasetOperations  import *
from ModelOperations import *

from sklearn.svm import SVC

titanic_data = check_and_load_data()
print("Size of loaded training dataset->")
print(titanic_data.shape)

X_train,y_train = train_test_split(titanic_data)
X_train = process_data(X_train)
print(X_train.shape)

modelname = "BestSVC.pkl"
# Loading the best model using specified parameters
if os.path.isfile(os.path.join(MODEL_PATH,modelname)):
    best_model = joblib.load(os.path.join(MODEL_PATH,modelname))
else:

    param_grid = {
        'kernel':["linear"],
        'gamma':["scale", "auto"],
        'C':[3,10,100]
    }
    model = SVC()

    grid_seach_result = GridSearchCV(model , param_grid , cv=5, verbose=20)
    grid_seach_result.fit(X_train, y_train)

    print(grid_seach_result.best_params_)
    print(grid_seach_result.best_score_)
    joblib.dump(grid_seach_result.best_estimator_,os.path.join(MODEL_PATH,modelname))
    best_model = grid_seach_result.best_estimator_

print("Model found with parameters ->")
print(best_model)

# Making predictions and evaluating the model, saving the paramters
evaluate_model(best_model,"SVC")