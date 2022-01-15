from DatasetOperations  import *
from ModelOperations import *

titanic_data = check_and_load_data()
print("Size of loaded training dataset->")
print(titanic_data.shape)

X_train,y_train = train_test_split(titanic_data)
X_train = process_data(X_train)
print(X_train.shape)

# Loading the best model using specified parameters
param_grid = {
        'weights':['uniform','distance'],
        'n_neighbors':[5,10,100]
    }
model = KNeighborsClassifier()
best_model = load_best_parameters(X_train,y_train,"BestKNClassifier.pkl",param_grid,model)
print("Model found with parameters ->")
print(best_model)

# Making predictions and evaluating the model, saving the paramters
evaluate_model(best_model,"KNClassifier")