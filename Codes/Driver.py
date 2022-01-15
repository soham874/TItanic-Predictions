from DatasetOperations  import *
from ModelOperations import *

titanic_data = check_and_load_data()
print("Size of loaded training dataset->")
print(titanic_data.shape)

X_train,y_train = train_test_split(titanic_data)
X_train = process_data(X_train)

best_model = load_best_parameters(X_train,y_train,"BestKNClassifier.pkl")