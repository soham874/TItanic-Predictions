from DatasetOperations  import *

titanic_data = check_and_load_data()
print("Size of loaded training dataset->")
print(titanic_data.shape)

X_train,y_train = train_test_split(titanic_data)
X_train = process_data(X_train)