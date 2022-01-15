from DatasetOperations  import *

X_train, y_train = check_and_load_data()
print("Size of loaded training dataset->")
print(X_train.shape)
print(y_train.shape)