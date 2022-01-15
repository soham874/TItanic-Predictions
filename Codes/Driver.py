from DatasetOperations  import *

titanic_data = check_and_load_data()
print("Size of loaded training dataset->")
print(titanic_data.shape)
# print(y_train.shape)

process_data(titanic_data)