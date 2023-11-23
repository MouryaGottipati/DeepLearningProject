import numpy as np
import pandas as pd

def loadDataset(train_path="../Data/train.csv",test_path="../Data/test.csv"):
    # Load training data
    train_data = pd.read_csv(train_path)
    x_train = train_data.drop('label', axis=1).values.reshape((-1, 28, 28, 1))
    # x_train_reshaped = x_train.reshape(-1, 28, 28, 1)
    y_train = train_data['label'].values
    
    # Load testing data
    test_data = pd.read_csv(test_path)
    x_test = test_data.drop('label', axis=1).values.reshape((-1, 28, 28, 1))
    # x_test_reshaped = x_test.reshape(-1, 28, 28, 1)
    y_test = test_data['label'].values
    return x_train,y_train,x_test,y_test