import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from Logistic_Regression import logisticRegression

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

lg = logisticRegression()
lg.fit(X_train, y_train)
predictions = lg.predict(X_test)

def accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true)/len(y_true)

acc = accuracy(predictions, y_test)
print(f'accuracy: {acc}')