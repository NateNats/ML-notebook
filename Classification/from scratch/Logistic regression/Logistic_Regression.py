import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class logisticRegression():
    def __init__(self, learning_rate=0.001, n_iters=10000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1/ (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_prediction = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_prediction)

            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weight = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, X):
        linear_prediction = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_prediction)
        y_predicted_cls = [0 if y <= 0.5 else 1 for y in y_predicted]
        return y_predicted_cls