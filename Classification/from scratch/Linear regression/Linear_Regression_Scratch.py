import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('C:/Users/Nicolaus')

def loss_function(m, b, points):
    total_errors = 0
    for i in range(len(points)):
        x = points.iloc[i]
        y = points.iloc[i]
        total_errors += (y - (m * x + b)) ** 2
    total_errors / float(len(points))

def gradient_descent(m_now, b_now, points, learning_rate):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range (n):
        x = points.iloc[i]
        y = points.iloc[i]

        m_gradient += -(2/n ) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n ) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * learning_rate
    b = b_now - b_gradient * learning_rate
    return m, b

m = 0
b = 0
learning_rate = 0.0001
epochs = 1000

for i in range (epochs):
    m, b = gradient_descent(m, b, data, learning_rate)

print(m, b)

plt.scatter(data, data, color='red')