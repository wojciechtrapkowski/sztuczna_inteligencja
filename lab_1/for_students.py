import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

def add_columns_of_ones_to_matrix(x_train, n : int):
    # Check if x_train is already 2D
    if x_train.ndim == 1:
        x_with_ones = np.column_stack((np.ones(len(x_train)), x_train))
    else:
        # Convert x_test to 2d matrix & add column of ones in the beginning
        x_with_ones = np.column_stack((np.ones(x_train.shape[0]), x_train))
    
    return x_with_ones

def create_observation_matrix(x_train, y_train):
    x_train_with_ones = add_columns_of_ones_to_matrix(x_train, len(x_train))
    # @ is matrix multiplication operator

    # np.lingalg.inv - calculate inverse of X
    theta = np.linalg.inv(x_train_with_ones.T @ x_train_with_ones) @ x_train_with_ones.T @ y_train
    
    return theta

def calculate_mse(x_train, y_train, theta_best):
    mse = 1/len(y_train)
    temp_sum = 0
    for i in range(len(y_train)):
        # temp_sum = y = ax+b
        temp_sum += ((x_train[i] * theta_best[1] + theta_best[0]) - y_train[i])**2
    return mse * temp_sum

def calculate_standarization(x_train, x_test, y_train, y_test):
    avg_x = np.mean(x_train, axis=0)
    std_x = np.std(x_train, axis=0)

    avg_y= np.mean(y_train, axis=0)
    std_y = np.std(y_train, axis=0)

    x_train = (x_train - avg_x) / std_x
    x_test = (x_test - avg_x) / std_x

    y_train = (y_train - avg_y) / std_y
    y_test = (y_test - avg_y) / std_y

    return x_train, x_test, y_train, y_test

def calculate_gradient_descent(x_train, y_train, learning_rate, epochs):
    m = len(y_train)
    x_train = add_columns_of_ones_to_matrix(x_train, len(x_train))
    y_train = y_train.reshape(-1, 1)  # Ensure y_train is a 2D array

    n = x_train.shape[1]  # Number of features
    theta = np.random.randn(n, 1)

    for epoch in range(epochs):
        predictions = np.dot(x_train, theta)
        error = predictions - y_train
        gradients = (2/m) * np.dot(x_train.T, error)
        theta = theta - learning_rate * gradients

    return theta

data = get_data()
# inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
theta_best = create_observation_matrix(x_train, y_train)
print(f"[THETA BEST] {theta_best}")

# TODO: calculate error - MSE (Mean Square Error)
# mse = calculate_mse(x_train, y_train, theta_best)
mse_test = calculate_mse(x_test, y_test, theta_best)
# print(f"[MSE TRAIN] {mse}")
print(f"[MSE TEST] {mse_test}")

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
standarized_x_train, standarized_x_test, standarized_y_train, standarized_y_test = calculate_standarization(x_train, x_test, y_train, y_test)

learning_rate = 0.01
epochs = 1000

# TODO: calculate theta using Batch Gradient Descent
theta_best = calculate_gradient_descent(standarized_x_train, standarized_y_train, learning_rate, epochs)
print(f"[THETA BEST] (with gradient descent) [{theta_best[0]} {theta_best[1]}]")

# TODO: calculate error
# mse = calculate_mse(standarized_x_train, standarized_y_train, theta_best)
mse_test = calculate_mse(standarized_x_test, standarized_y_test, theta_best)
# print(f"[MSE TRAIN] (with gradient descent) {mse}")
print(f"[MSE TEST] (with gradient descent) {mse_test}")

# plot the regression line
x = np.linspace(min(standarized_x_test), max(standarized_x_test), 100)
y = theta_best[0] + theta_best[1] * x
plt.plot(x, y)
plt.scatter(standarized_x_test, standarized_y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()
