import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso


np.random.seed(42)
n_samples = 100
n_features = 10

x = np.random.randn(n_samples, 1)
x = np.hstack([x + 0.01 * np.random.randn(n_samples, 1) for _ in range(n_features)])

true_theta = np.array([5, 4, 0, 0, 0, 0, 0, 0, 0, 0])
y = x @ true_theta + np.random.randn(n_samples)*2

print("Data Shape:", f"x = {x.shape}", f"y = {y.shape}", sep ="\n")

#Training and Test Split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)
print("Train/Test Shapes:", f"x_train = {x_train.shape}",f" y_train = {y_train.shape}",f" x_test = {x_test.shape}",f" y_test = {y_test.shape}", sep ="\n")

def ordinary_least_squares(x, y):
    return np.linalg.inv(x.T @ x) @ x.T @ y

theta_ols = ordinary_least_squares(x_train, y_train)
print("OLS Coefficients:", theta_ols)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

train_mse_ols = mse(y_train, x_train @ theta_ols)
test_mse_ols = mse(y_test, x_test @ theta_ols)

print(f"OLS Train MSE:{train_mse_ols:.4f}",f"OLS Test MSE:{test_mse_ols:.4f}", sep ="\n")

#Ridge Regression

def ridge_regression(x,y,lam):
    n_features = x.shape[1]
    return np.linalg.inv(x.T @ x + lam*np.eye(n_features)) @x.T @y

lam = 10
theta_ridge = ridge_regression(x_train, y_train, lam)
print("Ridge Coefficients:", theta_ridge)

train_mse_ridge = mse(y_train, x_train @ theta_ridge)
train_mse_ridge = mse(y_test, x_test @ theta_ridge)

print(f"Ridge Train MSE:{train_mse_ridge:.4f}",f"Ridge Test MSE:{train_mse_ridge:.4f}", sep ="\n")


lasso = Lasso(alpha = 1.0, max_iter= 10000)
lasso.fit(x_train, y_train)
theta_lasso = lasso.coef_
print("Lasso Coefficients:", theta_lasso)

train_mse_lasso = mse(y_train, lasso.predict(x_train))
test_mse_lasso = mse(y_test, lasso.predict(x_test))
print(f"Lasso Train MSE:{train_mse_lasso:.4f}",f"Lasso Test MSE:{test_mse_lasso:.4f}", sep ="\n")


#Comparing Coefficients
plt.figure(figsize=(10,6))
plt.plot(theta_ols, label ="OLS Coefficients", marker ='o')
plt.plot(theta_ridge, label = "Ridge Coefficients", marker = 'o')
plt.plot(theta_lasso, label = "Lasso Coefficients", marker = 'o')
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.legend()
plt.axhline(0)
plt.title("Comparison of Coefficients")
plt.show()
