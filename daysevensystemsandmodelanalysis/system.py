import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def conditional_variance(x):
    normal = x.T @ x
    eigvals = np.linalg.eigvals(normal)
    return np.var(eigvals)

print("System Utilities Loaded: MSE and Conditional Variance.")
