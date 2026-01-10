import numpy as np

class OLS:
    def fit(self,x,y):
        x_bias = np.c_[np.ones((x.shape[0], 1)), x] # adding Bias term
        self.theta = np.linalg.inv(x_bias.T @ x_bias) @ x_bias.T @ y # OLS Normal Equation
        
    def predict(self, x):
        x_bias = np.c_[np.ones((x.shape[0], 1)), x]
        return x_bias @ self.theta
    

class Ridge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def fit(self,x,y):
        x_bias = np.c_[np.ones((x.shape[0], 1)), x]
        I = np.eye(x_bias.shape[1])
        self.theta = np.linalg.inv(x_bias.T @ x_bias + self.alpha * I) @ x_bias.T @ y

    def predict(self, x):
        x_bias = np.c_[np.ones((x.shape[0], 1)), x]
        return x_bias @ self.theta
    
print("Model Classes Defined: OLS and Ridge.")
    

