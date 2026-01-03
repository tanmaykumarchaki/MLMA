import numpy as np
import pandas as pd 

#USING MANUAL IMPLEMENTATION APPLYING NORMAL EQUATION 
data = pd.DataFrame({
    'Plot': [1,2,3,4,5,6,7,8],
    'Price':[245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000]
})

x = data['Plot'].values
y = data['Price'].values

x_bias =  np.c_[np.ones((x.shape[0])), x]  #adding bias term 
theta = np.linalg.inv(x_bias.T.dot(x_bias)).dot(x_bias.T).dot(y) #Normal Equation
print("Computed Parameters (theta):", theta.round(2))

y_pred = x_bias @ theta
mse = np.mean((y - y_pred)** 2)
print("Mean Squared Error(Calculated):", mse.round(2))

#USING THE CLASS IMPLEMENTATION
class linearregression:
    def __init__(self):
        self.beta = None

    def _add_bias(self,x):
        return np.c_[np.ones((x.shape[0])), x] # adding bias inside the model class
    
    def predict(self, x):
        x_bias = self._add_bias(x)
        return x_bias @ self.beta
    def mse(self,x,y):
        y_pred = self.predict(x)
        return np.mean((y - y_pred)** 2)
    def fit(self, x, y):
        x_bias = self._add_bias(x)
        self.beta = np.linalg.inv(x_bias.T.dot(x_bias)).dot(x_bias.T).dot(y)
        return self
    
model = linearregression()
model.fit(x,y)
print("Model Intercept=", model.beta[0].round(2))
print("Model Slope =", model.beta[1].round(2))
print("Model MSE =", model.mse(x,y).round(2))