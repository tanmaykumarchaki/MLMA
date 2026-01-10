from daysevensystemsandmodelanalysis.data import generate_data
from daysevensystemsandmodelanalysis.model import OLS , Ridge 
from daysevensystemsandmodelanalysis.system import mse, conditional_variance
import numpy as np

x , y = generate_data(200)

split = int(0.8 * len(x))
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

ols = OLS()
ridge = Ridge(alpha = 1.0)

ols.fit(x_train, y_train)
ridge.fit(x_train, y_train)

ols_pred = ols.predict(x_test)
ridge_pred = ridge.predict(x_test)

print(f"OLS MSE:{mse(y_test, ols_pred)}", f"Ridge MSE:{mse(y_test, ridge_pred)}", f"Condition Number:{conditional_variance(x_train)}", sep = "\n")
