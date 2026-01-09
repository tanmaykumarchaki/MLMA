import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)

x = np.linspace(0,10,100).reshape(-1,1)
y = np.sin(x).ravel() + np.random.normal(0, 0.3, 100)

random_forest = RandomForestRegressor(n_estimators = 200,max_depth = 5, random_state = 42)
random_forest.fit(x,y)

x_test = np.linspace(0,10,200).reshape(-1,1)
y_pred = random_forest.predict(x_test)

plt.figure(figsize = (10,6))
plt.scatter(x,y, label = "Data")
plt.plot(x_test, y_pred, color = 'red' , label = 'Random Forest Prediction')
plt.legend()
plt.title('Random Forest Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

