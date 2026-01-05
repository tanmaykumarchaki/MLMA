import numpy as np 
import matplotlib.pyplot as plt

x = np.array([[1],[2],[3],[4],[5]])
y = np.array([[112],[106],[84],[63],[145]])

n = x.shape[0]
print(n)

ones = np.ones((n,1))
print(ones)

x_bias = np.c_[ones,x]
print(x_bias)
print(x_bias.shape)

theta = np.zeros((2,1))
learning_rate = 0.005
iterations = 200

def compute_cost(x,y,theta):
    y_pred = x.dot(theta)
    return (1/n) * np.sum((y_pred - y)**2)

cost_history = []

for i in range(iterations):
    y_pred = x_bias.dot(theta)
    grad = 2/n * x_bias.T.dot(y_pred - y)

    theta = theta - learning_rate * grad
    cost = compute_cost(x_bias,y,theta)
    cost_history.append(cost)

print("Final Theta:", theta)
print("Final Cost:", cost_history[-1])
print("Cost History:", cost_history)

#For Gradient Descent Visualization
plt.plot(range(iterations), cost_history)
plt.xlabel("iterations")
plt.ylabel("Cost")
plt.title("Gradient Descent (cost v/s iterations)")

#For Regresion Line Visualization
plt.scatter(x,y, label = "Data Points", color = 'red')
plt.plot(x, x_bias.dot(theta), label = "Regression Line")
plt.xlabel("x")
plt.ylabel("y")
plt.title('Linear Regression Fit')
plt.legend()
plt.show()


