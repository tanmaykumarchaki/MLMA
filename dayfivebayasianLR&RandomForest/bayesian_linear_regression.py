import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n=50
x = np.linspace(0,10,n).reshape(-1,1)
x_bias = np.hstack([np.ones((n,1)), x])

true_theta = np.array([1.5, 2.5])
sigma = 1.0

y = x_bias @ true_theta + np.random.normal(0, sigma, n)

mu_0 = np.zeros(2)
sigma_0 = np.eye(2) * 5

sigma_n = np.linalg.inv(np.linalg.inv(sigma_0) + (x_bias.T @ x_bias) / sigma**2)

mu_n = sigma_n @ (np.linalg.inv(sigma_0) @ mu_0 + ( x_bias.T @ y)/ sigma **2)
print("Posterior mean:\n", mu_n)

#Predictive Distribution
x_test = np.linspace(0,10,100).reshape(-1,1)
x_test_bias = np.hstack([np.ones((100,1)), x_test])

y_mean = x_test_bias @ mu_n
y_var = sigma**2 + np.sum(x_test_bias @ sigma_n * x_test_bias, axis = 1)

plt.figure(figsize=(10,6))
plt.scatter(x,y, label = "Data")
plt.plot(x_test, y_mean, label = "Predictive Mean")
plt.fill_between(
    x_test.ravel(),
    y_mean - 1.96 * np.sqrt(y_var),
    y_mean + 1.96 * np.sqrt(y_var),
    alpha=0.3,
    label = "95% Confidence Interval"
)

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Bayesian Linear Regression Predictive Distribution")
plt.legend()
plt.show()