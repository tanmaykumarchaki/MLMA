import numpy as np 
import matplotlib.pyplot as plt 

np.random.seed(0) # For reproducibility

n = 100  
x = np.linspace(0, 10, n).reshape(-1,1) # feature matrix
x_bias = np.hstack([np.ones((n,1)), x]) # adding bias term 

true_theta = np.array([2,3])   # True Parameter values dedicated to intercept and slope
sigma = 1.0
noise = np.random.normal(0, sigma, size=n)

y = x_bias @ true_theta + noise # target_variable  

print("True Theta:", true_theta)
print("Sigma:", sigma)
print("Noise:", noise)

def mse(y_true, y_pred):  #Mean Squared Error
    return np.mean((y_true - y_pred)**2) 

def log_likelihood(x, y, theta, sigma): # Log Likelihood Function , mostly use in statistics
    res = y - x @ theta
    return -1 / (2 * sigma**2) * np.sum(res**2)  # Negative log likelihood

theta_test = np.linspace(0 ,5 , 100)  # Testing different slope values
mse_values = []
log_likelihood_values = []

for t in theta_test:
    theta_vec = np.array([2,t])  # Varying only the slope
    y_pred = x_bias @ theta_vec  # Predicted values
    mse_values.append(mse(y_pred, y)) # Calculating MSE
    log_likelihood_values.append(log_likelihood(x_bias, y, theta_vec, sigma)) # Calculating Log Likelihood

print("MSE VALUES:", mse_values)
print("Log Likelihood Values:", log_likelihood_values)


#Visualization
plt.figure(figsize = (12, 5))

plt.subplot(1,2,1)
plt.plot(theta_test, mse_values)
plt.title("MSE / Theta")
plt.xlabel("Theta")
plt.ylabel("MSE")

plt.subplot(1,2,2)
plt.plot(theta_test, log_likelihood_values)
plt.title("Log Likelihood / Theta")
plt.xlabel("Theta")
plt.ylabel("Log Likelihood")

plt.tight_layout()
plt.show()





