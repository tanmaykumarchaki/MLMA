import numpy as np

def generate_data(n_samples, noise=1.0):
    np.random.seed(42)
    x = 2*np.random.rand(n_samples, 1) # Uniformly Distributed Data
    y = 3*x + 4 + noise * np.random.randn(n_samples, 1) # Linear Relation with Noise
    return x,y # returning features and target

print("Data Generation Complete.")

