import numpy as np

# We will program a simple evaluation function to compute MSE

def generate_data(n_samples = 100):
    np.random.seed(42)
    x = 2*np.random.rand(n_samples, 1) # Unifomly distributed data
    noise = np.random.randn(n_samples, 1) # Gaussian Noise
    y = 4 + 3 * x + noise # Linear relation with noise
    return x, y

def train_test_split_func(x,y,test_ratio = 0.2):
    n = len(x) # number of samples
    test_size = int(n * test_ratio) # size of test set
    indices = np.random.permutation(n) # shuffling indices
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    return x[train_idx], x[test_idx], y[train_idx], y[test_idx] # returing train and test sets

# Normal Equation

class normalequation:
    def fit(self,x,y):
        x_bias = np.c_[np.ones((x.shape[0], 1)), x] # adding bias term
        self.theta = np.linalg.inv(x_bias.T @ x_bias) @ x_bias.T @ y # Normal Equation
        
    def predict(self,x):
        x_bias = np.c_[np.ones((x.shape[0], 1)), x] # adding bias term
        return x_bias @ self.theta
    

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("Model Evaluation:")
    print('-'*20)
    print("-> Mean Squared Error (MSE) =", mse.round(4))
    print("-> Mean Absolute Error (MAE)=", mae.round(4))
    print('-'*20)
    print("The Model Evaluation is complete.", end = "\n\n")

    return y_test, y_pred


x , y = generate_data(200)
x_train, x_test, y_train, y_test = train_test_split_func(x,y)
model = normalequation()
model.fit(x_train, y_train)
evaluate_model(model, x_test, y_test)



