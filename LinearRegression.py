import numpy as np

class LinearRegression:
    def __init__(self):
        self.w = None

    """
    Fits the linear regression model using the normal equation:
    w = (X^T X)^-1 X^T y
    """
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape((-1, 1))
        
        X_ = np.hstack([np.ones((X.shape[0], 1)), X])
        print("X after adding bias term:")
        print(X_)
        
        # Compute Σ(x_i x_i^T)
        sum_matrix = np.zeros((X_.shape[1], X_.shape[1]))  
        for x in X_:
            sum_matrix += np.outer(x, x)  
        
        # Compute X^T X
        XTX = sum_matrix
        XTXInv = np.linalg.pinv(XTX)

        # Compute Σ(y_i x_i) = X^T y
        XTy = X_.T.dot(y)

        # Compute the weights (intercept and slope)
        self.w = XTXInv.dot(XTy)
        return self.w

    def predict(self, X):

        X = np.array(X)
        X_ = np.hstack([np.ones((X.shape[0], 1)), X]) 
        return X_.dot(self.w)  


X = np.array([[1,2], [2,3], [3,4], [4,5]]) 
y = np.array([3, 5, 7, 9])          

model = LinearRegression()
weights = model.fit(X, y)
print("Weights (intercept, slope):", weights.flatten())


predictions = model.predict(X)
print("Predictions:", predictions.flatten())
