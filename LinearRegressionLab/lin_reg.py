import numpy as np
from scipy import stats


class LinearRegression:
    def __init__(self):
        #we create the variables that will be needed for the linear regression
        self.b = None
        self.n = None
        self.d = None
    
    def fit(self, X, y):
        #Now we start defining n and d
        #Here I saved some elements that will be reused several times in the code 
        # to prevent the script from having to repeat the same calculations several times
        self.n = X.shape[0]
        self.d = X.shape[1]-1

        #Now that we have all needed values we can calculate OLS
        #The formula is b=(X^T * X)^-1 * X^T * y

        XT = X.T #This is the transpose of matrix X
        XTX = XT @ X # @ Performs matrix multiplication
        self.XTX_inv = np.linalg.inv(XTX) # This will give us the same effect as ^-1
        XTY = XT @ y # This is the transpose of X by y
        self.b = self.XTX_inv @ XTY # finally we complete the formula by multiplying the two parts 

        self.y_hat = self.predict(X) #get predictions for X
        self.residuals = y-self.y_hat

    def predict(self, X):
        return X @ self.b
    
    def variance_calc(self, X, y):
        #SSE- sum of quared errors
        sse = np.sum(self.residuals**2)
        # now we finally can calculate the variance
        variance = sse/(self.n - self.d - 1)
        return variance
    
    def st_dev_calc(self, X, y):
        var = self.variance_calc(X,y)
        st_dev = np.sqrt(var)
        return st_dev
    
    def rmse_calc (self, X, y):
        sse = np.sum(self.residuals**2)
        mse = sse/self.n
        rmse = np.sqrt(mse)
        return rmse


#First we extract the data from the file, using genfromtxt will help by automatically splitting and converting the data
data = np.genfromtxt("housing.csv", delimiter="," , 
                     names=True , dtype=None, encoding="utf-8")
# Delete any row that has a NaN anywhere in it
data = data[~np.isnan(data['total_bedrooms'])]
print("Columns:", data.dtype.names)

#next we want the target variable Y
y = data["median_house_value"]
print(f"Number of target values (n): {len(y)}")
n_rows = len(y)
#now that we have y we can continue working on setting all the elements needed for the operations
#we will now assign the features 
x_features = np.column_stack((data["housing_median_age"], data ["total_rooms"]))
print(f"Shape of features matrix: {x_features.shape}")
#now we need to create a column for the intercept
ones = np.ones(n_rows)
#now we create the matrix X
X = np.column_stack((ones, x_features))
model = LinearRegression()
model.fit(X,y)
var = model.variance_calc(X,y)
st_dev = model.st_dev_calc(X,y)
rmse = model.rmse_calc(X,y)
print(f"Model trained with n:{model.n} and d:{model.d}")
print(f"Coefficients b:", model.b)
print(f"Sample Variance: {var}")
print(f"Standard Deviation: {st_dev}")
print(f"RMSE : {rmse}")