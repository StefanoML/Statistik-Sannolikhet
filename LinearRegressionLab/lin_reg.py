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
    
    def r_squared_calc (self, X, y):
        y_mean = np.mean(y)
        syy = np.sum((y-y_mean)**2)
        sse = np.sum(self.residuals**2)
        ssr = syy - sse
        r_squared = ssr/syy
        return r_squared
    
    def f_test(self, X, y):
        y_mean = np.mean(y)
        syy = np.sum((y-y_mean)**2)
        sse = np.sum(self.residuals**2)
        ssr = syy - sse
        variance = self.variance_calc(X,y)
        #degrees of freedom
        df1 = self.d
        df2 = (self.n - self.d - 1)

        f_stat = (ssr/self.d)/variance

        p_value = stats.f.sf(f_stat, df1, df2)
        return f_stat, p_value
    
    def pearson_corr (self, X):
        features = X[:,1:]
        n_features = features.shape[1]

        corr_matrix = np.zeros((n_features, n_features))

        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    corr_matrix[i,j] = 1.0
                else:
                    col_i=features[:,i]
                    col_j=features[:,j]

                    #now we can calculate the correlation
                    r,_ = stats.pearsonr(col_i, col_j)

                    #finally we store it in matrix
                    corr_matrix[i,j] = r

        return corr_matrix
    
    def t_tests(self, X, y):
        std_dev = self.st_dev_calc(X,y)
        variance = self.variance_calc(X,y)

        #We build the covariance matrix that will check uncertainty for all coefficienys
        cov_matrix = self.XTX_inv

        #degrees of freedom
        df = self.n - self.d - 1

        #Now we test the coefficients
        t_stats=[]
        p_values=[]

        for i in range(len(self.b)):
            beta = self.b[i] #coefficient value
            uncertainty = cov_matrix[i,i] #checks uncert. from diagonal

            #Standard error
            se = np.sqrt(np.abs(variance * uncertainty))

            #t-statistic
            t = (beta/se).item()

            #p-value 
            cdf = stats.t.cdf(t, df)
            sf = stats.t.sf(t, df)
            p = 2*min(cdf, sf)

            t_stats.append(t)
            p_values.append(p)
        
        return np.array(t_stats), np.array(p_values)
    

    def confidence_interval(self, X, y, alpha=0.05):
        df=self.n-self.d-1

        t_crit = stats.t.ppf(1-alpha/2, df)

        variance = self.variance_calc(X,y)
        uncertainty_diag = np.diag(self.XTX_inv)
        se = np.sqrt(np.abs(variance * uncertainty_diag))

        lower_bounds=[]
        upper_bounds=[]

        for i in range(len(self.b)):
            margin = t_crit * se[i]

            lower = (self.b[i] - margin).item()
            upper = (self.b[i] + margin).item()

            lower_bounds.append(lower)
            upper_bounds.append(upper)

        return lower_bounds, upper_bounds
