"""
Linear Regression Formulas Reference
"""

import numpy as np
from scipy import stats

# ============================================
# 1. CALCULATE COEFFICIENTS (b vector)
# ============================================
# b = (X^T X)^(-1) X^T Y

def calculate_coefficients(X, Y):
    """Calculate regression coefficients using OLS"""
    b = np.linalg.inv(X.T @ X) @ X.T @ Y
    return b


# ============================================
# 2. PROPERTIES: n and d
# ============================================

def get_n_and_d(X):
    """Get sample size (n) and number of features (d)"""
    n = X.shape[0]  # number of rows
    d = X.shape[1] - 1  # number of columns minus intercept
    return n, d


# ============================================
# 3. SSE (Sum of Squared Errors)
# ============================================
# SSE = Σ(Yi - Ŷi)²

def calculate_SSE(Y, Y_predicted):
    """Calculate sum of squared errors"""
    SSE = np.sum((Y - Y_predicted) ** 2)
    return SSE


# ============================================
# 4. VARIANCE (σ̂²)
# ============================================
# σ̂² = SSE / (n - d - 1)

def calculate_variance(SSE, n, d):
    """Calculate unbiased variance estimate"""
    variance = SSE / (n - d - 1)
    return variance


# ============================================
# 5. STANDARD DEVIATION (σ̂)
# ============================================
# σ̂ = √σ̂²

def calculate_std_dev(variance):
    """Calculate standard deviation"""
    std_dev = np.sqrt(variance)
    return std_dev


# ============================================
# 6. RMSE (Root Mean Squared Error)
# ============================================
# MSE = SSE / n
# RMSE = √MSE

def calculate_RMSE(SSE, n):
    """Calculate RMSE"""
    MSE = SSE / n
    RMSE = np.sqrt(MSE)
    return RMSE


# ============================================
# 7. Syy (Total Sum of Squares)
# ============================================
# Syy = Σ(Y - Ȳ)²

def calculate_Syy(Y):
    """Calculate total sum of squares"""
    Y_mean = np.mean(Y)
    Syy = np.sum((Y - Y_mean) ** 2)
    return Syy


# ============================================
# 8. SSR (Sum of Squares Regression)
# ============================================
# SSR = Syy - SSE

def calculate_SSR(Syy, SSE):
    """Calculate regression sum of squares"""
    SSR = Syy - SSE
    return SSR


# ============================================
# 9. R² (Coefficient of Determination)
# ============================================
# R² = SSR / Syy

def calculate_R_squared(SSR, Syy):
    """Calculate R²"""
    R_squared = SSR / Syy
    return R_squared


# ============================================
# 10. F-STATISTIC (Overall Significance)
# ============================================
# F = (SSR/d) / σ̂²

def calculate_F_statistic(SSR, d, variance):
    """Calculate F-statistic for overall model significance"""
    F = (SSR / d) / variance
    return F


def get_F_pvalue(F, d, n):
    """Get p-value for F-statistic"""
    df1 = d
    df2 = n - d - 1
    p_value = stats.f.sf(F, df1, df2)  # survival function
    return p_value


# ============================================
# 11. COVARIANCE MATRIX
# ============================================
# c = (X^T X)^(-1) σ²

def calculate_covariance_matrix(X, variance):
    """Calculate covariance matrix for coefficients"""
    XTX_inv = np.linalg.inv(X.T @ X)
    cov_matrix = XTX_inv * variance
    return cov_matrix


# ============================================
# 12. T-STATISTIC (Individual Coefficients)
# ============================================
# t = β̂i / (σ̂√cii)

def calculate_t_statistic(beta_i, std_dev, c_ii):
    """Calculate t-statistic for individual coefficient"""
    t = beta_i / (std_dev * np.sqrt(c_ii))
    return t


def get_t_pvalue_twosided(t, n, d):
    """Get two-sided p-value for t-statistic"""
    df = n - d - 1
    cdf = stats.t.cdf(t, df)
    sf = stats.t.sf(t, df)
    p_value = 2 * min(cdf, sf)
    return p_value


# ============================================
# 13. CONFIDENCE INTERVAL (for coefficients)
# ============================================
# CI = β̂i ± tα/2 · σ̂ · √cii

def calculate_confidence_interval(beta_i, std_dev, c_ii, alpha, n, d):
    """Calculate confidence interval for coefficient"""
    df = n - d - 1
    t_critical = stats.t.ppf(1 - alpha/2, df)
    margin = t_critical * std_dev * np.sqrt(c_ii)
    CI_lower = beta_i - margin
    CI_upper = beta_i + margin
    return CI_lower, CI_upper


# ============================================
# 14. PEARSON CORRELATION
# ============================================
# r = Cov(Xa,Xb) / √(Var(Xa)Var(Xb))

def calculate_pearson(X_a, X_b):
    """Calculate Pearson correlation between two variables"""
    r, p_value = stats.pearsonr(X_a, X_b)
    return r, p_value


# ============================================
# 15. PREDICTIONS
# ============================================
# Ŷ = Xb

def predict(X, b):
    """Make predictions using coefficients"""
    Y_predicted = X @ b
    return Y_predicted


# ============================================
# EXAMPLE USAGE WORKFLOW
# ============================================

def example_workflow():
    """
    Example of how to use these formulas together
    """
    # Assume X and Y are your data
    # X should include a column of 1s for intercept
    
    # Step 1: Calculate coefficients
    b = calculate_coefficients(X, Y)
    
    # Step 2: Get n and d
    n, d = get_n_and_d(X)
    
    # Step 3: Make predictions
    Y_pred = predict(X, b)
    
    # Step 4: Calculate SSE
    SSE = calculate_SSE(Y, Y_pred)
    
    # Step 5: Calculate variance and std dev
    variance = calculate_variance(SSE, n, d)
    std_dev = calculate_std_dev(variance)
    
    # Step 6: Calculate RMSE
    RMSE = calculate_RMSE(SSE, n)
    
    # Step 7: Calculate Syy and SSR
    Syy = calculate_Syy(Y)
    SSR = calculate_SSR(Syy, SSE)
    
    # Step 8: Calculate R²
    R_squared = calculate_R_squared(SSR, Syy)
    
    # Step 9: F-test for overall significance
    F = calculate_F_statistic(SSR, d, variance)
    F_pvalue = get_F_pvalue(F, d, n)
    
    # Step 10: Covariance matrix for individual tests
    cov_matrix = calculate_covariance_matrix(X, variance)
    
    # Step 11: t-test for each coefficient
    for i in range(len(b)):
        c_ii = cov_matrix[i, i]
        t = calculate_t_statistic(b[i], std_dev, c_ii)
        t_pvalue = get_t_pvalue_twosided(t, n, d)
        
        # Step 12: Confidence interval for each coefficient
        alpha = 0.05  # for 95% CI
        CI_lower, CI_upper = calculate_confidence_interval(
            b[i], std_dev, c_ii, alpha, n, d
        )
    
    # Step 13: Pearson correlation between features
    # Example: correlation between first two features
    r, p = calculate_pearson(X[:, 0], X[:, 1])
    
    return


# ============================================
# NOTES
# ============================================
"""
Key reminders:
1. X must include intercept column (column of 1s)
2. d = number of features (NOT including intercept)
3. n = number of data points (rows)
4. All these functions use numpy arrays
5. For categorical variables, encode them first (one-hot encoding)
"""