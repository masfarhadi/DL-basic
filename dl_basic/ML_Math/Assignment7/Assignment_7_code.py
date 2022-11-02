# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 10:58:46 2020

@author: farhadi
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:,.3g}'.format  # write numbers in scientfic format

# define input data: part of sine function with random noise
x = np.array([i*np.pi/180 for i in range(90,330,4)])
np.random.seed(12345)
y = np.sin(x) + np.random.normal(0,0.2,len(x))
data = pd.DataFrame(np.column_stack([x,y]),columns=['x','y'])

# plot input data
plt.figure(figsize=[12,8])
plt.plot(data['x'],data['y'],'.')
plt.title("Noisy data")
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# expand data with powers of x up to x^15
max_degree = 15
for i in range(2,max_degree+1):
    colname = 'x_%d'%i
    data[colname] = data['x']**i
print(data.head())

# linear regression
from sklearn.linear_model import LinearRegression
def linear_regression(data, max_degree):
    
    list_of_predictors = []                             # create empty list of predictors
    list_of_coefficients = []                           # create empty list of coefficients
    predictors=['x']
    for degree in range(1, max_degree+1):               # create predictions for all polynomials with degrees 1 to max_degree
    
        if degree >=2:                                  # extend predictors with higher powers of x
            predictors.extend(['x_{}'.format(degree)])

        lin_reg = LinearRegression(normalize=True)      # linear regression
        lin_reg.fit(data[predictors],data['y'])
        y_pred = lin_reg.predict(data[predictors])
        list_of_predictors.append(y_pred)
        
        rss = sum((y_pred-data['y'])**2)                # residual sum of squares
        result = [rss]
        result.extend([lin_reg.intercept_])             # intercept, i.e. coeff. of x^0
        result.extend(lin_reg.coef_)                    # coefficents
        list_of_coefficients.append(result)
    
    return list_of_predictors, list_of_coefficients     # return predictors and coefficients (including rss)

# plot linear regression results for a set of polynomial degrees from 1 to 15

degrees = [1,2,3,4,5,7,9,12,15]
list_of_predictors, list_of_coefficients = linear_regression(data,max_degree)
plt.figure(figsize=[12,8])
for i,degree in enumerate(degrees):
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    plt.plot(data['x'],list_of_predictors[degree-1])
    plt.plot(data['x'],data['y'],'.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Polynomial degree {}'.format(degree))  
    
# print list of coefficients

col = ['RSS','Coeff. of x^0'] + ['Coeff. of x^{}'.format(i) for i in range(1,max_degree+1)]
ind = ['Polynomial degree {}'.format(i) for i in range(1,max_degree+1)]
coeff_table = pd.DataFrame(index=ind, columns=col)
for i,entry in enumerate(list_of_coefficients):
    coeff_table.iloc[i,0:i+3] = entry

coeff_table

# ridge regression

from sklearn.linear_model import Ridge

def ridge_regression(data, max_degree, alpha_ridge):
    
    list_of_predictors = []
    list_of_coefficients = []
    
    predictors=['x']
    for degree in range(2, max_degree+1):
        if degree >=2:
            predictors.extend(['x_{}'.format(degree)])
   
    for alpha in alpha_ridge:
# to do
        ridge_reg = Ridge(alpha, normalize=True,)      # rigid regression
        ridge_reg.fit(data[predictors],data['y'])
        y_pred = ridge_reg.predict(data[predictors])
        list_of_predictors.append(y_pred)
        
        rss = sum((y_pred-data['y'])**2)                # residual sum of squares
        result = [rss]
        result.extend([ridge_reg.intercept_])             # intercept
        result.extend(ridge_reg.coef_)                    # coefficents
        list_of_coefficients.append(result) 
# end to do    
    return list_of_predictors, list_of_coefficients

# plot ridge regression results for max_degree and a set of values for alpha
alpha_ridge = [1e-18, 1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1, 10, 100]
list_of_predictors_ridge, list_of_coefficients_ridge = ridge_regression(data,max_degree,alpha_ridge)
plt.figure(figsize=[12,8])
for i,alpha in enumerate(alpha_ridge):
    # plot
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    plt.plot(data['x'],list_of_predictors_ridge[i])
    plt.plot(data['x'],data['y'],'.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('alpha = %.3g'%alpha)
    
# print list of coefficients

col = ['RSS','Coeff. of x^0'] + ['Coeff. of x^{}'.format(i) for i in range(1,max_degree+1)]
ind = ['alpha: {}'.format(alpha_ridge[i-1]) for i in range(1,len(alpha_ridge)+1)]
coeff_table_ridge = pd.DataFrame(index=ind, columns=col)
for i,entry in enumerate(list_of_coefficients_ridge):
    coeff_table_ridge.iloc[i,0:max_degree+2] = entry

coeff_table_ridge