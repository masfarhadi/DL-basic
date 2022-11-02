# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:25:12 2020

@author: farhadi
"""

import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt


def Gauss_dist(n,mu=5,var=4):
    sigma = np.sqrt(var)  # standard deviation
    x = np.random.normal(mu, sigma, n)   
    return x


def est_A(x):
    n = np.shape(x)[0]
    var_hat_A = 1/(n-1)*sum((x.mean()-x)**2)
    return var_hat_A


def est_B(x):
    n = np.shape(x)[0]
    var_hat_B = 1/(n)*sum((x.mean()-x)**2)
    return var_hat_B

def estimators(n):
    var_hat_A = np.zeros(10000)
    var_hat_B = np.zeros(10000)
    # repeat the random distribution generation and variance estimators calculation
    for ii in range(10000):
        x = Gauss_dist(n)
        var_hat_A[ii] = est_A(x)
        var_hat_B[ii] = est_B(x)
    
    # calculate mean and variance of estimators
    mean_A = np.mean(var_hat_A)
    var_A = np.var(var_hat_A)
    mean_B = np.mean(var_hat_B)
    var_B = np.var(var_hat_B)
    return mean_A, var_A, mean_B, var_B

n = 10
mean_A, var_A, mean_B, var_B = estimators(n)
print('for n=10 the mean and variance of estimators are:')
print ('mean_A = %.6f'%mean_A, 'and variance_A = %.6f'%var_A)
print ('mean_B = %.6f'%mean_B, 'and variance_B = %.6f'%var_B)

n=1000
mean_A, var_A, mean_B, var_B = estimators(n)
print('for n=1000 the mean and variance of estimators are:')
print ('mean_A = %.6f'%mean_A, 'and variance_A = %.6f'%var_A)
print ('mean_B = %.6f'%mean_B, 'and variance_B = %.6f'%var_B)

