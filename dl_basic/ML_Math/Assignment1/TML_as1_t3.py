# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:18:08 2020

@author: farhadi
"""
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import random
from mpl_toolkits import mplot3d


# load the data
data = np.genfromtxt('DataSet1b.csv', delimiter=',', skip_header=1)
# estimate the parameter according to the previously derived formula

# Begin your code
x = data 
n = np.size(x) 
lam = n/np.sum(x)

f = lam*np.exp(-lam*x)

plt.figure()
plt.scatter(x,f,marker='.', color='red', label='ML')
H = plt.hist(x,int(n/10), density=1, label='Hist')
plt.legend()