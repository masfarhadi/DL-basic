# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:41:56 2020

@author: farhadi
"""

import sklearn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import random
from mpl_toolkits import mplot3d



#read data, split into X(features) and y(labels)
Z = np.genfromtxt('DataSet1a.csv', delimiter=',', skip_header=1)
X, y = Z[:,:-1], Z[:,-1]
#further split features according to labels
Xpos=X[y==1]
Xneg=X[y==-1]

# Begin your code
mu = np.mean(X, axis=0)
print('the mean values of X is: ',mu)
cov_mtx = np.cov(X[:,0], X[:,1])
print('the covariance martix is ',cov_mtx)
Xpos_size = Xpos.shape[0]
Xneg_size = Xneg.shape[0]
X_size = X.shape[0]
print('p(y=+1) = ',Xpos_size/X_size ,'\np(y=-1) = ',Xneg_size/X_size)


#read data, split into X(features) and y(labels)
Z = np.genfromtxt('DataSet1a.csv', delimiter=',', skip_header=1)
X, y = Z[:,:-1], Z[:,-1]
#further split features according to labels
Xpos=X[y==1]
Xneg=X[y==-1]

# Begin your code
mu_pos = np.mean(Xpos, axis=0)
print('the first mean values E(x|y=+1)=',mu_pos)
mu_neg = np.mean(Xneg, axis=0)
print('the second mean values E(x|y=-1)=',mu_neg)
cov_pos = np.cov(Xpos[:,0], Xpos[:,1])
print('the first covariance martix is ',cov_pos)
cov_neg = np.cov(Xneg[:,0], Xneg[:,1])
print('the second covariance martix is ',cov_neg)
Xpos_size = Xpos.shape[0]
Xneg_size = Xneg.shape[0]
X_size = X.shape[0]
p_pos = Xpos_size/X_size 
p_neg = Xneg_size/X_size
print('p(y=+1) = ',p_pos ,'\np(y=-1) = ',p_neg)


A = np.linalg.inv(cov_pos)-np.linalg.inv(cov_neg)
print('A=',A)
w = np.linalg.inv(cov_pos)@mu_pos-np.linalg.inv(cov_neg)@mu_neg
print('w=',w)
b = -0.5*mu_pos.T@np.linalg.inv(cov_pos)@mu_pos +\
    0.5*mu_neg.T@np.linalg.inv(cov_neg)@mu_neg -\
    0.5*np.log(np.linalg.det(cov_pos)) + 0.5*np.log(np.linalg.det(cov_neg)) +\
    np.log(p_pos) - np.log(p_neg)
print('b=',b)

plt.figure()
plt.scatter(Xpos[:, 0], Xpos[:, 1],marker ='+',label='y=+1')
plt.scatter(Xneg[:, 0], Xneg[:, 1],marker='*',label='y=-1' )
plt.legend()

X1 = np.sort(X[:,0])
X2 = np.sort(X[:,1])
XS = np.array([X1, X2])
g = -0.5*XS.T@A@XS + w.T@XS+b
xx, yy = np.meshgrid(X1,X2)
plt.contour(xx, yy, g, levels=[0])# 
# plt.contour3D(xx, yy, g, levels=[0])