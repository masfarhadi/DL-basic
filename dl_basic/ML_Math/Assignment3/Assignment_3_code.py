# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:34:48 2020

@author: farhadi
"""

from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt

#def poisson_func(x, lam):
#    p = lam**(x)*np.exp(-lam)/np.math.factorial(x)
#    return p

def E_step(x, lam, alpha):
    # initialization 
    K = np.size(lam)
    n = np.size(x)
    temp = np.zeros((n,K)) 
    r = np.zeros((n,K))
    
    # loop to calcualte r matrix (slide 14) [above equation]
    for kk in range(K):
        temp[:,kk] = alpha[kk]*poisson.pmf(x,mu=lam[kk])
       
    r = temp/np.sum(temp,axis=1).reshape(-1,1)
    return r

def M_step(x, r):
    # calculate alpha and lambda  [above equations](similar to slide 19)
    n = r.shape[0]
    alpha = 1/n*np.sum(r,axis=0)
    lam = np.matmul(x,r)/np.sum(r,axis=0)
    return lam, alpha
    
def log_liklihood(x, lam, alpha):
    # L = x*np.log(lam)-lam-np.log(np.math.factorial(x)) 
    # intialization 
    K = np.size(lam)
    n = np.size(x)
    p = np.zeros((n,K))
    
    # loop to calculate probability k slide (slide 11) 
    for jj in range(K):
        p[:, jj] = alpha[jj]*poisson.pmf(x,lam[jj])
      
    # log liklihood calculation (slide 11)
    log_like = np.sum(np.log(np.sum(p,axis=1)),axis=0)
    return log_like


def iterative_process(x, lam, alpha):
    # initialization
    err = 1
    L= list()
    L.append(0) 
    # loop to reach the threshold
    while err>1e-5:
        L.append(log_liklihood(x, lam, alpha))
        err = np.abs(L[-1]-L[-2])
        r = E_step(x, lam, alpha)
        lam, alpha = M_step(x,r)
    
    
    return L, lam, alpha, r


data = np.loadtxt(open("cnvdata.csv", "rb"), delimiter=",", skiprows=1)

plt.hist(data,bins=50)
plt.title('Data distribution')
plt.ylabel('$p(X)$')
plt.xlabel('$X$')
plt.show()

# first 
K = 4
alpha = 1/K*np.ones(K)
x = data.copy()
lam = np.array([10,50,100,150])

L, lam, alpha, r = iterative_process(x, lam, alpha)

plt.figure()
plt.hist(x,bins=50,density=True)
plt.title('Data distribution')
plt.ylabel('$p(X)$')
plt.xlabel('$X$')
plt.show()

t = np.arange(np.size(x))
pt = 0
for pp in range(K):
    pt += alpha[pp]*poisson.pmf(t,lam[pp])
    
plt.plot(t,pt,linewidth=3)
# Second
K = 6
alpha = 1/K*np.ones(K)
lam = np.random.rand(K)*250

L, lam, alpha, r = iterative_process(x, lam, alpha)

plt.figure()
plt.hist(x,bins=50,density=True)
plt.title('Data distribution')
plt.ylabel('$p(X)$')
plt.xlabel('$X$')
plt.show()

t = np.arange(np.size(x))
pt = 0
for pp in range(K):
    pt += alpha[pp]*poisson.pmf(t,lam[pp])
    
plt.plot(t,pt,linewidth=3)

