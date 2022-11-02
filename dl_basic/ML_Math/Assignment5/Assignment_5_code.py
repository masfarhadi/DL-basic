# -*- coding: utf-8 -*-
"""
Created on Mon May 18 02:06:55 2020

@author: farhadi
"""

# Some code that should help you. Nothing to do here

import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
# Create empty list that should store feature values.
x=[]
# Create empty list that should store labels.
y=[]
# Randomly create one point within unit circle with label y=1, store features and labels in the corresponding lists.
length_1 = np.random.uniform(0, 1)
angle_1 = np.pi * np.random.uniform(0, 2)
x.append([np.sqrt(length_1) * np.cos(angle_1), np.sqrt(length_1) * np.sin(angle_1)])
y.append(1)
# Randomly create another point within unit circle with label y=-1, store features and labels it in the corresponding lists.
length_2 = np.random.uniform(0, 1)
angle_2 = np.pi * np.random.uniform(0, 2)
x.append([np.sqrt(length_2) * np.cos(angle_2), np.sqrt(length_2) * np.sin(angle_2)])
y.append(-1)
print(x,y)
# Code for linear SVC containing only the previously created points as data, also outputs corresponding margin.
clf = svm.SVC(kernel='linear',C=10)
clf.fit(x,y)
margin=2/np.linalg.norm(clf.coef_)
print("Margin of SVM =",margin)

def create_data(k):
    r = np.random.uniform(0, 1, size=k)
    phi = np.pi * np.random.uniform(0, 2, size=k)
    a = np.array([np.sqrt(r) * np.cos(phi)])
    b = np.array([np.sqrt(r) * np.sin(phi)])
    X = np.concatenate((a.T, b.T),axis=1)
    y = [+1 if elem>0 else -1 for elem in a.ravel() ]
    color=['red' if elem2==+1 else 'blue' for elem2 in y ] #  ['red' if elem>0 else 'blue'  for elem in a.ravel() ]
    
    if False:
        fig = plt.figure()
        ax = fig.add_subplot()
        circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None')
        ax.add_patch(circ)
        ax.scatter(X[:,0], X[:,1], c=color,label='the data')
        ax.axis('equal')
        plt.show()
    
    return X,y

def compute_svm(k):
    X_list = []
    y_list = []
    margin_list = []
    for j in range(10,k+1):
        X,y = create_data(j)
        X_list.append(X)
        y_list.append(y)
        clf = svm.SVC(kernel='linear',C=10)
        clf.fit(X,y)
        margin=2/np.linalg.norm(clf.coef_)
        margin_list.append(margin)
    
    # print("Margins of SVM =",margin_list)
    
    return margin_list

k = 1000
n_points = np.arange(10,k+1)
gamma = np.asarray(compute_svm(k))
R = 1
dvc = (R**2)/(gamma**2)
plt.figure()
plt.plot(n_points, dvc)
plt.xlabel('Number of data points ')
plt.ylabel('VC dimension ')
plt.grid()
plt.figure()
plt.plot(n_points, gamma)
plt.xlabel('Number of data points ')
plt.ylabel('Margin')
plt.grid()


    
        
    
    