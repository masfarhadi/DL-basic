# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:34:48 2020

@author: farhadi
"""
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib
import matplotlib.pyplot as plt

# Complete the functions with your code

def logistic_gradient(w, x, y):
    ######################################
    # Begin your code
    sigmoid = lambda z:1/(1 + np.exp(-z))
    
    y = y.reshape(-1,1)
    z = np.dot(x, w)
    g = sigmoid(z)
    gradient = np.dot(x.T, (g- y)) 
    return gradient
    # End your code
    ######################################
    
def cost(w, x, y):
    ######################################
    # Begin your code
    sigmoid = lambda z:1/(1 + np.exp(-z))
    
    y = y.reshape(-1,1)
    z = np.dot(x, w)
    g = sigmoid(z)
    cross_ent_loss = - np.sum(y * np.log(g) + (1 - y) * np.log(1 - g))
    return cross_ent_loss
    # End your code
    ######################################
    
def numerical_gradient(w, x, y):
    ######################################
    # Begin your code
    eps = 1e-4
    num_gradient = np.zeros(w.shape)
    for ii in range(w.size):
        e = np.zeros(w.shape)
        e[ii]=eps
        num_gradient[ii] =(cost(w+e, x, y)-cost(w-e, x, y))/(2*eps)
    return num_gradient
    # End your code
    ######################################
    
    
# Generate random data matrix and compare outputs

#################################################
# Begin your code
x = np.random.normal(size=(5, 10))
y = np.random.randint(2, size=5)
w = np.random.normal(size=(10, 1))
log_gradient = logistic_gradient(w, x, y)
num_gradient = numerical_gradient(w, x, y)
outputs_mse = np.mean((log_gradient-num_gradient)**2)
print('output MSE =', outputs_mse )

# End your code
#################################################




# Complete the functions with your code

def fitLogRegModel(x_train, y_train, eta=1e-1, max_iter=100000):
    ######################################
    # Begin your code
    w = np.random.uniform(-1,1,size=(x_train.shape[1], 1))
    # Use of break statement inside the loop
    loss=[1]
    for cnt in range(max_iter):
        w = w - eta*logistic_gradient(w, x_train, y_train)
        loss.append(cost(w, x_train, y_train))
        if cnt % 1000 == 0:
            print( 'iter =', cnt, 'loss =', loss[-1])
        if np.abs(loss[-1]-loss[-2])<eta:
            break
    
    print("Final loss = ",loss[-1], ' iterations =  ',cnt)
    return w
    # End your code
    ######################################

def predictLogReg(w, x_pred):
    ######################################
    # Begin your code
    sigmoid = lambda z:1/(1 + np.exp(-z))
    
    z = np.dot(x_pred, w)
    y_pred = np.squeeze(sigmoid(z)) # np.rint()
    return y_pred
    # End your code
    ######################################


## test code 
wt = fitLogRegModel(x, y)
yt = predictLogReg(wt, x)
MSEt = np.mean((y-np.rint(yt))**2)



# Nothing to do here

from sklearn.utils import shuffle
# Read data, split into X(features) and y(labels)
Z = np.genfromtxt('DataSetLR.csv', delimiter=',',skip_header=1)
X, y = Z[:,:-1], Z[:,-1]
y[:] = [0 if x==-1 else x for x in y]
# Plot data distribution
color= ['red' if elem==1 else 'blue' for elem in y ]
plt.scatter(X[:,0], X[:,1], c=color,label='the data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Complete dataset')
# Split into test and training set
X_train=X[:np.int(X.shape[0]/2)]
X_test=X[np.int(X.shape[0]/2):]
y_train=y[:np.int(len(y)/2)]
y_test=y[np.int(len(y)/2):]


w_train = fitLogRegModel(X_train, y_train)
pred_train = predictLogReg(w_train, X_train)
pred_test = predictLogReg(w_train, X_test)

# Nothing to do here
# Plot training and test dataset
# Plot predictions for training and test dataset

fig = plt.figure()
fig = plt.figure(figsize = (12,10))
plt.subplot(2, 2, 1)
color= ['red' if elem>0.5 else 'blue' for elem in y_train ]
plt.scatter(X_train[:,0], X_train[:,1], c=color,label='the data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Training dataset');

plt.subplot(2, 2, 2)
color= ['red' if elem>0.5 else 'blue' for elem in pred_train ]
plt.scatter(X_train[:,0], X_train[:,1], c=color,label='the data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Training dataset - predictions');

plt.subplot(2, 2, 3)
color= ['red' if elem>0.5 else 'blue' for elem in y_test ]
plt.scatter(X_test[:,0], X_test[:,1], c=color,label='the data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Test dataset');

plt.subplot(2, 2, 4)
color= ['red' if elem>0.5 else 'blue' for elem in pred_test ]
plt.scatter(X_test[:,0], X_test[:,1], c=color,label='the data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Test dataset - predictions');


from sklearn.metrics import accuracy_score, balanced_accuracy_score

y_pred_train = np.rint(pred_train)
score_train = accuracy_score(y_train, y_pred_train)
balanced_score_train = balanced_accuracy_score(y_train, y_pred_train)
print('Accuracy for training set = ', score_train, 'and Balanced Accuracy = ', balanced_score_train)

y_pred_test = np.rint(pred_test)
score_test = accuracy_score(y_test, y_pred_test)
balanced_score_test = balanced_accuracy_score(y_test, y_pred_test)
print('Accuracy for test set = ', score_test, 'and Balanced Accuracy = ', balanced_score_test)


from sklearn.metrics import roc_curve, auc
# Compute ROC curve and ROC area for each class
fpr, tpr, thresholds = roc_curve(y_test, pred_test)
roc_auc = auc(fpr, tpr)
print('Area Under the Curve (AUC) = %0.2f' % roc_auc)

plt.figure()
lw = 1
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()