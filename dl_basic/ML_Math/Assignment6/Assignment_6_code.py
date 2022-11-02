# -*- coding: utf-8 -*-
"""
Created on Thu May 28 12:02:08 2020

@author: farhadi
"""

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from IPython.display import HTML, Image

#%matplotlib inline
sns.set()

#Generate dataset
from sklearn.datasets.samples_generator import make_circles
X, y = make_circles(100, factor=.1, noise=.1)

###################################
# Begin your code
plt.figure()
color= ['red' if elem==1 else 'blue' for elem in y ]
plt.scatter(X[:,0], X[:,1], c=color,label='data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('2D Input dataset')

fig = plt.figure() 
ax1 = Axes3D(fig) # fig.add_subplot(111, projection='3d')
ax1.scatter(np.sqrt(2)*X[:,0]*X[:,1], X[:,0]**2, X[:,1]**2,c=color, label='data')
ax1.set_xlabel('X Label')
ax1.set_ylabel('Y Label')
ax1.set_zlabel('Z Label')
# ax1.view_init(elev=30., azim=-10)
ax1.text2D(0.05, 0.95,'$\Phi_1$ feature space', transform=ax1.transAxes)


fig = plt.figure() 
ax2 = Axes3D(fig) # fig.add_subplot(111, projection='3d')
ax2.scatter(X[:,0], X[:,1], np.exp(-X[:,0]**2-X[:,1]**2),c=color, label='data')
ax2.set_xlabel('X Label')
ax2.set_ylabel('Y Label')
ax2.set_zlabel('Z Label')
#ax2.view_init(elev=30., azim=-10)
ax2.text2D(0.05, 0.95,'$\Phi_2$ feature space', transform=ax1.transAxes)

# End your code
###################################