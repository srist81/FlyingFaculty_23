# -*- coding: utf-8 -*-
"""
Created on Wed May  3 10:42:59 2023

@author: Rist
"""

import im_cl_utils as ut  #utility functions
import numpy as np
np.random.seed(121) #for debugging to get each time same answer


# ut.prep_mnist_mini()

data_file = 'data/mnist_mini.pkl.gz'

#load data
(X_train, y_train),(X_test,y_test) = ut.load_mnist_data(data_file)

# make sure to shuffle data
X_train, y_train = ut.shuffle(X_train, y_train)
X_test, y_test = ut.shuffle(X_test, y_test)

#plot data
Nshow = 8
ut.plot_images(X_train[0:Nshow],y_train[0:Nshow])



N,H,W = X_train.shape
Ntest,H,W = X_test.shape
max_pixel_value = float(255)


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=100,random_state = 1,multi_class='multinomial')
lr.fit(X_train,y_train)


# Test the classifier on the testing data
score_train = lr.score(X_train, y_train)
score_test = lr.score(X_test, y_test)
print(f"Train accuracy: {score_train:.2f}") # Print the classification accuracy
print(f"Test accuracy: {score_test:.2f}") # Print the classification accuracy

#show correct and incorrect classified numbers
Nshow = 8
    
y_pred = lr.predict(X_test) #predicted labels
ind_true = np.where(y_test == y_pred)[0]
ind_false = np.where(y_test != y_pred)[0]

#correct classified images
Nplot = np.minimum(len(ind_true),Nshow) #how many images shall be shown
xs = X_test[ind_true[0:Nplot]].reshape(Nplot,H,W)
ys = y_test[ind_true[0:Nplot]]
ys_pred = y_pred[ind_true[0:Nplot]]
ut.plot_images(xs,ys,ys_pred) # show training images and predicted labels for them

#incorrect classified images
Nplot = np.minimum(len(ind_false),Nshow) #how many images shall be shown
xs = X_test[ind_false[0:Nplot]].reshape(Nplot,H,W)
ys = y_test[ind_false[0:Nplot]]
ys_pred = y_pred[ind_false[0:Nplot]]
ut.plot_images(xs,ys,ys_pred) # show training images and predicted labels for them
    
    
    
#alternatively use SGD for large datasets
from sklearn.linear_model import SGDClassifier
lr_sgd = SGDClassifier(loss = 'log_loss')

lr_sgd.fit(X_train,y_train)

# Test the classifier on the testing data
score_train = lr_sgd.score(X_train, y_train)
score_test = lr_sgd.score(X_test, y_test)
print(f"Train accuracy: {score_train:.2f}") # Print the classification accuracy
print(f"Test accuracy: {score_test:.2f}") # Print the classification accuracy





