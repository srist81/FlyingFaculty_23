# -*- coding: utf-8 -*-
"""
Created on Wed May  3 08:13:32 2023

Iris Dataset Example

@author: Rist
"""


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


def plot_decision_regions(X,y,classifier,resolution = 0.02,title = ''):
    #makes a plot of decision boundary and regions
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(set(y))])
    
    #plot decision boundary
    x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1
    x2_min, x2_max = X[:,1].min()-1, X[:,1].max()+1
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
                          np.arange(x2_min,x2_max,resolution))
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha = 0.4,cmap =cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    
    #plot all objects
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl,0],y=X[y == cl,1],
                    alpha = 0.8, c = colors[idx],
                    marker = markers[idx],label = cl,
                    edgecolor ='black')
    plt.title(title)
    plt.show()
        
    


from sklearn import datasets

#load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

print('classes:', np.unique(y))

#split data into training and test set
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,
                                                 random_state=0,
                                                 stratify = y)


#visualize (training) data

# Create a figure with subplots for each feature
fig, axs = plt.subplots(nrows=1, ncols=X.shape[1], figsize=(16, 4))
# Loop over each feature and create a boxplot
for i in range(X_train.shape[1]):
    for j in range(len(set(y))):
        #axs[i].boxplot(X[:, i])
        axs[i].boxplot(X_train[y_train == j, i], positions=[j], widths=0.6)
    axs[i].set_title(iris.feature_names[i])
    axs[i].set_xticks(range(len(set(y))), iris.target_names)

fig.suptitle('Boxplots of Iris Features')
plt.show()



# make SVM classifier
from sklearn.svm import SVC

svm = SVC(kernel = 'linear',C = 1,random_state=1)
svm.fit(X_train,y_train)

# Test the classifier on the testing data
score = svm.score(X_test, y_test)
print(f"Classification accuracy: {score:.2f}") # Print the classification accuracy



#Reduce numer of features
which_features = [1,2]
X_train,X_test,y_train,y_test = train_test_split(X[:,which_features],y,test_size = 0.3,
                                                 random_state=0,
                                                 stratify = y)


#create a 2D plot of 2 given features 
labels = iris.target_names # name of species we want to classify
colors = ['red', 'blue', 'green']

plt.figure(figsize=(10, 8))
for i in range(len(labels)):
    plt.scatter(X_train[y_train == i, 0], X_train[y_train == i, 1], c=colors[i], label=labels[i])
plt.xlabel(iris.feature_names[which_features[0]],fontsize = 16)
plt.ylabel(iris.feature_names[which_features[1]],fontsize = 16)
plt.title('Iris Dataset')
plt.legend(fontsize = 14) # Add the legend
plt.show()


#train the model
svm = SVC(kernel = 'linear',C = 1,random_state=1)
svm.fit(X_train,y_train)

# Test the classifier on the testing data
score_train = svm.score(X_train, y_train)
score_test = svm.score(X_test, y_test)
print(f"Train accuracy: {score_train:.2f}") # Print the classification accuracy
print(f"Test accuracy: {score_test:.2f}") # Print the classification accuracy



#what is the best value of C ???

from sklearn.model_selection import GridSearchCV

# Define the parameter grid to search over
param_grid = {
    'C': np.linspace(0.001,1,10)
}

# Create a SVC object
svm = SVC(kernel = 'linear',random_state=1)

# Create a GridSearchCV object with 5-fold cross-validation
grid_search = GridSearchCV(svm, param_grid, cv=5)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Print the best value of alpha found by GridSearchCV
print(grid_search.best_params_['C'])

#train the model
svm = SVC(kernel = 'linear',C = grid_search.best_params_['C'],random_state=1)
svm.fit(X_train,y_train)

# Test the classifier on the testing data
score_train = svm.score(X_train, y_train)
score_test = svm.score(X_test, y_test)
print(f"Train accuracy: {score_train:.2f}") # Print the classification accuracy
print(f"Test accuracy: {score_test:.2f}") # Print the classification accuracy

plot_decision_regions(X_train,y_train,svm,title = 'training set')
plot_decision_regions(X_test,y_test,svm,title = 'test set')



#alternatively use SGD for large datasets
from sklearn.linear_model import SGDClassifier
svm_sgd = SGDClassifier(loss = 'hinge')

svm_sgd.fit(X_train,y_train)

# Test the classifier on the testing data
score_train = svm_sgd.score(X_train, y_train)
score_test = svm_sgd.score(X_test, y_test)
print(f"Train accuracy: {score_train:.2f}") # Print the classification accuracy
print(f"Test accuracy: {score_test:.2f}") # Print the classification accuracy

plot_decision_regions(X_train,y_train,svm_sgd,title = 'training set')
plot_decision_regions(X_test,y_test,svm_sgd,title = 'test set')



# excercise
# make the cross validation for the svm_sgd model



