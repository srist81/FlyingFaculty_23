# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 09:18:53 2023

@author: Rist
"""

import im_cl_utils as ut  #utility functions
import numpy as np
import tensorflow.keras as keras


np.random.seed(121) #for debugging to get each time same answer


# ut.prep_mnist_mini()

data_file = 'data/mnist_mini.pkl.gz'

#load data
(X_train, y_train),(X_test,y_test) = ut.load_mnist_data(data_file)

# make sure to shuffle data
X_train, y_train = ut.shuffle(X_train, y_train)
X_test, y_test = ut.shuffle(X_test, y_test)

N,H,W = X_train.shape
max_pixel_value = float(255)

# normalize pixel intensities and reshape
X_train = X_train.reshape((-1,H*W)) / max_pixel_value
X_test = X_test.reshape((-1,H*W)) / max_pixel_value

#make target cathegorical
y_train_onehot = keras.utils.to_categorical(y_train)
y_test_onehot = keras.utils.to_categorical(y_test)

#add noise
#X_train = X_train*1.0 +  0.5*np.random.randn(X_train.shape[0],X_train.shape[1]) 
#X_test = X_test*1.0 + 0.5*np.random.randn(X_test.shape[0],X_test.shape[1]) 



#plot data
Nshow = 8
ut.plot_images(X_train[0:Nshow].reshape(-1,H,W),         y_train[0:Nshow])


def setup_nn(dim_layers,LR,activations,DR,M):
    #creates a nn with 2 hidden layers, and last softmax layer
    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(
            units = dim_layers[0],
            input_dim = X_train.shape[1],
            activation = activations[0]          
            ))

    model.add(
        keras.layers.Dense(
            units = dim_layers[1],
            input_dim = dim_layers[0],
            activation = activations[1]         
            ))

    model.add(
        keras.layers.Dense(
            units = y_train_onehot.shape[1],
            input_dim = dim_layers[1],
            activation = activations[2]         
            ))
    
    sgd_optimizer = keras.optimizers.SGD(learning_rate = LR,decay = DR,momentum = M)

    model.compile(optimizer = sgd_optimizer,
                  loss = 'categorical_crossentropy',metrics = ["accuracy"])
    return model 




dim_layers = [128,128]
activations = ['relu','tanh','softmax']
LR = 0.005
DR = 1e-8 #decay rate how 
M = 0.9 #momentum for gradient between [0,1]
model = setup_nn(dim_layers,LR,activations,DR,M)

#use early stopping
callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                         patience=10)

history = model.fit(X_train,y_train_onehot,
                    batch_size =64,
                    epochs = 200,
                    verbose = 1, #print progress
                    validation_split = 0.2,
                    callbacks=[callback])


ut.plot_accu(history.history)
ut.plot_loss(history.history)


p_test_pred = model.predict(X_test,verbose = 0)
y_test_pred = np.argmax(p_test_pred, axis=-1)
print('test accuracy is:',np.sum(y_test_pred == y_test)/len(y_test))

#using model.evaluate
test_loss , test_acc = model.evaluate(X_test,y_test_onehot,verbose = 0)
print('\nTest loss: {:.6f}  accuracy: {:.6f} '.format(test_loss, test_acc))


ind_true = np.where(y_test == y_test_pred)[0]
ind_false = np.where(y_test != y_test_pred)[0]

#correct classified images
Nplot = np.minimum(len(ind_true),Nshow) #how many images shall be shown
xs = X_test[ind_true[0:Nplot]]
ys = y_test[ind_true[0:Nplot]]
ys_pred = y_test_pred[ind_true[0:Nplot]]
ut.plot_images(xs.reshape(-1,H,W),ys,ys_pred) # show training images and predicted labels for them

#incorrect classified images
Nplot = np.minimum(len(ind_false),Nshow) #how many images shall be shown
xs = X_test[ind_false[0:Nplot]]
ys = y_test[ind_false[0:Nplot]]
ys_pred = y_test_pred[ind_false[0:Nplot]]
ut.plot_images(xs.reshape(-1,H,W),ys,ys_pred) # show training images and predicted labels for them