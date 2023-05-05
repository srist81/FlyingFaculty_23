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
X_train = X_train/ max_pixel_value
X_test = X_test / max_pixel_value

#add noise
#X_train = X_train*1.0 +  0.5*np.random.randn(N,H,W) 
#X_test = X_test*1.0 + 0.5*np.random.randn(len(X_test),H,W) 

#plot data
Nshow = 8
ut.plot_images(X_train[0:Nshow].reshape(-1,H,W), y_train[0:Nshow])

# Make sure images have shape (28, 28, 1)
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

#make target cathegorical
y_train_onehot = keras.utils.to_categorical(y_train)
y_test_onehot = keras.utils.to_categorical(y_test)

num_classes = y_train_onehot.shape[1]

model = keras.Sequential(
    [
        keras.Input(shape=(H,W,1)),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()
LR = 0.001
DR = 1e-6
M = 0.9
sgd_optimizer = keras.optimizers.SGD(learning_rate = LR,decay = DR,momentum = M)
model.compile(optimizer = sgd_optimizer,
              loss = 'categorical_crossentropy',metrics = ["accuracy"])

# alternative
#model.compile(optimizer = 'adam',
#              loss = 'categorical_crossentropy',metrics = ["accuracy"])

#use early stopping
callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                         patience=10)

history = model.fit(X_train,y_train_onehot,
                    batch_size =32,
                    epochs = 20,
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