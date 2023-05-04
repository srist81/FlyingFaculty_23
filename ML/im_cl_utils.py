# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 09:28:25 2023

@author: Rist
"""
import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math



def plot_images(X,labels=[],pred_labels=[]):
    #plots the images
    # X [NumImages,Nx,Ny] Nx,Ny are pixel dimensions
    # labels [NumImages] labels of images
    # pred_labels predicted labels of images
    X = np.squeeze(X)
    if len(X.shape) == 2: 
        num_images = 1
        X = X[np.newaxis,:,:]
    else:
        num_images = X.shape[0]
    num_rows = math.floor(math.sqrt(num_images))
    num_cols = math.ceil(num_images/num_rows)
    for i in range(num_images):
        reshaped_image = np.squeeze(X[i,:,:])
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(reshaped_image, cmap = cm.Greys_r)
        plt.axis('off')
        if len(labels) == num_images:
            plt.title('is '+ str(labels[i]))
        if len(pred_labels) == num_images:
            plt.title('pred ' + str(pred_labels[i]))
        if len(pred_labels) == num_images and len(labels) == num_images:
            plt.title('is '+ str(labels[i]) +' pred ' + str(pred_labels[i]))
    plt.show()

def filter_images(xs,ys,digit1,digit2):
    #picks only digit1 and digit2 from xs and ys 
    idx = np.logical_or(np.equal(ys,digit1) , np.equal(ys,digit2))
    xs = np.squeeze(xs[idx]) #squeeze out 1st dimension
    ys = ys[idx]
    return (xs,ys)


def shuffle(xs,ys):
    #shuffles data 
    indices = np.arange(0,len(ys))
    indices_shuffled = np.random.shuffle(indices) #mutated indizes
    xs = np.squeeze(xs[indices_shuffled]) #shuffled training data
    ys = np.squeeze(ys[indices_shuffled]) #shuffled training labels
    return xs,ys

def get_first_data(xs,ys,N):
    #returns first N samples of data
    N = np.minimum(N,len(ys))
    return xs[0:N],ys[0:N]

def load_mnist_data(filename):
    #loads a fraction of mnist data
    with gzip.open(filename, 'rb') as f:
         (X_train, y_train),(X_test,y_test) = pickle.load(f)
         return (X_train, y_train),(X_test,y_test)

def save_data(data,filename):
    #saves data to filename in compressed form
    with gzip.open(filename, 'wb') as f:
        pickle.dump(data, f)
        
def prep_mnist_mini():
    #function used to prepare the slice of mnist data
    # full mnist 60000 images, our data 8000 train, 2000 test
    from keras.datasets import mnist

    np.random.seed(121) #for debugging to get each time same answer

    (X_train, y_train),(X_test,y_test) = mnist.load_data()


    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    X_train, y_train = get_first_data(X_train, y_train,8000)
    X_test, y_test =  get_first_data(X_test, y_test,2000)
    
    data = ((X_train, y_train),(X_test,y_test))
    save_data(data,'data/mnist_mini.pkl.gz')
    
  
def inflate_X(X,N_border):
    #make border of zeros for images in order to avoid boundary effects
    
    N,H,W = X.shape
    N_border = 10
    X_inflated = np.zeros((N,H+2*N_border,W+2*N_border))
    X_inflated[:,N_border:N_border+H,N_border:N_border+W] = X

    return X_inflated
    

def move_X(X,dx,dy):
    #displace image data randomly

    X_moved = 0*X
    for idx,img in enumerate(X):
        #print(idx)
        H,W = img.shape
        
        try:
            
            ddy = np.random.randint(-dy,dy)
            stripe = np.zeros((np.abs(ddy),W))
            
            x_disp = img
            if ddy != 0:
                if ddy > 0: #move down
                    x_disp = np.vstack((stripe,img[0:-ddy,:]))
                else: #move up
                    x_disp = np.vstack((img[-ddy:,:],stripe))
                
            
            ddx = np.random.randint(-dx,dx)
            stripe = np.zeros((H,np.abs(ddx)))
            
            if ddx != 0:    
                if ddx > 0: #move right
                    x_disp = np.hstack((stripe,x_disp[:,0:-ddx]))
                else: #move left
                    x_disp = np.hstack((x_disp[:,-ddx:],stripe))
                
            #plt.imshow(x_disp, cmap = cm.Greys_r)
            #plt.show()
            
            X_moved[idx] = x_disp
        except:
            aa = 1
        

    return X_moved

        
    