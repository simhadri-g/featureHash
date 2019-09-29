# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 22:41:04 2019

@author: Windows 8.1
"""

import numpy as np
import sklearn.metrics as sk
import matplotlib.pyplot as plt
import xxhash
x = xxhash.xxh64()

epochs = 10000
input_size, hidden_size, output_size = 3, 7, 1
lr = .1 # learning rate

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    y=sigmoid(x)*(1-sigmoid(x))
    return y

def hasingFunction(row,column):
    hashing = np.zeros((row,column))
    print(row,column)
    # Always returns same hash value. Awesome!
    for i in range(row):
        for j in range(column):
            y = str(i)+str(j)
            y=bytes(y,'utf-8')
            #x.update(y)
            # print("y",y)
            #k=x.intdigest()
            # print(k)
            #k=k%comp_ratio
            k = xxhash.xxh64(y,10).intdigest()%comp_ratio
            # print(k,type(k))
            hashing[i][j] = k
    return hashing

def shareWeigths(virtualMat,  realMat):
    r,c = virtualMat.shape
    #print('--------------')
    virtualMat = np.zeros((r,c))
    for i in range(r):
        for j in range(c):
            y = str(i)+str(j)
            y=bytes(y,'utf-8')
            x.update(y)
            #print("y",y)
            k=x.intdigest()
            # print(k)
            k=k%comp_ratio
            # print(k,type(k))
            
            #print(hashMat[i][j],'==',k,'==',xxhash.xxh64(y,10).intdigest()%comp_ratio)
            
            virtualMat[i][j] += realMat[0][xxhash.xxh64(y,10).intdigest()%comp_ratio]
    return virtualMat

def reallyWeridIdea(virtualMat,realMat):
    r,c = virtualMat.shape
    #errMat = np.zeros((r,c))
    for i in range(r):
        for j in range(c):
            y = str(i)+str(j)
            y=bytes(y,'utf-8')
            realMat[0][xxhash.xxh64(y,10).intdigest()%comp_ratio] = virtualMat[i][j]
    #print('real',realMat)
    return realMat
    
#    error = yp - output_layer
#    if(epoch%100==0):
#        print("================ epoch = ",epoch,"=================")
#        print("\nerror",error)
#    merror.append(np.average(np.abs(error)))
#    #print("\nmerror = ",merror)
#    
#    # backpropogation
#    dE = error * lr
#    w1_virtual += np.dot(dE.T, hidden_layer)
#    w1_real = [int(w1_hash[i][j])]
#    dH = dE.dot(w1_virtual) * sigmoid_derivative(hidden_layer)
#    w_virtual += np.dot(X.T, dH )
        

X = np.array([[0,0,0], [1,0,1], [0,1,0], [0,0,1],[0,0,0], [1,0,1], [0,1,0], [1,1,1]])
yp = np.array([ [0],   [0],   [1],   [1],[0],   [0],   [1],   [1]])

#hidden_layer = np.zeros((1,hidden_size))
# compression ratio cannot be more than number of hidden nodes or hidden layer size
comp_ratio = int(hidden_size-int(input("compression ratio = ")))
print("actual comp_ratio",comp_ratio)

merror = []

w_hash = hasingFunction(input_size,hidden_size)
w1_hash =  hasingFunction(output_size, hidden_size)

w_virtual = np.random.uniform( size=(input_size,hidden_size))
w_real = np.random.uniform(size=(1,comp_ratio))
w_virtual = shareWeigths(w_virtual,w_real)
w1_virtual = np.random.uniform( size=(output_size, hidden_size))
w1_real = np.random.uniform(size=(1,comp_ratio))
w1_virtual = shareWeigths(w1_virtual,w1_real)
#w_hash = np.zeros((input_size,hidden_size))
#w1_hash = np.zeros((output_size, hidden_size))




#print("hidden_layer shape =",hidden_layer.shape)
print("w_virtual shape =",w_virtual.shape)
print("w_real shape =",w_real.shape)
print("w1_virtual shape =",w1_virtual.shape)
print("w1_real shape =",w1_real.shape)




for epoch in range(epochs):
    
    # feed forward
    hidden_layer = sigmoid(np.dot(X,w_virtual))
    #print("hidden = ",hidden_layer)
    output_layer = sigmoid(np.dot(hidden_layer, w1_virtual.T))
    #print("\n output =",output_layer)
    
    
    error = yp - output_layer
    if(epoch%100==0):
        print("================ epoch = ",epoch,"=================")
        print("\nerror",error)
    merror.append(np.average(np.abs(error)))
    #print("\nmerror = ",merror)
    
    # backpropogation
    dE = error * lr
    w1_virtual += np.dot(dE.T, hidden_layer)
    w1_real = reallyWeridIdea(w1_virtual,w1_real)
    #print("================ here w1V==========")
        
    w1_virtual = shareWeigths(w1_virtual,w1_real)
    
    dH = dE.dot(w1_virtual) * sigmoid_derivative(hidden_layer)
    w_virtual += np.dot(X.T, dH )
    w_real = reallyWeridIdea(w_virtual,w_real)
    #print("================ here w2V==========")
    
    w_virtual = shareWeigths(w_virtual,w_real)
    
plt.title("mean error")
plt.ylabel("error")
plt.xlabel("epochs")
plt.plot(merror)
plt.show()

    
    
    


