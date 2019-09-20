# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 11:05:06 2019

@author: Dell
"""
import numpy as np
epochs = 50000
input_size, hidden_size, output_size = 2, 3, 1
lr = .1 # learning rate
#w_hidden = np.random.uniform(size=(input_size, hidden_size))
#w_output = np.random.uniform(size=(hidden_size, output_size))
#act_hidden = (np.dot(X, w_hidden))
#print(X,w_hidden,"shata",act_hidden)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    y=sigmoid(x)*(1-sigmoid(x))
    return y
X = np.array([[0,0], [0,1], [1,0], [1,1]])
yp = np.array([ [0],   [1],   [1],   [0]])
#for epoch in range(epochs):
# 
#    # Forward
#    act_hidden = sigmoid(np.dot(X, w_hidden))
#    output = np.dot(act_hidden, w_output)
#    
#    # Calculate error
#    error = y - output
#    
#    if epoch % 5000 == 0:
#        print(f'error sum {sum(error)}')
#
#    # Backward
#    dZ = error * LR
#    w_output += act_hidden.T.dot(dZ)
#    dH = dZ.dot(w_output.T) * sigmoid_prime(act_hidden)
#    w_hidden += X.T.dot(dH)
#    
#    X_test = X[1] # [0, 1]
#
#
#np.dot(act_hidden, w_output)

comp_ratio=int(input())
w_virtual=np.zeros((input_size,hidden_size))
w1_virtual=np.zeros((hidden_size,output_size))
w1_real=np.random.uniform(size=(comp_ratio))
w_real=np.random.uniform(size=(comp_ratio))
import xxhash
x = xxhash.xxh64()
hash_value = []
for i in range(input_size):
    for j in range(hidden_size):
        y = str(i)+str(j)
        y=bytes(y,'utf-8')
        x.update(y)
        k=x.intdigest()
        
        k=k%comp_ratio
        hash_value.append(k)
        w_virtual[i][j] = w_real[k]
        
for i in range(hidden_size):
    for j in range(output_size):
        y = str(i)+str(j)
        y=bytes(y,'utf-8')
        x.update(y)
        k=x.intdigest()
        
        k=k%comp_ratio
        hash_value.append(k)
        w1_virtual[i][j] = w1_real[k]


        
print(y)
for _ in range(epochs):
    #Forward Propagation
    hidden_layer_activation = np.dot(X,w_virtual)
    #hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output,w1_virtual)
    #output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)
    #print(predicted_output)
    
    #Backpropagation
    error = yp - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
#    print(d_predicted_output,"D",w1_virtual,"S",w1_real)
    error_hidden_layer = d_predicted_output.dot(np.array([w1_real]))
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    #Updating Weights and Biases
    w1_real += hidden_layer_output.T.dot(d_predicted_output) * lr
    #outpu3t_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * lr
    w_real += X.T.dot(d_hidden_layer) * lr
    #hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lrion from inde of the element to the 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        