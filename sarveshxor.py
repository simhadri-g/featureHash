# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 11:05:06 2019

@author: Dell
"""
import numpy as np
epochs = 100
input_size, hidden_size, output_size = 2, 6, 1
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
print("f",np.exp(yp))
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
w_real=np.random.uniform(size=(input_size,comp_ratio))
print("w_real", w_real)
import xxhash
x = xxhash.xxh64()
hash_value = []
def w_(input_size,hidden_size,w_real):
    for i in range(input_size):
        for j in range(hidden_size):
            y = str(i)+str(j)
            y=bytes(y,'utf-8')
            x.update(y)
            k=x.intdigest()
            
            k=k%comp_ratio
            hash_value.append(k)
            w_virtual[i][j] = w_real[i][k]
    return w_virtual
def w1_(hidden_size, output_size, w1_real):
    for i in range(hidden_size):
        for j in range(output_size):
            y = str(i)+str(j)
            y=bytes(y,'utf-8')
            x.update(y)
            k=x.intdigest()
            
            k=k%comp_ratio
            hash_value.append(k)
            w1_virtual[i][j] = w1_real[k]
    return w1_virtual
 

def has_for_hlo(hidden_layer_output,X,comp_ratio): 
    hashed_hlo=np.zeros((len(X),comp_ratio))      
    for i in range(len(X)):
        for j in range(comp_ratio):
            y = str(i)+str(j)
            y=bytes(y,'utf-8')
            x.update(y)
            k=x.intdigest()
            
            k=k%comp_ratio
            hash_value.append(k)
            hashed_hlo[i][j] = hidden_layer_output[i][k] 
    return hashed_hlo
        #w1_virtual[i][j] = w1_real[k]

print("WWWW", w_virtual,"\n " , w_real)

real = np.array([[i] for i in w1_real])

realx =  w_real
for _ in range(epochs):
    #Forward Propagation
    w_virtual = w_(input_size, hidden_size, realx)
    hidden_layer_activation = np.dot(X,w_virtual)
    print("x",X)
    #hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)
    ###############
    hlo = has_for_hlo(hidden_layer_output, X, comp_ratio)
    
    print("hlo",hlo)
    print("hidden",hidden_layer_output)
    
    w1_virtual = w1_(hidden_size, output_size, real )
#    output_layer_activation = np.dot(hlo,real)
    ola = np.dot(hidden_layer_output,w1_virtual)
#    print("ola", ola)
    print("ouptut",ola)
    #output_layer_activation += output_bias
    predicted_output = sigmoid(ola)
    #print(predicted_output)
    
    #Backpropagation
    error = yp - predicted_output
    print("error",error)
    
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    print("dpred",d_predicted_output)
    
    
    error_hidden_layer = d_predicted_output.dot(np.array(real).T)
    print("errorhl",error_hidden_layer)
    
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hlo)
    print(d_hidden_layer)
    
    print("damn real",real.shape)
    
    #Updating Weights and Biases
    real += hlo.T.dot(d_predicted_output) * lr
    print("w1",real)
    #output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * lr
    
    realx += X.T.dot(d_hidden_layer) * lr
    print("w",realx)
    #hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr #ion from inde of the element to the 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
