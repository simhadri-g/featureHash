# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 11:05:06 2019

@author: Dell
"""
import numpy as np

from sklearn.metrics import mean_squared_error

epochs = 10000
input_size, hidden_size, output_size = 2, 2, 1
lr = .1 # learning rate
#w_hidden = np.random.uniform(size=(input_size, hidden_size))
#w_output = np.random.uniform(size=(hidden_size, output_size))
#act_hidden = (np.dot(X, w_hidden))
#print(X,w_hidden,"shata",act_hidden)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    #y=sigmoid(x)*(1-sigmoid(x))
    y = x * (1-x)
    return y
X = np.array([[0,0], [0,1], [1,0], [1,1]])
yp = np.array([ [0],   [1],   [1],   [0]])
print("f",np.exp(yp))

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
            w_virtual[i][j] = k
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
            w1_virtual[i][j] = k
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
            hashed_hlo[i][j] = k 
    return hashed_hlo


def has_for_hlo1(hidden_layer_output,X,comp_ratio): 
    hashed_hlo=np.zeros((len(X),comp_ratio))
    i=0      
    j = 0
    while (i < len(X)) and j < (comp_ratio):
        y = str(i)+str(j)
        y=bytes(y,'utf-8')
        x.update(y)
        k=x.intdigest()
        
        k=k%comp_ratio
        hash_value.append(k)
        if hidden_layer_output[i][k] not in hashed_hlo:
            hashed_hlo[i][j] = hidden_layer_output[i][k]
            i +=1
            j+=1
    return hashed_hlo



        #w1_virtual[i][j] = w1_real[k]

print("WWWW", w_virtual,"\n " , w_real)

hidden_bias =np.random.uniform(size=(1,comp_ratio))
output_bias = np.random.uniform(size=(1,output_size))

real = np.array([[i] for i in w1_real])

realx =  w_real
print("realx", realx, "\n real", real)
w_virtual_ind = w_(input_size, hidden_size, realx)

w1_virtual_ind = w1_(hidden_size, output_size, real )

hidden_layer_activation = np.dot(X,w_virtual)
#    print("x",X)
    
hidden_layer_output = sigmoid(hidden_layer_activation)

hlo = has_for_hlo(hidden_layer_output, X, comp_ratio)

print("indi",w_virtual_ind,"\nindei",w1_virtual_ind, "\ndamn hlo", hlo)
import math

for _ in range(epochs):
    #Forward Propagation
    p = 0 
    q = 0
    for i in range(len(w_virtual)):
        for j in w_virtual_ind[i]:
            w_virtual[p][q]= realx[int(i)][int(j)]
            q += 1
        p += 1
        q = 0
    p = 0
    q = 0
    for k in w1_virtual_ind:
        w1_virtual[p][q] = real[int(k)]
        p +=1
        
    print("man made", w_virtual,"woman made",w1_virtual)
    hidden_layer_activation = np.dot(X,w_virtual)
#    print("x",X)
    
    hidden_layer_output = sigmoid(hidden_layer_activation)
    ###############
    p = 0 
    q = 0
    for i in range(len(hlo)):
        for j in hlo[i]:
            hlo[p][q]= hidden_layer_output[int(i)][int(j)]
            q += 1
        p += 1
        q = 0
    print("hlo",hidden_layer_output,"\nf****",hlo)
    hlo += hidden_bias
#    print("hlo",hlo)
#    print("hidden",hidden_layer_output)
    
    
    
#    output_layer_activation = np.dot(hlo,real)
    ola = np.dot(hidden_layer_output,w1_virtual)
#    print("ola", ola)
#    print("ouptut",ola)
    ola += output_bias
    predicted_output = sigmoid(ola)
    print("predictions",predicted_output)
    
    #Backpropagation
    
    error = mean_squared_error(yp,predicted_output)
#    error = (yp -  predicted_output)
#    error = []
#    
#    
#    for i,j in zip(yp,predicted_output):
#        error.append(math.pow((i-j),2))
    print("error",error)
#    error = np.array(error)
#    error = error.reshape(predicted_output.shape)
    
    d_predicted_output = error * sigmoid_derivative(predicted_output)
#    print("dpred",d_predicted_output)
    
    
    error_hidden_layer = d_predicted_output.dot(np.array(real).T)
#    print("errorhl",error_hidden_layer)
    
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hlo)
#    print(d_hidden_layer)
    
#    print("damn real",real.shape)
    
    #Updating Weights and Biases
    real += hlo.T.dot(d_predicted_output) * lr
#    print("w1",real)
    #output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * lr
    
    realx += X.T.dot(d_hidden_layer) * lr
    
    output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * lr
    print( np.sum(d_hidden_layer,axis=0,keepdims=True),"c",hidden_bias)
    hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr
#    print("w",realx)
    #hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr #ion from inde of the element to the 
w1_virtual = w1_(hidden_size, output_size, real )
print(w1_virtual,"real",real)   
        
        
        
        
        
        
        
        
        
        
        
