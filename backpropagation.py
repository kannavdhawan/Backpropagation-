# Imports 
import pandas as pd
import math
import numpy as np
import random
import pickle
random.seed(1)
# defining constants
n_input_neurons=784
n_hidden_neurons=10
n_output_neurons=4
epochs=170
layers_dims= (n_input_neurons, n_hidden_neurons, n_output_neurons)
print("No of epochs: ",epochs)
print("No of input neurons: ",n_input_neurons)
print("No of hidden neurons: ",n_hidden_neurons)
print("Learning rate: ",0.1)
print("--Loss--\n")
#Load data
def load_data(data,labels):
    Input_data= pd.read_csv(data,header=None)
    Labels= pd.read_csv(labels,header=None)

    train_size=math.floor(0.8*(len(Input_data)))
    
    x_train=Input_data.iloc[0:train_size,:]
    y_train=Labels.iloc[0:train_size,:]
    
    x_test=Input_data.iloc[train_size:,:]
    y_test=Labels.iloc[train_size:,:]

    # print(train_size)
    # print(x_train.head())
    # print(y_train.head())
    # print(x_test.head())
    # print(y_test.head())
    return x_train,y_train,x_test,y_test

# preprocess
def preprocess(x_train,y_train,x_test,y_test):
    x_train=x_train.to_numpy().T
    y_train=y_train.to_numpy().T
    x_test=x_test.to_numpy().T
    y_test=y_test.to_numpy().T
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    return x_train,y_train,x_test,y_test
#Initializing the network
def create_network(n_input_neurons, n_hidden_neurons, n_output_neurons):
    random.seed(1)
    x=0.01
    weight1 = np.random.randn(n_hidden_neurons, n_input_neurons)*x  # weights between inp and hidden layer
    bias1 = np.zeros(shape=(n_hidden_neurons, 1)) # biases initialized to 0 for hidden layer 
    weight2 = np.random.randn(n_output_neurons, n_hidden_neurons)*x #weights between hidden and output layer
    bias2 = np.zeros(shape=(n_output_neurons, 1)) # biases initialized to 0 for output layer 
    params = {"weight1": weight1,"bias1": bias1,"weight2": weight2,"bias2": bias2}
    return params

# sigmoid
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

#softmax
def softmax(x):
    exps = np.exp(x)            #exps = np.exp(x)
    return exps / np.sum(exps, axis=0)

# forward propagation functions

def forward_base(A_n,Weights, bias):
    output = np.dot(Weights,A_n) + bias # returns the value of each neuron for all the images(lets say). shape-->(no_of_neurons,no_of_images)
    cache_set = (A_n, Weights, bias) # inputs,prev weights, prev bias 
    return output,cache_set

def forward_propagate(last_A, weights, bias,activation):  
    if activation == "sigmoid":
        output, cache_set = forward_base(last_A, weights, bias) #linear_cache_set consists of a tuple of (A,w,b)
        A_out = sigmoid(output)    
    elif activation == "softmax":
        output,cache_set = forward_base(last_A, weights, bias)
        A_out = softmax(output)
    cache_set_activation = output
    cache = (cache_set, cache_set_activation)
    return A_out, cache

# cost function
def ce_cost(A_Layer, Y_output):
    size = Y_output.shape[1] # 19000 samples (lets say)
    resultant_cost = (-1 / size) * np.sum(np.multiply(Y_output, np.log(A_Layer)) + np.multiply(1 - Y_output, np.log(1 - A_Layer)))
    resultant_cost = np.squeeze(resultant_cost)   
    # print("shape of cost :",resultant_cost.shape)
    return resultant_cost

# Backward propagate functions 

def backward_sigmoid(der_A, cache_set):
    x = cache_set
    y = 1/(1+np.exp(-x))
    der_Z = der_A * y * (1-y)    
    return der_Z

def backward_base(derv_z, cache_set_linear,x):
    A_n,Weights, bias= cache_set_linear 
    size = A_n.shape[1]
    der_W = np.dot(derv_z, A_n.T) / size
    der_b = np.sum(derv_z, axis=1, keepdims=x)/ size
    der_A_prev = np.dot(Weights.T, derv_z)
    return der_A_prev, der_W, der_b

def backward_propagate(der_A, cache_set_2,activation_function):
    cache_set_linear, cache_set_activation = cache_set_2 #linear cache and activation cache 
    if activation_function == "sigmoid":
        der_Z = backward_sigmoid(der_A, cache_set_activation)
    elif activation_function == "softmax":
        der_Z=der_A
    der_A_prev, der_W, der_b = backward_base(der_Z, cache_set_linear,True)  
    return der_A_prev, der_W, der_b

def param_change(params, grads, learning_rate):
    size = len(params) // 2 # number of layers in the neural network
    for i in range(size):
        params["weight" + str(i + 1)] = params["weight" + str(i + 1)] - learning_rate * grads["dW" + str(i + 1)]
        params["bias" + str(i + 1)] = params["bias" + str(i + 1)] - learning_rate * grads["db" + str(i + 1)]
    return params

def mlp_2L(X, Y, layers_dims, epochs, learning_rate=0.1):
    g= {}
    cost_list = []     
    np.random.seed(1)
    (n_input_neurons, n_hidden_neurons, n_output_neurons)=layers_dims
    params = create_network(n_input_neurons, n_hidden_neurons, n_output_neurons)
    weight1 = params["weight1"]
    bias1 = params["bias1"]
    weight2 = params["weight2"]
    bias2 = params["bias2"] 
    for i in range(epochs):  
        A1, cache1 = forward_propagate(X, weight1, bias1,'sigmoid')
        A2, cache2 = forward_propagate(A1, weight2, bias2,'softmax')
        cost = ce_cost(A2, Y)
        dA2= - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        dA1, dW2, db2 = backward_propagate(dA2, cache2,'softmax')
        dA0, dW1, db1 = backward_propagate(dA1, cache1,'sigmoid')
        g['dW1'] = dW1
        g['db1'] = db1
        g['dW2'] = dW2
        g['db2'] = db2
        params = param_change(params, g, learning_rate)
        weight1 = params["weight1"]
        bias1 = params["bias1"]
        weight2 = params["weight2"]
        bias2 = params["bias2"] 
        print("cost at",i,"is: ",cost)
        cost_list.append(cost)
    print("shape tes: ",params['bias1'].shape)
    return params

def classify(parameters_new,x_test):
    weight1 = parameters_new["weight1"]
    bias1 = parameters_new["bias1"]
    weight2 = parameters_new["weight2"]
    bias2 = parameters_new["bias2"]
    out_z1, cache1 = forward_propagate(x_test, weight1, bias1,'sigmoid')
    out_z2, cache2 = forward_propagate(out_z1, weight2, bias2,'softmax')
    predicted_labels = np.argmax(out_z2.T,axis=1)
    real_labels = np.argmax(y_test.T,axis =1)
    return out_z1, cache1,out_z2,cache2,predicted_labels,real_labels


def pickle_dumps(parameters_new):
    weights=pickle.dump(parameters_new,open("weights.pkl","wb"))

def accuracy_metric(real_labels,predicted_labels):
    correct = 0
    for i in range(len(real_labels)):
        if real_labels[i] == predicted_labels[i]:
            correct += 1
    ret=correct/float(len(real_labels))*100.0
    return ret

# function calls 

x_train,y_train,x_test,y_test=load_data('train_data.csv','train_labels.csv')
x_train,y_train,x_test,y_test=preprocess(x_train,y_train,x_test,y_test)
parameters_new = mlp_2L(x_train, y_train, epochs=171,layers_dims = (n_input_neurons, n_hidden_neurons, n_output_neurons))
out_z1, cache1,out_z2,cache2,predicted_labels,real_labels=classify(parameters_new,x_test)
pickle_dumps(parameters_new)
print(accuracy_metric(real_labels,predicted_labels))