# Imports 

import pandas as pd
import math
import numpy as np
import random
import pickle
random.seed(1)

#Loading the data for training 

def load_data(data,labels):
  
    Input_data= pd.read_csv(data,header=None)
    Labels= pd.read_csv(labels,header=None)

    train_size=math.floor(0.8*(len(Input_data))) # defining size for splitting the data 
    
    x_train=Input_data.iloc[0:train_size,:] # 80% of train data 
    y_train=Labels.iloc[0:train_size,:]     # 80% of train labels 
    
    x_test=Input_data.iloc[train_size:,:]   #20% of train data--> val data  
    y_test=Labels.iloc[train_size:,:]       #20% of train data--> val data 

    # print(train_size)
    # print(x_train.head())
    # print(y_train.head())
    # print(x_test.head())
    # print(y_test.head())
    
    return x_train,y_train,x_test,y_test

# ''' preprocessing the data --> converting into numpy and transposing for passing into further functions 

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


#"""""""""""""""""""""""Activations

# " sigmoid: function used for hidden layer neurons. 
# The differentiable function predicts the probability as an output between 0 and 1. 

#softmax: functions used for output layer neurons to generate the probs. converts the logit scores into probabilities. 
# The softmax function is logistic activation function used for multiclass classification.

#""""""""""""""""""""""""

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps, axis=0)



# ''''''''''''''''''''''''''''''''''''''''''''''''''Initializing the network'''''''''''''''''''''''''''''''''''''''''
# Initializing the parameters: 

# 1. weight1, bias1 for layer 1 | weights1: 10*784 | bias1: 10*1  
# 2. weight2, bias2 for layer 2 | weights2: 4*10   | bias2: 4*1
# 3. These will be used for each data point/image in an epoch.  

# Returns: A dictionary with weights and biases as keys:value pairs 
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def create_network(layers_dims):

    n_input_neurons, n_hidden_neurons, n_output_neurons=layers_dims
    x=0.01
    random.seed(x*100)

    weight1 = np.random.randn(n_hidden_neurons, n_input_neurons)*x  # weights between input and hidden layer
    bias1 = np.zeros(shape=(n_hidden_neurons, 1)) # biases initialized to 0 for hidden layer
    weight2 = np.random.randn(n_output_neurons, n_hidden_neurons)*x #weights between hidden and output layer
    bias2 = np.zeros(shape=(n_output_neurons, 1)) # biases initialized to 0 for output layer 

    params = {"weight1": weight1,"bias1": bias1,"weight2": weight2,"bias2": bias2}
    return params

#"""""""""""""""""""""""""""""""""""""""""Forward base for each perceptron in each layer """""""""""""""""""""""""""""
# forward propagation functions

# 1. Performs the dot product of weights and inputs + bias for each neuron for both the layers. 
# 2. Example: 
                # (Weights: (10,784).A_n: (784,1900)) + bias: (10,1)==> output: (10,1900)  
                # 1900-> Images(let's say) 
                # output: 10 neurons in each row representing 10 neurons.
# 3. We will call this in "forward_propagate" for both the layers. 
# 4. Returns: output and previous values(Lets say old_set)=(A_n, Weights, bias)

# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def forward_base(A_n,Weights, bias):

    output = np.dot(Weights,A_n) + bias # returns the value of each neuron for all the images(lets say). shape-->(no_of_neurons,no_of_images)
    old_set = (A_n, Weights, bias) # inputs,prev weights, prev bias 

    return output,old_set

 
#"""""""""""""""""""""""""""""""""""""""""Forward propagate"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# forward propagation functions

# {For Hidden layer}

# 1. Accepts: X(784,19000), weight1(10,784), bias1(10,1),'sigmoid'
# 2. Returns: out1(10,19000), cache1
# 3. out1=sigmoid(out1)
# 4. cache1=(A_n, Weights, bias) + op before activation function

#{For output layer}

# 1. Accepts: X(784,19000), weight1(10,784), bias1(10,1),'softmax'
# 2. Returns: out2(10,19000), cache2
# 3. out2=softmax(out2)
# 4. cache2=(A_n, Weights, bias) + op before activation function

# We call this function in mlp_2L

# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def forward_propagate(last_A, weights, bias,activation):  

    if activation == "sigmoid":
        output, l_c = forward_base(last_A, weights, bias) #linear_cache_set consists of a tuple of (A,w,b)
        A_out = sigmoid(output)    

    elif activation == "softmax":
        output,l_c = forward_base(last_A, weights, bias)
        A_out = softmax(output)

    cache_set_activation = output
    cache = (l_c, cache_set_activation)

    return A_out, cache

# """""""""""""""""""""""""""""""""" cross entropy cost function"""""""""""""""""""""""""""""""""""""""""""""""""""""

#1.In order to keep a track on whether our model is learning during the fw and bw propagation, we use cross entropy cost function.
#2.We use squeeze to correct the shape Ex: [[5]] into 5.
#3. Returns the cost. 

# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def ce_cost(n_layer, y):

    size = y.shape[1] # 19000 samples (lets say)
    resultant_cost = (-1 / size) * np.sum(np.multiply(y, np.log(n_layer)) + np.multiply(1 - y, np.log(1 - n_layer)))
    resultant_cost = np.squeeze(resultant_cost)   

    # print("shape of cost :",resultant_cost.shape)
    return resultant_cost


#"""""""""""""""""""""""""""""""""""""""""""" Backward functions """"""""""""""""""""""""""""""""""""""""""""""""""""""""""" 

# To calculate the gradient of loss function given the parameters.

# 1. cache_set_linear: (A_n, Weights, bias) from the fw prop of the current layer. (1st layer or second layer)
# 2. derv_z: Gradient of cost 
# Returns: 
        # der_W: gradient w.r.t Weights 
        # der_b: gradient w.rt bias 
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def backward_sigmoid(der_A, c_s):

    x = c_s
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
#"""""""""""""""""""""""""""""""""""""""""""""""""""""""Backward_propagate"""""""""""""""""""""""""""""""""""""""""""

# 1. Accepts: 
        # der_A initially from mlp_2L = - (np.divide(Y, out2) - np.divide(1 - Y, 1 - out2))
        # cache: lienar and activation cache. 
        # Activation_function: softmax for op layer and sigmoid for hidden layer. 
# 2. Returns: 
        #der_A_prev: Gradient of cost wrt activation
        #der_W and der_b gradients. 
        # for both the layers.
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def backward_propagate(der_A, cache_set_2,activation_function):

    cache_set_linear, cache_set_activation = cache_set_2 #linear cache and activation cache 

    if activation_function == "sigmoid":
        der_Z = backward_sigmoid(der_A, cache_set_activation)

    elif activation_function == "softmax":
        der_Z=der_A
    der_A_prev, der_W, der_b = backward_base(der_Z, cache_set_linear,True)  

    return der_A_prev, der_W, der_b

#""""""""""""""""""""""""""""""""""""""parameter updates"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# 1. Accepts: 
            # params dictionary containing weights and biases for both the layers. 
            # A dictionary g with values of gradients obtaned form backward_propagate
# 2. Returns: Updated dictionary with parameters. 

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def param_change(params, grads, learning_rate):

    size = len(params) // 2 # number of layers in the neural network

    for i in range(size):
        params["weight" + str(i + 1)] = params["weight" + str(i + 1)] - learning_rate * grads["dW" + str(i + 1)]
        params["bias" + str(i + 1)] = params["bias" + str(i + 1)] - learning_rate * grads["db" + str(i + 1)]

    return params

#"""""""""""""""""""""""""""""""""""""""""""""""""""mlp_2L"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 1. Accepts:
        # 1. X: Input train data
        # 2. Y: Labels
        # 3. dims: (neuorns_input,neurons_hidden,neurons_output)
        # 4. epochs: 171 in our case after testing. 
        # 5. learning_rate: 0.1 

# 2. Calls: 
        # for each epoch: 
            # forward propagate for both the layers. 
            # cross entropy cost function.
            # backward propagate for both the layers. 
            # param_change to update the parameters. 

# 3. Returns: 
        #1. updated parameters dictionary.
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def gradient_update(dW1,db1,dW2,db2):
    g= {} 
    #gradient updation
    g['dW1'] = dW1
    g['db1'] = db1
    g['dW2'] = dW2
    g['db2'] = db2
    return g
 


def mlp_2L(X, Y, dims, epochs, learning_rate):

    np.random.seed(1)
    params = create_network(dims)

    weight1 = params["weight1"]
    bias1 = params["bias1"]
    weight2 = params["weight2"]
    bias2 = params["bias2"] 

    cost_list=list()   

    for i in range(epochs):  
        #fwd prop
        out1, cache1 = forward_propagate(X, weight1, bias1,'sigmoid')
        out2, cache2 = forward_propagate(out1, weight2, bias2,'softmax')
        #cross entropy
        cost = ce_cost(out2, Y)

        dA2= - (np.divide(Y, out2) - np.divide(1 - Y, 1 - out2))
        #bwd prop
        dA1, dW2, db2 = backward_propagate(dA2, cache2,'softmax')
        dA0, dW1, db1 = backward_propagate(dA1, cache1,'sigmoid')
        
        g=gradient_update(dW1,db1,dW2,db2)
        
        
        #parameter update
        params = param_change(params, g, learning_rate)

        weight1 = params["weight1"]
        bias1 = params["bias1"]
        weight2 = params["weight2"]
        bias2 = params["bias2"] 

        print("cost at",i,"is: ",cost)
        cost_list.append(cost)

    print("shape tes: ",params['bias1'].shape)
    
    return params

#""""""""""""""""""""""""""""""""""""""""""""""""classification"""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 1. Accepts: 
        # The parameters which were updated. 
        # The test set. 
# 2. Returns: 
        # Predicted labels 
        # and other parameters.
#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def classify(parameters_new,x_test):
    
    weight1 = parameters_new["weight1"]
    bias1 = parameters_new["bias1"]
    weight2 = parameters_new["weight2"]
    bias2 = parameters_new["bias2"]
    #fwd propagate
    out_z1, cache1 = forward_propagate(x_test, weight1, bias1,'sigmoid')
    out_z2, cache2 = forward_propagate(out_z1, weight2, bias2,'softmax')
    
    predicted_labels = np.argmax(out_z2.T,axis=1)
    real_labels = np.argmax(y_test.T,axis =1)
    
    return out_z1, cache1,out_z2,cache2,predicted_labels,real_labels

# Saves the model 

def pickle_dumps(parameters_new):
    
    pickle.dump(parameters_new,open("weights.pkl","wb"))

# Accuracy evaluation

def accuracy_metric(real_labels,predicted_labels):
    
    correct = 0
    for i in range(len(real_labels)):
        if real_labels[i] == predicted_labels[i]:
            correct += 1
    ret=correct/float(len(real_labels))*100.0
    return ret

if __name__ == '__main__':

    # defining constants
    n_input_neurons=784 # each image 
    n_hidden_neurons=10 # Taking the number of neurons in the middle layer=10 
    n_output_neurons=4  # output layer neurons 
    epochs=171  # Using multiple runs
    dims= (n_input_neurons, n_hidden_neurons, n_output_neurons)  # making a tuple to pass in further functions 
    learning_rate=0.1
    # print("No of epochs: ",epochs)
    # print("No of input neurons: ",n_input_neurons)
    # print("No of hidden neurons: ",n_hidden_neurons)
    # print("Learning rate: ",0.1)
    # print("--Loss--\n")
# function calls 

    x_train,y_train,x_test,y_test=load_data('train_data.csv','train_labels.csv')

    x_train,y_train,x_test,y_test=preprocess(x_train,y_train,x_test,y_test)

    parameters_new = mlp_2L(x_train,y_train,dims,epochs,learning_rate)
    final_parameters=parameters_new
    # print("typeparams: ",type(final_parameters))
    # print(final_parameters)
    out_z1, cache1,out_z2,cache2,predicted_labels,real_labels=classify(parameters_new,x_test)

    pickle_dumps(final_parameters)

    print(accuracy_metric(real_labels,predicted_labels))