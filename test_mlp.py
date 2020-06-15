import numpy as np
import pandas as pd 
import pickle 
import sys
from backpropagation import forward_propagate
from acc_calc import accuracy

STUDENT_NAME = 'KANNAV DHAWAN'
STUDENT_ID = '20831774'

def test_mlp(data_file):

    #loading the pickle 
    
    infile= open("weights.pkl",'rb')
    parameters_final=pickle.load(infile)
    print(parameters_final)
    # parameters_new = pickle.load(open(weights, 'rb'))
    test_data=pd.read_csv(data_file,header=None).to_numpy().T
    # print("test data shape:",test_data)


    # fetching the saved parameters in pickle 

    weight1 = parameters_final["weight1"]
    bias1 = parameters_final["bias1"]
    weight2 = parameters_final["weight2"]
    bias2 = parameters_final["bias2"]

    #passing the parameters to forward propagate for test_data

    out1, prev1 = forward_propagate(test_data, weight1, bias1,'sigmoid')
    out2, prev2 = forward_propagate(out1, weight2, bias2,'softmax')
    predicted_labels=out2
    # print("Predicted labels shape: ",predicted_labels.shape)
    # print("Dataframe explicit testing: ",pd.DataFrame(predicted_labels).head())
    predicted_labels = np.argmax(predicted_labels.T,axis=1)
    print("predicted_labels: ",predicted_labels)
    y_pred=[]
    for i in predicted_labels:
	    y=np.zeros(4)
	    y[i]=1
	    y_pred.append(y)
    y_pred=np.asarray(y_pred)
    print("one hot encoded pred: ",y_pred)
    # print("y_pred shape",y_pred.shape)
    return y_pred