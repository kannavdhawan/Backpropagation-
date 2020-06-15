from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy
import pandas as pd 
y_pred = test_mlp('./sample.csv')
test_labels=pd.read_csv("labels.csv",header=None).to_numpy()
test_accuracy= accuracy(test_labels, y_pred)*100
print(test_accuracy)