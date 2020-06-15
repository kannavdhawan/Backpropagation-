from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy
import pandas as pd 
import sys
def main(path_file):
    # path_file='./sample.csv'
    y_pred = test_mlp(path_file)
    test_labels=pd.read_csv("labels.csv",header=None).to_numpy()
    test_accuracy= accuracy(test_labels, y_pred)*100
    print(test_accuracy)
if __name__=='__main__':
    main(sys.argv[1])
