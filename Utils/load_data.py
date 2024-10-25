import numpy as np
import pandas as pd

features=["body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z"
]

def load_file(path):
    return pd.read_csv(path, delim_whitespace=True, header=None)

def load_X(type_of_data, datasetdir):
    data=[]
    for X in features:
        filename=f"{datasetdir}/{type_of_data}/Inertial Signals/{X}_{type_of_data}.txt"
        data.append(load_file(filename).to_numpy())
    
    return np.transpose(data, (1,2,0))

def load_y(datasetdir):
    y_train=load_file(f"{datasetdir}/train/y_train.txt")
    y_test=load_file(f"{datasetdir}/test/y_test.txt")
    activity_labels=load_file(f"{datasetdir}/activity_labels.txt")

    y_train[0]=y_train[0].map(dict(activity_labels.values))
    y_test[0]=y_test[0].map(dict(activity_labels.values))
    y_train.columns=['Activity']
    y_test.columns=['Activity']

    return y_train, y_test