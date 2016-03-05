import matplotlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import log_loss
from nolearn.lasagne import NeuralNet
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.updates import adagrad,nesterov_momentum
from lasagne.nonlinearities import softmax

layers0 = [('input', InputLayer),
('dropoutf', DropoutLayer),
('dense0', DenseLayer),
('dropout', DropoutLayer),
('dense1', DenseLayer),
('dropout2', DropoutLayer),
('dense2', DenseLayer),
('output', DenseLayer)]

num_classes  = 3


def setTrainDataAndMakeModel(X_train,Y_train,X_test):
       
    num_features = X_train.shape[1]
    X_train_32 = X_train.astype(np.int32)
    Y_train_32 = Y_train.astype(np.int32)
    X_test_32 =  X_test.astype(np.int32)
    
    net0 = NeuralNet(layers=layers0,
    input_shape=(None, num_features),
    dropoutf_p=0.1,
    dense0_num_units=600,
    dropout_p=0.3,
    dense1_num_units=600,
    dropout2_p=0.1,
    dense2_num_units=600,
    output_num_units=num_classes,
    output_nonlinearity=softmax,
    update=nesterov_momentum,
    #update=adagrad,
    update_learning_rate=0.008,
    eval_size=0.2,
    #make 1 when testing
    verbose=0,
    max_epochs=40)

    net0.fit(X_train_32,Y_train_32)
    
    return net0.predict_proba(X_test_32)