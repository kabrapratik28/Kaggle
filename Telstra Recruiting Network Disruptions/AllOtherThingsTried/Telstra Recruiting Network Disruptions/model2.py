import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV

import xgboost as xgb



#======================DONT COPY THIS FUNCTION ========================
def makeOutPutFrame(yprob,setOfIds, modelName):
    outputFinalFrame = pd.DataFrame({'id': setOfIds})
    for j in range(3):
        outputFinalFrame['predict_'+str(j)] = yprob[:,j]    
    #outputFinalFrame.set_index('id',inplace=True)
    #outputFinalFrame.to_csv("modelName"+".csv")
    return outputFinalFrame


x_data = pd.read_csv('X.csv')
y_data = pd.read_csv('Y.csv')

x_data_id = x_data['id'].values
x_answer = x_data['fault_severity'].values
x_only_data = x_data.drop(['id','fault_severity'],axis=1).values

y_data_id = y_data['id'].values
y_only_data = y_data.drop(['id'],axis=1).values

#=======final model =============

#===================== XGBOOST related settings ===============================
# setup parameters for xgboost
param = {}
param['num_class'] = 3  # number of classes
# use softmax multi-class classification
param['objective'] = 'multi:softprob'
param['eval_metric'] = 'mlogloss'  ## change to mlogerror
param['nthread'] = 4

# scale weight of positive examples
param['eta'] = .1
param['max_depth'] = 25
param['gamma'] = 1.5
param['min_child_weight']= 0.5
param['subsample']= 0.8
param['colsample_bytree']= 0.4
num_round = 135


xg_train = xgb.DMatrix(x_only_data , label=x_answer)

#===================== split data for train and test related ===============================
a = xgb.cv(param,xg_train,num_round,5)
bestNoOfRounds = a['test-mlogloss-mean'].argmin()
bestAnsForThisSetting =  a.ix[bestNoOfRounds]
bestModelEval = bestAnsForThisSetting[0]
print bestNoOfRounds
print bestAnsForThisSetting
print bestModelEval
#print '===a==='
'''

def makeOutPutFrame(yprob,setOfIds, modelName):
    outputFinalFrame = pd.DataFrame({'id': setOfIds})
    for j in range(3):
        outputFinalFrame['predict_'+str(j)] = yprob[:,j]    
    outputFinalFrame.set_index('id',inplace=True)
    outputFinalFrame.to_csv(modelName+".csv")


xg_train = xgb.DMatrix( x_only_data, label=x_answer)

xg_test = xgb.DMatrix(y_only_data)
bst = xgb.train(param, xg_train, num_round )
yprob =  bst.predict( xg_test )
makeOutPutFrame(yprob,dataset_test_index_val,"xgboostanswers")
'''