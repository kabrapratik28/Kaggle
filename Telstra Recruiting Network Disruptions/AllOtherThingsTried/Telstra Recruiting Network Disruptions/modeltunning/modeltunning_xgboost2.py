import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV

import xgboost as xgb

#our model and imports
from models import xgboostmodel
from models import randomforestclassifier
from models import KNN

import csv

x_data = pd.read_csv('X.csv')
y_data = pd.read_csv('Y.csv')

x_data_id = x_data['id'].values
x_answer = x_data['fault_severity'].values
x_only_data = x_data.drop(['id','fault_severity'],axis=1).values

y_data_id = y_data['id'].values
y_only_data = y_data.drop(['id'],axis=1).values

#====================== Make predication columns in pandas, return data frames =======================
def makeOutPutFrame(yprob,setOfIds, modelName):
    outputFinalFrame = pd.DataFrame({'id': setOfIds})
    for j in range(3):
        outputFinalFrame['predict_'+str(j)] = yprob[:,j]    
    #outputFinalFrame.set_index('id',inplace=True)
    #outputFinalFrame.to_csv("modelName"+".csv")
    return outputFinalFrame

#train test split
#X_train, X_test, Y_train,Y_test = train_test_split(dataset_train_sel_fea,dataset_train_ans, train_size=0.75)

#make list for every model
xgboostStorer = []
knn8Storer = []
knn16Storer = []
knn32Storer = []
#loop count
loopcount = 0 


#===================== XGBOOST related settings ===============================
# setup parameters for xgboost
param = {}
param['num_class'] = 3  # number of classes
# use softmax multi-class classification
param['objective'] = 'multi:softprob'
param['eval_metric'] = 'mlogloss'  ## change to mlogerror
param['nthread'] = 4
param['silent'] = 1

# scale weight of positive examples
param['eta'] = .1
param['max_depth'] = 20
param['gamma'] = 2
param['min_child_weight']=2
param['subsample']=.5
param['early_stopping_rounds']=25
num_round = 250

xg_train = xgb.DMatrix(x_only_data , label=x_answer)

#===================== split data for train and test related ===============================
'''
a = xgb.cv(param,xg_train,num_round,5)
print '===a==='
a.to_csv("see.csv")
'''

minval = 1000
counter = 0
 
for coalsample_by_tree in [.2,.4,.6,.8]:
    for sub_sample in [.2,.4,.6,.8]:
        for min_child in [.2,.5,.7,1]:
            print "min child "+str(min_child)                
            for gamma in [0,.5,1,1.5,2]:
                for eta in [0.05,0.1,0.15,.2]:
                    print "running for eta "+ str(eta)
                    for max_depth in [5,10,20,25,30]:                   
                        param['eta'] = eta
                        param['max_depth'] = max_depth
                        param['gamma'] = gamma
                        param['min_child_weight']= min_child
                        param['subsample']= sub_sample
                        param['colsample_bytree'] = coalsample_by_tree
                        a = xgb.cv(param,xg_train,num_round,5)
                        bestNoOfRounds = a['test-mlogloss-mean'].argmin()
                        bestAnsForThisSetting =  a.ix[bestNoOfRounds]
                        bestModelEval = bestAnsForThisSetting[0]
                        if bestModelEval < minval :
                            print "============================="
                            print 'bestNoOfRounds  '+ str(bestNoOfRounds) 
                            print 'bestModelEval  '+str(bestModelEval)
                            minval =  bestModelEval
                            bestAnsForThisSetting = bestAnsForThisSetting.reset_index()
                            bestAnsForThisSetting.to_csv("bestmodeltunning.csv")
                            f = open("goldenanswerByModelTunning.csv", "w")
                            w = csv.writer(f)
                            for ekey, eval in param.iteritems():
                                print ekey, eval
                                w.writerow([ekey, eval])
                            w.writerow(['no of rounds', str(bestNoOfRounds)])
                            f.close()
                            bestparam = param
                            greatNoOfRounds =  bestNoOfRounds                           
                            print "============================="
                        else:
                            print "counter "+str(counter) +" ans " +str(bestModelEval)
                        counter = counter + 1
                        
print "================bestparam================="
print bestparam
print greatNoOfRounds
'''
def makeOutPutFrame(yprob,setOfIds, modelName):
    outputFinalFrame = pd.DataFrame({'id': setOfIds})
    for j in range(3):
        outputFinalFrame['predict_'+str(j)] = yprob[:,j]    
    outputFinalFrame.set_index('id',inplace=True)
    outputFinalFrame.to_csv(modelName+".csv")


xg_train = xgb.DMatrix( dataset_train_sel_fea_val, label=dataset_train_ans_val)

xg_test = xgb.DMatrix(dataset_test_sel_fea_val)
bst = xgb.train(param, xg_train, num_round )
yprob =  bst.predict( xg_test )
makeOutPutFrame(yprob,dataset_test_index_val,"xgboostanswers")
'''