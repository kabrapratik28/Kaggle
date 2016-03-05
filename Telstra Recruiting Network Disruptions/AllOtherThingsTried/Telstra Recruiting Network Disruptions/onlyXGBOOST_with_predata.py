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
from models import lasange
from models import MultinomialNB

#dimension reduction
from sklearn.decomposition import TruncatedSVD
#preprocessing
from sklearn import preprocessing

#===================== train and test related ===============================
train = pd.read_csv('dataset_train.csv')
test = pd.read_csv('dataset_test.csv')
print "Done with reading data !"
#===================== features considered for train and test related ===============================
features_to_be_considered = []
'''
location
'''
location_f = []
#OHE
for l in range(1,1127):
    location_f.append('loc'+str(l))
'''
severity
'''
severity_f = []
for s in range(1,6):
    severity_f.append('severity_'+str(s))
'''
event type
'''
event_f = [] 
for e in range(1,55):
    event_f.append('e'+str(e))
'''
log feature
'''
log_f = []
for l in range(1,387):
    log_f.append('f'+str(l))  
'''
resource
'''
resource_f = [] 
for r in range(1,11):
    resource_f.append('r'+str(r))
#===================== features reduction for train and test related ===============================
location_dimension_reduce_no = 8
severity_dimension_reduce_no = 2
event_dimension_reduce_no = 5
log_dimension_reduce_no = 7
resource_dimension_reduce_no = 3

'''
location
'''
location_f_svd = []
#OHE
for l in range(1,location_dimension_reduce_no+1):
    location_f_svd.append('loc_svd'+str(l))
'''
severity
'''
severity_f_svd = []
for s in range(1,severity_dimension_reduce_no+1):
    severity_f_svd.append('severity_svd'+str(s))
'''
event type
'''
event_f_svd = [] 
for e in range(1,event_dimension_reduce_no+1):
    event_f_svd.append('e_svd'+str(e))
'''
log feature
'''
log_f_svd = []
for l in range(1,log_dimension_reduce_no+1):
    log_f_svd.append('f_svd'+str(l))  
'''
resource
'''
resource_f_svd = [] 
for r in range(1,resource_dimension_reduce_no+1):
    resource_f_svd.append('r_svd'+str(r))



dataset_train_index_val = train['id'].values
dataset_test_index_val = test['id'].values

dataset_train_ans = train['fault_severity']
dataset_train_ans_val = dataset_train_ans.values



features_to_be_considered_log_1_x = []
features_to_be_considered_scale = []
additional_features_considered  = ['total_number_of_log_feature_appeared',
                                   'total_volume_sum_of_log_feature',
                                   'event_type_count','resource_count']

                                   
features_to_be_considered_normal = location_f + severity_f + event_f + log_f + resource_f 
features_to_be_considered_normal = features_to_be_considered_normal + additional_features_considered
features_to_be_considered_svd = location_f_svd + severity_f_svd + event_f_svd + log_f_svd + resource_f_svd
for i in features_to_be_considered_normal:
    features_to_be_considered_log_1_x.append(i+'_log_1_x')
    features_to_be_considered_scale.append(i+'_scale')

kmeansClus = []
for ww in range(0,7):     
    for i in range(0,3):
        kmeansClus.append('kc_'+str(ww)+"_"+str(i+1))
        

"""
 features_to_be_considered_normal, 
 features_to_be_considered_normal+features_to_be_considered_svd,
 features_to_be_considered_svd ,
 features_to_be_considered_log_1_x,
 features_to_be_considered_scale, 
 features_to_be_considered_log_1_x + features_to_be_considered_svd,
 features_to_be_considered_scale + features_to_be_considered_svd,
 
 #Clustering as input also 
 features_to_be_considered_normal+kmeansClus, 
 features_to_be_considered_normal+features_to_be_considered_svd+kmeansClus,
 features_to_be_considered_svd+kmeansClus ,
 features_to_be_considered_log_1_x+kmeansClus,
 features_to_be_considered_scale+kmeansClus, 
 features_to_be_considered_log_1_x + features_to_be_considered_svd+kmeansClus,
 features_to_be_considered_scale + features_to_be_considered_svd+kmeansClus

"""

features_to_be_considered = [
 features_to_be_considered_log_1_x ,
                             ] 


#====================== Make predication columns in pandas, return data frames =======================
def makeOutPutFrame(yprob,setOfIds, modelName):
    outputFinalFrame = pd.DataFrame({'id': setOfIds})
    for j in range(3):
        outputFinalFrame['predict_'+str(j)] = yprob[:,j]    
    #outputFinalFrame.set_index('id',inplace=True)
    #outputFinalFrame.to_csv("modelName"+".csv")
    return outputFinalFrame


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

# scale weight of positive examples
param['eta'] = .1
param['max_depth'] = 25
param['gamma'] = 1.5
param['min_child_weight']= 0.5
param['subsample']= 0.8
param['colsample_bytree']= 0.4
num_round = 135

for ww in range(0,len(features_to_be_considered)):
    dataset_train_sel_fea_val = ""
    dataset_train_ans_val = ""
    dataset_train_sel_fea_val = train[features_to_be_considered[ww]].values
    dataset_train_ans_val = train['fault_severity'].values
    xg_train = xgb.DMatrix( dataset_train_sel_fea_val, label=dataset_train_ans_val)
    
    #===================== split data for train and test related ===============================
    a = xgb.cv(param,xg_train,num_round,5)
    bestNoOfRounds = a['test-mlogloss-mean'].argmin()
    bestAnsForThisSetting =  a.ix[bestNoOfRounds]
    bestModelEval = bestAnsForThisSetting[0]
    with open("answer.txt", "a") as myfile:
        myfile.write(str(bestNoOfRounds))
        myfile.write(str(bestAnsForThisSetting))
        myfile.write(str(bestModelEval))
        myfile.write('===a=== ww ' + str(ww)+"\n")
    print bestAnsForThisSetting
    print "======"
   
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