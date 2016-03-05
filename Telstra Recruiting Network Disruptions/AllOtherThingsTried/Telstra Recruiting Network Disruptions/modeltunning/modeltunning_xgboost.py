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

print "reading csv's"
event_type = pd.read_csv('data/event_type.csv')
log_feature = pd.read_csv('data/log_feature.csv')
resource_type = pd.read_csv('data/resource_type.csv')
severity_type = pd.read_csv('data/severity_type.csv')

#===================== train and test related ===============================
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
train['location'] =  train['location'].map(lambda x: int(x.strip('location ')))
test['location'] =  test['location'].map(lambda x: int(x.strip('location ')))

#===================== severity related ===============================
#severity type only
severity_type['severity_type_only'] =  severity_type['severity_type'].map(lambda x: int(x.strip('severity_type ')))
#drop string severity column
severity_type.drop(["severity_type"],axis=1,inplace =True)

#inner join train,test and severity as 1 as 1 mapping.
dataset_train = pd.merge(train, severity_type, on =['id'])
dataset_test = pd.merge(test, severity_type, on =['id'])

#===================== event type related ===============================
#>>>>>>>>   derived
#>>>>>>>>   take total number of events
event_type['event_type_count'] = 1
total_no_of_events = event_type.groupby('id').event_type_count.sum()
total_no_of_events = total_no_of_events.reset_index()
#join on train
dataset_train = pd.merge(dataset_train,total_no_of_events,on =['id'])
dataset_test = pd.merge(dataset_test,total_no_of_events,on =['id'])

#take numbers only.
event_type['event_type_only'] = event_type['event_type'].map(lambda x: int(x.strip('event_type ')))
#sort event type by id
#event_type_sorted = event_type.sort(['id'])

#drop other frames
event_type.drop(['event_type','event_type_count'],axis=1,inplace=True)

# id and related list of events
#http://stackoverflow.com/questions/22219004/grouping-rows-in-list-in-pandas-groupby
#**** do not directlt assign **** bz id seq. not same.
event_id_and_list = event_type.groupby('id')['event_type_only'].apply(lambda x: x.tolist())
event_id_and_list = event_id_and_list.reset_index()

#for each column check if it is in list for that row if yes then 1 else 0
for i in range(1,55):
    event_id_and_list["e"+str(i)] = event_id_and_list["event_type_only"].apply(lambda x : 1 if i in x else 0) 

#drop event type only list (Uncomment below to GOOD VISUALISE IN CSV)
event_id_and_list.drop(['event_type_only'],axis=1,inplace=True)

dataset_train = pd.merge(dataset_train, event_id_and_list, on =['id'])
dataset_test = pd.merge(dataset_test, event_id_and_list, on =['id'])

#===================== log feature related ===============================
def assignVolumesColumns(x,log_feature):
    temp = 0
    for i in x :
        if i[0] == log_feature:
            temp =  i[1]
            break
    return temp

#working on log features
log_feature['log_feature_only'] = log_feature['log_feature'].map(lambda x: int(x.strip('feature ')))
log_feature['log_and_volume_all'] = zip(log_feature.log_feature_only, log_feature.volume)
log_feature_grouped = log_feature.groupby('id')['log_and_volume_all'].apply(lambda x: x.tolist())
log_feature_grouped = log_feature_grouped.reset_index()
for j in range(1,387):
    log_feature_grouped['f'+str(j)] = log_feature_grouped['log_and_volume_all'].apply(lambda k : assignVolumesColumns(k,j))

#drop 
log_feature_grouped_final = log_feature_grouped.drop(["log_and_volume_all"],axis=1)
#log_feature_grouped.to_csv("log_feature_grouped.csv")
dataset_train = pd.merge(dataset_train,log_feature_grouped_final,on=['id'])
dataset_test = pd.merge(dataset_test,log_feature_grouped_final,on=['id'])

#===================== resource related ===============================
#resource type 
resource_type['resource_type_only'] = resource_type['resource_type'].map(lambda x: int(x.strip('resource_type ')))
resource_type.drop(['resource_type'],axis=1,inplace=True)
resource_type = resource_type.groupby('id')['resource_type_only'].apply(lambda x: x.tolist())
resource_type = resource_type.reset_index()

resource_type['resource_count'] = resource_type['resource_type_only'].apply(lambda x:len(x))
for j in range(1,11):
    resource_type['r'+str(j)] = resource_type["resource_type_only"].apply(lambda x : 1 if j in x else 0) 

#resource_type.to_csv("resource_related_all.csv")
resource_type.drop(["resource_type_only"],axis=1,inplace=True)

dataset_train = pd.merge(dataset_train,resource_type,on=['id'])
dataset_test = pd.merge(dataset_test,resource_type,on=['id'])

#=========================================================================
#dataset_train.to_csv("dataset_train.csv")
#dataset_test.to_csv("dataset_test.csv")
#=========================================================================

#===================== features considered for train and test related ===============================
features_to_be_considered = []
'''
severity
'''
severity_f = ['severity_type_only'] 
'''
event type
'''
event_f = ['event_type_count'] 
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
resource_f = ['resource_count'] 
for r in range(1,11):
    resource_f.append('r'+str(r))

features_to_be_considered = severity_f + event_f + log_f + resource_f

dataset_train_sel_fea = dataset_train[features_to_be_considered]
dataset_train_index_val = dataset_train['id'].values
dataset_train_sel_fea_val = dataset_train_sel_fea.values
dataset_train_ans = dataset_train['fault_severity']
dataset_train_ans_val = dataset_train_ans.values
dataset_train_id_sevarity_Frame = dataset_train[['id','fault_severity']]

dataset_test_sel_fea_val = dataset_test[features_to_be_considered].values
dataset_test_index_val = dataset_test['id'].values

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

#=== stratified split
print "StratifiedKFold Started"
skf = StratifiedKFold(dataset_train_ans.values, n_folds=5) #chnage fold size to +/- train test.

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


xg_train = xgb.DMatrix( dataset_train_sel_fea_val, label=dataset_train_ans_val)

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
        for min_child in [.5,1,2,3,4,5]:
            print "min child "+str(min_child)                
            for gamma in [0,.5,1,1.5,2]:
                for eta in [0.1,0.15,.2,.25,.3,.35]:
                    print "running for eta "+ str(eta)
                    for max_depth in [20,25,30,40,50,60]:                   
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