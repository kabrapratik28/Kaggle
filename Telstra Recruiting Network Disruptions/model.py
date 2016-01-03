import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

event_type = pd.read_csv('event_type.csv')
log_feature = pd.read_csv('log_feature.csv')
resource_type = pd.read_csv('resource_type.csv')
severity_type = pd.read_csv('severity_type.csv')

#===================== train and test related ===============================
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
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
#>>>>>>>>   derieved
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

resource_type.to_csv("resource_related_all.csv")
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
dataset_train_sel_fea_val = dataset_train_sel_fea.values
dataset_train_ans = dataset_train['fault_severity']
dataset_train_ans_val = dataset_train_ans.values

#===================== XGBOOST related settings ===============================
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softprob'
# scale weight of positive examples
#param['eta'] = 0.1
param['max_depth'] = 500
#param['silent'] = 1   ##setiing this will not print msgs.
param['nthread'] = 4
param['num_class'] = 3  # number of classes
#param['eval_metric'] = 'mlogloss'
num_round = 50
#===================== split data for train and test related ===============================

#train test split
#X_train, X_test, Y_train,Y_test = train_test_split(dataset_train_sel_fea,dataset_train_ans, train_size=0.75)

#=== stratified split
skf = StratifiedKFold(dataset_train_ans.values, n_folds=5) #chnage fold size to +/- train test.
for train_index, test_index in skf:
    X_train, X_test = dataset_train_sel_fea_val[train_index], dataset_train_sel_fea_val[test_index]
    Y_train, Y_test = dataset_train_ans_val[train_index], dataset_train_ans_val[test_index]

    #===================== Algorithm to apply related ===============================
    #Random forest classifier
    
    model = RandomForestClassifier(100)
    model.fit(X_train,Y_train)

    output = model.predict(X_test)
    print "RFC"
    print accuracy_score(Y_test,output)
    
    #xgb 
    xg_train = xgb.DMatrix( X_train, label=Y_train)
    xg_test = xgb.DMatrix(X_test, label=Y_test)
    watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
    
    bst = xgb.train(param, xg_train, num_round, watchlist )    
    

#====================== model to apply on actual test data =======================
'''
model.fit(dataset_train[features_to_be_considered].values,dataset_train['fault_severity'].values)
outputFinal = model.predict_proba(dataset_test[features_to_be_considered].values) 
outputFinalFrame = pd.DataFrame({'id': dataset_test['id']})
for j in range(3):
    outputFinalFrame['predict_'+str(j)] = outputFinal[:,j]

outputFinalFrame.set_index('id',inplace=True)
outputFinalFrame.to_csv("output.csv")


xg_train = xgb.DMatrix( dataset_train_sel_fea_val, label=dataset_train_ans_val)
xg_test = xgb.DMatrix(dataset_test[features_to_be_considered].values)
bst = xgb.train(param, xg_train, num_round )    
yprob = bst.predict( xg_test )
'''
##things to do
'''
0.Make as much feature as possible 

use KFold
https://github.com/dmlc/xgboost/blob/master/demo/guide-python/sklearn_examples.py#L22

***PARAMETER TUNNING
https://github.com/dmlc/xgboost/blob/master/demo/guide-python/sklearn_examples.py#L51

Use deep learning using  
Lasagne and Theano
(Lasagne is a lightweight library to build and train neural networks in Theano)
http://blog.kaggle.com/2015/06/09/otto-product-classification-winners-interview-2nd-place-alexander-guschin/
https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/14335/1st-place-winner-solution-gilberto-titericz-stanislav-semenov
    |
    |_> https://kaggle2.blob.core.windows.net/forum-message-attachments/79598/2514/FINAL_ARCHITECTURE.png?sv=2012-02-12&se=2016-01-05T09%3A20%3A50Z&sr=b&sp=r&sig=837azHavE9PLc9h0hrTKtvZ3cdB1AQ4yCElKuAV0MTc%3D

Study on this :
https://www.kaggle.com/tqchen/otto-group-product-classification-challenge/understanding-xgboost-model-on-otto-data/notebook

'''