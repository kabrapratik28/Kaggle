import pandas as pd
import numpy as np

import xgboost as xgb

import random
random.seed(21)
np.random.seed(21)

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

for i in range(1,6):
    severity_type["severity_"+str(i)] = severity_type["severity_type_only"].apply(lambda x : 1 if i == x else 0) 

#severity_greater_than_eq_3_has_only has fault severity 0_and_1
#plotData = sns.jointplot(x="severity_type_only", y="fault_severity", data=df)
#plotData = sns.lmplot("severity_type_only", "fault_severity", hue="fault_severity", data=df, fit_reg=False,x_jitter=.2,y_jitter=.2)
severity_type["severity_greater_than_eq_3_has_only_f0_and_1"] = severity_type["severity_type_only"].apply(lambda x : 5 if x>=3 else 0) 
#severity type only delete (bz it's categorical)    
#severity_type.drop(["severity_type_only"],axis=1,inplace =True)

#inner join train,test and severity as 1 as 1 mapping.
dataset_train = pd.merge(train, severity_type, on =['id'])
dataset_test = pd.merge(test, severity_type, on =['id'])

#===================== location related ===============================
train_id_location = train[['id','location']]
test_id_location = test[['id','location']]
train_test_id_location = pd.concat([train_id_location,test_id_location])

for i in range(1,1127):
    train_test_id_location["loc"+str(i)] = train_test_id_location["location"].apply(lambda x : 1 if i == x else 0) 

#by visualizing graph
#print location plot as joint plot stats up down
#plotData = sns.jointplot(x="fault_severity", y="location", data=df)
#plotData = sns.lmplot("fault_severity", "location", hue="fault_severity", data=df, fit_reg=False,x_jitter=.2)
train_test_id_location["is_loc_below_500_not_be_sev_2"] = train_test_id_location["location"].apply(lambda x : 2 if x<600 else 0) 


only_location_OHE_with_id =  train_test_id_location.drop(['location'],axis=1)
only_location_OHE = train_test_id_location.drop(['id','location'],axis=1)
 
#===try out ==> only_location_OHE_with_id, location_svd_df
dataset_train = pd.merge(dataset_train, only_location_OHE_with_id, on =['id'])
dataset_test = pd.merge(dataset_test, only_location_OHE_with_id, on =['id'])

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

listOfColumnNames  = []
for j in range(1,387):
    listOfColumnNames.append('f'+str(j))
    
#total number of log feature appeared (length of list)
log_feature_grouped['total_number_of_log_feature_appeared'] = log_feature_grouped['log_and_volume_all'].apply(lambda s: len(s))
#total volume of log feature appeared (before normalizing sum of all f1,..,f383 columns)
log_feature_grouped['total_volume_sum_of_log_feature'] = 0
for eachColumnName in listOfColumnNames:
    log_feature_grouped['total_volume_sum_of_log_feature'] = log_feature_grouped['total_volume_sum_of_log_feature']+  log_feature_grouped[eachColumnName]

def sum_of_prod_of_log_vol(listOfTp):
    ans =0 
    for k in listOfTp:
        ans = ans + k[0]*k[1]
    return ans
    
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
        
resource_type.drop(["resource_type_only"],axis=1,inplace=True)

dataset_train = pd.merge(dataset_train,resource_type,on=['id'])
dataset_test = pd.merge(dataset_test,resource_type,on=['id'])

#========================= mix featuers from above columns ===================
#avg log fea apperared
dataset_train['avg_log_feature'] = dataset_train['total_volume_sum_of_log_feature']/dataset_train['total_number_of_log_feature_appeared']
dataset_test['avg_log_feature'] = dataset_test['total_volume_sum_of_log_feature']/dataset_test['total_number_of_log_feature_appeared']

#total_no_of_log_res_eve
dataset_train['total_no_of_log_res_eve'] = dataset_train['total_number_of_log_feature_appeared']+dataset_train['event_type_count']+dataset_train['resource_count']
dataset_test['total_no_of_log_res_eve'] = dataset_test['total_number_of_log_feature_appeared']+dataset_test['event_type_count']+dataset_test['resource_count']

#no_of_events/no_of_resources
dataset_train['no_of_eve_per_res'] = dataset_train['event_type_count']/dataset_train['resource_count']
dataset_test['no_of_eve_per_res'] = dataset_test['event_type_count']/dataset_test['resource_count']

#mul_log_res_eve
dataset_train['mul_log_res_eve'] = dataset_train['total_number_of_log_feature_appeared']*dataset_train['event_type_count']*dataset_train['resource_count']
dataset_test['mul_log_res_eve'] = dataset_test['total_number_of_log_feature_appeared']*dataset_test['event_type_count']*dataset_test['resource_count']

#--------------------------Shraddha---------------------------------------
#severity_type vs resource count
res_severity_merged = pd.merge(resource_type, severity_type,on=['id'])
for j in range(1,11):
    res_severity_merged.drop(['r'+str(j)],axis=1,inplace=True)
for j in range(1,6):
    res_severity_merged.drop(['severity_'+str(j)],axis=1,inplace=True)
res_severity_merged.drop(['severity_greater_than_eq_3_has_only_f0_and_1'],axis=1,inplace=True)
def check_res_severity(x,y):
    if(x>2 and y>1):
        return 1
    else:
        return 0 
res_severity_merged["res_severity"] = res_severity_merged.apply(lambda x : 1 if x["resource_count"]>1 and x["severity_type_only"]>2 else 0, axis=1)
res_severity_merged.drop(["resource_count"],axis=1,inplace=True)
res_severity_merged.drop(["severity_type_only"],axis=1,inplace=True)

dataset_train = pd.merge(dataset_train,res_severity_merged,on=['id'])
dataset_test = pd.merge(dataset_test,res_severity_merged,on=['id'])
#--------------------------Shraddha Done---------------------------------------

#=========================================================================
#dataset_train.to_csv("dataset_train.csv")
#dataset_test.to_csv("dataset_test.csv")
#=========================================================================

#===================== features considered for train and test related ===============================
features_to_be_considered = []
'''
location
'''
location_f = ['is_loc_below_500_not_be_sev_2']
#OHE
for l in range(1,1127):
    location_f.append('loc'+str(l))

'''
severity
'''
severity_f = ['severity_greater_than_eq_3_has_only_f0_and_1']
for s in range(1,6):
    severity_f.append('severity_'+str(s))
'''
event type
'''
event_f = ['event_type_count',] 
for e in range(1,55):
    event_f.append('e'+str(e))
'''
log feature
'''
log_f = ['total_number_of_log_feature_appeared','total_volume_sum_of_log_feature',]
for l in range(1,387):
    log_f.append('f'+str(l))
    
'''
resource
'''
resource_f = ['resource_count'] 
for r in range(1,11):
    resource_f.append('r'+str(r))

features_to_be_considered = location_f + severity_f + event_f + log_f + resource_f + ['res_severity',] 


dataset_train_sel_fea = dataset_train[features_to_be_considered]
#log(1+x)
dataset_train_sel_fea = dataset_train_sel_fea.apply(lambda x: x+1).apply(np.log10)



dataset_test_sel_fea = dataset_test[features_to_be_considered]
#log(1+x)
dataset_test_sel_fea = dataset_test_sel_fea.apply(lambda x: x+1).apply(np.log10)


dataset_train_sel_fea_val = dataset_train_sel_fea.values
dataset_train_index_val = dataset_train['id'].values
dataset_train_ans = dataset_train['fault_severity']
dataset_train_ans_val = dataset_train_ans.values
dataset_train_id_sevarity_Frame = dataset_train[['id','fault_severity']]

dataset_test_sel_fea_val = dataset_test_sel_fea.values
dataset_test_index_val = dataset_test['id'].values
print "Done With making data"


from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier 
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

import pickle

model =  RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
            max_depth=27, max_features='log2', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=5,
            min_weight_fraction_leaf=0.0, n_estimators=25, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)       


from sklearn.feature_selection import RFECV
selector = RFECV(model, step=100,cv=5,scoring='log_loss')
selector = selector.fit(dataset_train_sel_fea,dataset_train_ans)

with open("selector_ranking_", 'wb') as f:
    pickle.dump(selector.ranking_, f)
    
with open("selector_ranking_", 'rb') as f:
    my_list = pickle.load(f)

