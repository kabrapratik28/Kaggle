import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

event_type = pd.read_csv('event_type.csv')
log_feature = pd.read_csv('log_feature.csv')
resource_type = pd.read_csv('resource_type.csv')
severity_type = pd.read_csv('severity_type.csv')

#pd.unique(event_type['event_type'].values)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#inner join train and severity as 1 as 1 mapping.
dataset_train = pd.merge(train, severity_type, on =['id'])

#derieved
#take total number of events
event_type['event_type_count'] = 1
total_no_of_events = event_type.groupby('id').event_type_count.sum()
total_no_of_events = total_no_of_events.reset_index()
#join on train
dataset_train = pd.merge(dataset_train,total_no_of_events,on =['id'])

#take numbers only.
event_type['event_type_only'] = event_type['event_type'].map(lambda x: int(x.strip('event_type ')))
#sort event type by id
event_type_sorted = event_type.sort(['id'])

#drop other frames
event_type_sorted = event_type_sorted.drop(['event_type','event_type_count'],axis=1)

# id and related list of events
#http://stackoverflow.com/questions/22219004/grouping-rows-in-list-in-pandas-groupby
#**** do not directlt assign **** bz id seq. not same.
event_id_and_list = event_type_sorted.groupby('id')['event_type_only'].apply(lambda x: x.tolist())
event_id_and_list = event_id_and_list.reset_index()

dataset_train = pd.merge(dataset_train, event_id_and_list, on =['id'])

'''
def each_feature_maker(list_of_idex):
    new_empty_list = [0] * 54   # 54 is number of features
    for i in list_of_idex:
        new_empty_list[i-1] = 1 
    return new_empty_list
    
a = dataset_train["event_type_only"].apply(lambda x : each_feature_maker(x))
'''
#for each column check if it is in list for that row if yes then 1 else 0
for i in range(1,55):
    dataset_train["e"+str(i)] = dataset_train["event_type_only"].apply(lambda x : 1 if i in x else 0) 

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

'''    
#dataset_train.to_csv("testing.csv")
dataset_train = dataset_train.drop(['event_type_only','location','severity_type'],axis=1)

X_train, X_test = train_test_split(dataset_train, train_size=0.75)

X_train_output = X_train['fault_severity']
X_train_input = X_train.drop(['fault_severity'],axis=1)

X_test_output = X_test['fault_severity']
X_test_input = X_test.drop(['fault_severity'],axis=1)

rf = RandomForestClassifier(50)
rf.fit(X_train_input.values,X_train_output.values)

output = rf.predict(X_test_input.values)
print accuracy_score(X_test_output,output)


#done with ... severity_type, event type and log feature  
'''