import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score,log_loss
from sklearn.grid_search import GridSearchCV
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

#severity type only delete (bz it's categorical)    
severity_type.drop(["severity_type_only"],axis=1,inplace =True)

#inner join train,test and severity as 1 as 1 mapping.
dataset_train = pd.merge(train, severity_type, on =['id'])
dataset_test = pd.merge(test, severity_type, on =['id'])

#===================== location related ===============================
train_id_location = train[['id','location']]
test_id_location = test[['id','location']]
train_test_id_location = pd.concat([train_id_location,test_id_location])

for i in range(1,1127):
    train_test_id_location["loc"+str(i)] = train_test_id_location["location"].apply(lambda x : 1 if i == x else 0) 

only_location_OHE_with_id =  train_test_id_location.drop(['location'],axis=1)
only_location_OHE = train_test_id_location.drop(['id','location'],axis=1)
 
#===try out ==> only_location_OHE_with_id
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

listOfColumnNames  = []
for j in range(1,387):
    listOfColumnNames.append('f'+str(j))
    
#total number of log feature appeared (length of list)
log_feature_grouped['total_number_of_log_feature_appeared'] = log_feature_grouped['log_and_volume_all'].apply(lambda s: len(s))
#total volume of log feature appeared (before normalizing sum of all f1,..,f383 columns)
log_feature_grouped['total_volume_sum_of_log_feature'] = 0
for eachColumnName in listOfColumnNames:
    log_feature_grouped['total_volume_sum_of_log_feature'] = log_feature_grouped['total_volume_sum_of_log_feature']+  log_feature_grouped[eachColumnName]

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

svd = TruncatedSVD(n_components=location_dimension_reduce_no)
svd_train = pd.DataFrame(svd.fit_transform(dataset_train[location_f]),columns= location_f_svd)
svd_train['id'] = dataset_train['id'].values
svd_test = pd.DataFrame(svd.transform(dataset_test[location_f]),columns= location_f_svd)
svd_test['id'] = dataset_test['id'].values
dataset_train = pd.merge(dataset_train,svd_train,on=['id'])
dataset_test = pd.merge(dataset_test,svd_test,on=['id'])


svd = TruncatedSVD(n_components=severity_dimension_reduce_no)
svd_train = pd.DataFrame(svd.fit_transform(dataset_train[severity_f]),columns= severity_f_svd)
svd_train['id'] = dataset_train['id'].values
svd_test = pd.DataFrame(svd.transform(dataset_test[severity_f]),columns= severity_f_svd)
svd_test['id'] = dataset_test['id'].values
dataset_train = pd.merge(dataset_train,svd_train,on=['id'])
dataset_test = pd.merge(dataset_test,svd_test,on=['id'])


svd = TruncatedSVD(n_components=event_dimension_reduce_no)
svd_train = pd.DataFrame(svd.fit_transform(dataset_train[event_f]),columns= event_f_svd)
svd_train['id'] = dataset_train['id'].values
svd_test = pd.DataFrame(svd.transform(dataset_test[event_f]),columns= event_f_svd)
svd_test['id'] = dataset_test['id'].values
dataset_train = pd.merge(dataset_train,svd_train,on=['id'])
dataset_test = pd.merge(dataset_test,svd_test,on=['id'])

svd = TruncatedSVD(n_components=log_dimension_reduce_no)
svd_train = pd.DataFrame(svd.fit_transform(dataset_train[log_f]),columns= log_f_svd)
svd_train['id'] = dataset_train['id'].values
svd_test = pd.DataFrame(svd.transform(dataset_test[log_f]),columns= log_f_svd)
svd_test['id'] = dataset_test['id'].values
dataset_train = pd.merge(dataset_train,svd_train,on=['id'])
dataset_test = pd.merge(dataset_test,svd_test,on=['id'])


svd = TruncatedSVD(n_components=resource_dimension_reduce_no)
svd_train = pd.DataFrame(svd.fit_transform(dataset_train[resource_f]),columns= resource_f_svd)
svd_train['id'] = dataset_train['id'].values
svd_test = pd.DataFrame(svd.transform(dataset_test[resource_f]),columns= resource_f_svd)
svd_test['id'] = dataset_test['id'].values
dataset_train = pd.merge(dataset_train,svd_train,on=['id'])
dataset_test = pd.merge(dataset_test,svd_test,on=['id'])

log1x = True
scaleData = True
isKmeans  = True

features_to_be_considered = location_f + severity_f + event_f + log_f + resource_f
additional_features_considered  = ['total_number_of_log_feature_appeared',
                                   'total_volume_sum_of_log_feature',
                                   'event_type_count','resource_count']
features_to_be_considered = features_to_be_considered + additional_features_considered

dataset_train_sel_fea = dataset_train[features_to_be_considered]
dataset_test_sel_fea = dataset_test[features_to_be_considered]
dataset_train_sel_fea_id = dataset_train['id'].values
dataset_test_sel_fea_id = dataset_test['id'].values

if log1x:
    #log(1+x)
    dataset_train_sel_fea = dataset_train_sel_fea.apply(lambda x: x+1).apply(np.log10)
    #log(1+x)
    dataset_test_sel_fea = dataset_test_sel_fea.apply(lambda x: x+1).apply(np.log10)
        
    features_to_be_considered_log_x_1 = []
    for  i in features_to_be_considered :
        features_to_be_considered_log_x_1.append(i+'_log_1_x')

    log_1_x_train = pd.DataFrame(dataset_train_sel_fea.values,columns=features_to_be_considered_log_x_1)
    log_1_x_test = pd.DataFrame(dataset_test_sel_fea.values,columns=features_to_be_considered_log_x_1)
    log_1_x_train['id'] = dataset_train_sel_fea_id
    log_1_x_test['id'] = dataset_test_sel_fea_id
    
    dataset_train = pd.merge(dataset_train,log_1_x_train,on=['id'])
    dataset_test = pd.merge(dataset_test,log_1_x_test,on=['id'])
    
if scaleData:
    # standardize the data attributes
    dataset_train_sel_fea = preprocessing.scale(dataset_train_sel_fea)
    dataset_test_sel_fea = preprocessing.scale(dataset_test_sel_fea)
    
    features_to_be_considered_scale = []
    for  i in features_to_be_considered :
        features_to_be_considered_scale.append(i+'_scale')
        
    dataset_train_sel_fea = pd.DataFrame(dataset_train_sel_fea,columns=features_to_be_considered_scale)
    dataset_test_sel_fea = pd.DataFrame(dataset_test_sel_fea,columns=features_to_be_considered_scale)
    dataset_train_sel_fea['id'] = dataset_train_sel_fea_id
    dataset_test_sel_fea['id'] = dataset_test_sel_fea_id
    dataset_train = pd.merge(dataset_train,dataset_train_sel_fea,on=['id'])
    dataset_test = pd.merge(dataset_test,dataset_test_sel_fea,on=['id'])


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


if isKmeans :
    #================== KMeans ============================
    from sklearn.cluster import KMeans
    from sklearn.metrics import confusion_matrix
    
    features_to_be_considered = [features_to_be_considered_normal, 
                             features_to_be_considered_normal+features_to_be_considered_svd,
                             features_to_be_considered_svd ,
                             features_to_be_considered_log_1_x,
                             features_to_be_considered_scale, 
                             features_to_be_considered_log_1_x + features_to_be_considered_svd,
                             features_to_be_considered_scale + features_to_be_considered_svd
                             ] 
        
    df_ans_train = pd.DataFrame(dataset_train_index_val,columns=['id'])
    df_ans_test = pd.DataFrame(dataset_test_index_val,columns=['id'])
    
    for ww in range(0,len(features_to_be_considered)):
        
        dataset_train_sel_fea = ""
        dataset_test_sel_fea = ""
        dataset_train_sel_fea = dataset_train[features_to_be_considered[ww]]
        dataset_test_sel_fea  = dataset_test[features_to_be_considered[ww]]  
        
        dataset_train_sel_fea_val = dataset_train_sel_fea.values
        dataset_test_sel_fea_val = dataset_test_sel_fea.values
        
        kmeans = KMeans(3)
        kmeans_class_only_ans_train = kmeans.fit_predict(dataset_train_sel_fea)
        kmeans_class_only_ans_test = kmeans.predict(dataset_test_sel_fea)
        
        #confusion_matrix(dataset_train_ans_val, kmeans_class_only_ans)
        
        df_ans_train['kmeans_class_only_ans'+"_"+str(ww)] = kmeans_class_only_ans_train
        df_ans_test['kmeans_class_only_ans'+"_"+str(ww)] = kmeans_class_only_ans_test
        
        #print confusion_matrix( dataset_train_ans, kmeans_class_only_ans_train)
        
        for i in range(0,3):
            df_ans_train['kc_'+str(ww)+"_"+str(i+1)] = df_ans_train['kmeans_class_only_ans'+"_"+str(ww)].apply(lambda x : 1 if i == x else 0)
            df_ans_test['kc_'+str(ww)+"_"+str(i+1)] = df_ans_test['kmeans_class_only_ans'+"_"+str(ww)].apply(lambda x : 1 if i == x else 0)
        #dataset_train_sel_fea.drop(['kmeans_class_only_ans'],axis=1,inplace=True)
        
        print "kmeans round done" + str(ww)
        
dataset_train = pd.merge(dataset_train,df_ans_train,on=['id'])
dataset_test = pd.merge(dataset_test,df_ans_test,on=['id'])
    
    #================== KMeans Over============================


dataset_train.to_csv("dataset_train.csv")
dataset_test.to_csv("dataset_test.csv")

clms_we_have_train = list(dataset_train.columns.values)   
clms_we_have_test = list(dataset_test.columns.values)

print len(clms_we_have_train), len(clms_we_have_test)

file = open("all_feat_names.txt", "w")
for i in clms_we_have_train:
    file.write(i+"\n")

file.close()   


'''
dataset_train_index_val = dataset_train['id'].values
dataset_test_index_val = dataset_test['id'].values

dataset_train_ans = dataset_train['fault_severity']
dataset_train_ans_val = dataset_train_ans.values

dataset_train_sel_fea_val = dataset_train_sel_fea.values
dataset_test_sel_fea_val = dataset_test_sel_fea.values
'''    
print "Done With making data"
