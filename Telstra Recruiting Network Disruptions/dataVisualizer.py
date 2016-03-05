import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns; sns.set(style="white", color_codes=True)

#dimension reduction
from sklearn.decomposition import TruncatedSVD
#preprocessing
from sklearn import preprocessing

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
 
'''
#reduce dimension to 100 using SVD
noOfDimensions = 100
svd = TruncatedSVD(n_components=noOfDimensions, random_state=10)
svd_loc = svd.fit_transform(only_location_OHE.values)
# NAN chceking
#    train_test_id_location.isnull().values.any()

indexName = []
for i  in range(1,noOfDimensions+1):
    indexName.append('svd_loc_'+str(i))
    
location_svd_df = pd.DataFrame(svd_loc,columns=indexName)


location_svd_df['id'] = train_test_id_location['id'].values
dataset_train = pd.merge(dataset_train, location_svd_df, on =['id'])
dataset_test = pd.merge(dataset_test, location_svd_df, on =['id'])
'''

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
'''
#sort event type by id ##FOR VISUALIZING DIDN't WOTKED FEATURED UNCOMMENT BELOW 3 Lines
event_type_sorted = event_type.sort(['id'])
event_type_sorted_with_fault_sev = pd.merge(event_type_sorted,dataset_train[['id','fault_severity']],on=['id'])
event_type_sorted_with_fault_sev.drop(["event_type" , "event_type_count"],axis=1,inplace=True)
'''

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

#sum of all events
#event_id_and_list['event_type_sum'] = event_id_and_list['event_type_only'].apply(lambda x : sum(x))

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
    
#sum of product of log fea and vol.
log_feature_grouped["sum_of_prod_of_log_fea"] = log_feature_grouped["log_and_volume_all"].apply(lambda x: sum_of_prod_of_log_vol(x))
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

#SVD
#for l in range(1,noOfDimensions+1):
#    location_f.append('svd_loc_'+str(l))  
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
log_f = ['total_number_of_log_feature_appeared','total_volume_sum_of_log_feature','sum_of_prod_of_log_fea']
for l in range(1,387):
    log_f.append('f'+str(l))

#for scaled version
#log_f = ['total_number_of_log_feature_appeared','total_vol_log_features_after_scaled']
#for l in range(1,387):
#    log_f.append('fs'+str(l))
    
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

'''
#================== KMeans ============================
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

kmeans = KMeans(3)
kmeans_class_only_ans = kmeans.fit_predict(dataset_train_sel_fea)
#confusion_matrix(dataset_train_ans_val, kmeans_class_only_ans)
kmeans_class_distance_ans = kmeans.transform(dataset_train_sel_fea)

dataset_train_sel_fea['kmeans_class_only_ans'] = kmeans_class_only_ans
dataset_train_sel_fea['kcd1'] = kmeans_class_distance_ans[:,0]
dataset_train_sel_fea['kcd2'] = kmeans_class_distance_ans[:,1]
dataset_train_sel_fea['kcd3'] = kmeans_class_distance_ans[:,2]
for i in range(0,3):
    dataset_train_sel_fea['kc'+str(i+1)] = dataset_train_sel_fea['kmeans_class_only_ans'].apply(lambda x : 1 if i == x else 0)

dataset_train_sel_fea.drop(['kmeans_class_only_ans'],axis=1,inplace=True)

#================== KMeans Over============================
'''
'''
dataset_train_sel_fea_val = dataset_train_sel_fea.values
dataset_train_index_val = dataset_train['id'].values
dataset_train_ans = dataset_train['fault_severity']
dataset_train_ans_val = dataset_train_ans.values
dataset_train_id_sevarity_Frame = dataset_train[['id','fault_severity']]

dataset_test_sel_fea_val = dataset_test_sel_fea.values
dataset_test_index_val = dataset_test['id'].values
'''

#plotData = sns.jointplot(x="location", y="fault_severity", data=dataset_train)
df = dataset_train[["fault_severity","severity_type_only","location","event_type_count","resource_count","sum_of_prod_of_log_fea"]]
#joint plot of all the things matrix like printing... 
#plotData = sns.pairplot(df,  hue="fault_severity")

plotData = sns.jointplot("sum_of_prod_of_log_fea", "fault_severity", data=df)
plotData = sns.lmplot("sum_of_prod_of_log_fea", "fault_severity", hue="fault_severity", data=df, fit_reg=False,x_jitter=0.0,y_jitter=0.2)

#display outside in new window...
# %matplotlib qt                    

print "Done With making data"

#======================= ALL PLOTS OF DATA ========================
#print location plot as joint plot stats up down
##plotData = sns.jointplot(x="fault_severity", y="location", data=df)
##plotData = sns.lmplot("fault_severity", "location", hue="fault_severity", data=df, fit_reg=False,x_jitter=.2)

#severity_greater_than_eq_3_has_only_f0_and_1
#plotData = sns.jointplot(x="severity_type_only", y="fault_severity", data=df)
#plotData = sns.lmplot("severity_type_only", "fault_severity", hue="fault_severity", data=df, fit_reg=False,x_jitter=.2,y_jitter=.2)

#event between 25 and 40 has very less severity 3  ##DIDN'T WORK FEATURED
## ***** WILL ONLY WORK ABOVE #
#sns.jointplot(x="event_type_only", y="fault_severity", data=event_type_sorted_with_fault_sev)
#sns.lmplot("event_type_only", "fault_severity", hue="fault_severity", data=event_type_sorted_with_fault_sev, fit_reg=False,x_jitter=0.0,y_jitter=.2)

#log feature between 225 and 270 has very less dault sev 2 ##DIDN'T WORKED
#plotData = sns.lmplot("log_feature_only", "fault_severity", hue="fault_severity", data=data, fit_reg=False,x_jitter=0.1,y_jitter=.2)

#between feat no (225 and 315) and vol(above 24) only  (how many  out of total) ##DIDN'T WRKED
#id log_feature  volume  log_feature_only log_and_volume_all fault_severity
#data = pd.merge(log_feature,dataset_train[['id','fault_severity']],on=['id'])
#plotData = sns.jointplot(x="log_feature_only", y="volume", data=data)
#plotData = sns.lmplot("log_feature_only", "volume", hue="fault_severity", data=data, fit_reg=False,x_jitter=0.1,y_jitter=.2)
'''
from collections import Counter
def vol_app_max(listOfTup):
    listOfVol = []
    for i in listOfTup:
        listOfVol.append(i[1])
    counter = Counter(listOfVol)
    max_count = max(counter.values())
    mode = [k for k,v in counter.items() if v == max_count]
    return reduce(lambda x, y: x + y, mode) / float(len(mode))
    
log_feature_grouped['max_log_vol_app'] = log_feature_grouped['log_and_volume_all'].apply(lambda x: vol_app_max(x))
plotData = sns.lmplot("max_log_vol_app", "fault_severity", hue="fault_severity", data=df, fit_reg=False,x_jitter=0.1,y_jitter=.2)
'''
'''
##select k main features
resource_type_train = pd.merge(resource_type,train[['id','fault_severity']],on=['id'])
k_best_for_resource=5
k_best_resource = []
for i in range(0,k_best_for_resource):
    k_best_resource.append('res_k_'+str(i))
    
selectK = SelectKBest(chi2, k=k_best_for_resource)
selectK.fit(resource_type_train.drop(['id','fault_severity'],axis=1),resource_type_train['fault_severity'])
resource_type_k_best_answers = selectK.transform(resource_type.drop(['id'],axis=1))
resource_type_k_best_df = pd.DataFrame(resource_type_k_best_answers,columns=k_best_resource)
resource_type_k_best_df['id'] = resource_type['id']

dataset_train = pd.merge(dataset_train,resource_type_k_best_df,on=['id'])
dataset_test = pd.merge(dataset_test,resource_type_k_best_df,on=['id'])
'''
'''
def check_event_cou_and_res_count(x,y):
    if(x<=3 and y<=1):
        return 1 
    else:
        return 0
        
dataset_train['eve_type_res_count'] = dataset_train.apply(lambda x : 1 if x["event_type_count"]<=3 and x["resource_count"]<=1 else 0, axis=1)
dataset_test['eve_type_res_count'] = dataset_test.apply(lambda x : 1 if x["event_type_count"]<=3 and x["resource_count"]<=1 else 0, axis=1)
#resource count vs event type count 
plotData = sns.jointplot(x="event_type_count", y="resource_count", data=df)
plotData = sns.lmplot("event_type_count", "resource_count", hue="fault_severity", data=df, fit_reg=True,x_jitter=.2,y_jitter=.2)
'''
##SVD tried on location function
'''
plotData = sns.jointplot(x="severity_type_only", y="event_type_count", data=df)
plotData = sns.lmplot("severity_type_only", "event_type_count", hue="fault_severity", data=df, fit_reg=True,x_jitter=.2,y_jitter=.2)
'''