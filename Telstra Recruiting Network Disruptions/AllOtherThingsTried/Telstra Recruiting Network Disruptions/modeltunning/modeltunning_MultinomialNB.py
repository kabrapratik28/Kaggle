import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import  BaggingClassifier 
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import randint as sp_randint
from sklearn.naive_bayes import MultinomialNB
from time import time
import random
#our model and imports
from models import xgboostmodel
from models import randomforestclassifier
from models import KNN

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

'''
#===============================RANDOM SEARCH================================
# build a classifier
clf = MultinomialNB()
# specify parameters and distributions to sample from
param_dist = {"alpha": sp_randint(70000,2000000),
              "fit_prior":[True,False]
              }

# run randomized search
n_iter_search = 10

start = time()
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search,scoring='log_loss')

random_search.fit(dataset_train_sel_fea_val, dataset_train_ans_val )

print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))

with open("MultinomialNBbest.txt", "w") as myfile:
    myfile.write("======random forest=========\n")
    myfile.write(str(random_search.best_params_)+"\n")
    myfile.write(str(random_search.best_estimator_)+"\n")
    myfile.write(str(random_search.best_score_)+"\n")
    myfile.write("======random forest=========\n")
myfile.close()


#===============================RANDOM SEARCH================================
'''
#train test split
X_train, X_test, Y_train,Y_test = train_test_split(dataset_train_sel_fea,dataset_train_ans, train_size=0.75)


clf = MultinomialNB(alpha=125535, class_prior=None, fit_prior=True)
clf.fit(X_train, Y_train)
ypreds = clf.predict_proba(X_test)    
print "%.2f" % log_loss(Y_test,ypreds, eps=1e-15, normalize=True)

clfbag = BaggingClassifier(clf, n_estimators=5)
clfbag.fit(X_train, Y_train)
ypreds = clfbag.predict_proba(X_test)    
print "%.2f" % log_loss(Y_test,ypreds, eps=1e-15, normalize=True)

calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=5)
calibrated_clf.fit(X_train, Y_train)
ypreds = calibrated_clf.predict_proba(X_test)    
print "%.2f" % log_loss(Y_test,ypreds, eps=1e-15, normalize=True)