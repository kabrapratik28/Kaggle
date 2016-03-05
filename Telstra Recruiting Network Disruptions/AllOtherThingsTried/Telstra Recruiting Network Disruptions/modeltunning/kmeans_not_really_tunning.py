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

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

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

features_to_be_considered = [features_to_be_considered_normal, 
                             features_to_be_considered_normal+features_to_be_considered_svd,
                             features_to_be_considered_svd ,
                             features_to_be_considered_log_1_x,
                             features_to_be_considered_scale, 
                             features_to_be_considered_log_1_x + features_to_be_considered_svd,
                             features_to_be_considered_scale + features_to_be_considered_svd
                             ] 

df_ans = pd.DataFrame(dataset_train_index_val,columns=['id'])

for ww in range(0,len(features_to_be_considered)):
    
    dataset_train_sel_fea = ""
    dataset_test_sel_fea = ""
    dataset_train_sel_fea = train[features_to_be_considered[ww]]
    dataset_test_sel_fea  = test[features_to_be_considered[ww]]  
    
    dataset_train_sel_fea_val = dataset_train_sel_fea.values
    dataset_test_sel_fea_val = dataset_test_sel_fea.values
    
    
        
    kmeans = KMeans(3)
    kmeans_class_only_ans_train = kmeans.fit_predict(dataset_train_sel_fea)
    kmeans_class_only_ans_test = kmeans.predict(dataset_test_sel_fea)
    
    #confusion_matrix(dataset_train_ans_val, kmeans_class_only_ans)
    
    df_ans['kmeans_class_only_ans'+"_"+str(ww)] = kmeans_class_only_ans_train
    print confusion_matrix( dataset_train_ans, kmeans_class_only_ans_train)
    
    #kmeans_class_distance_ans = kmeans.transform(dataset_train_sel_fea)
    #df_ans['kcd1_'+str(ww)] = kmeans_class_distance_ans[:,0]
    #df_ans['kcd2_'+str(ww)] = kmeans_class_distance_ans[:,1]
    #df_ans['kcd3_'+str(ww)] = kmeans_class_distance_ans[:,2]
    for i in range(0,3):
        df_ans['kc_'+str(ww)+"_"+str(i+1)] = df_ans['kmeans_class_only_ans'+"_"+str(ww)].apply(lambda x : 1 if i == x else 0)
       
    #dataset_train_sel_fea.drop(['kmeans_class_only_ans'],axis=1,inplace=True)
    
    print "done" + str(ww)
    
#df_ans.to_csv("kmeans_out.csv")