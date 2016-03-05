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

#===try out ==> only_location_OHE_with_id, location_svd_df
dataset_train = pd.merge(dataset_train, only_location_OHE_with_id, on =['id'])
dataset_test = pd.merge(dataset_test, only_location_OHE_with_id, on =['id'])

dataset_train = pd.merge(dataset_train, location_svd_df, on =['id'])
dataset_test = pd.merge(dataset_test, location_svd_df, on =['id'])


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

'''
#NORMALIZE DATA AND ADD TO TAL FEATURES
listOfColumnNamesForScaled  = []
for j in range(1,387):
    listOfColumnNamesForScaled.append('fs'+str(j))

all_ids =  log_feature_grouped_final['id'].values   
#Normalize data of OHE log features (volumes) 
log_feature_grouped_final_removed_id_tc = log_feature_grouped_final.drop(['id','total_number_of_log_feature_appeared','total_volume_sum_of_log_feature'],axis=1)
# normalize the data attributes (volumes)
normalized_values_OHE_log_features = preprocessing.scale(log_feature_grouped_final_removed_id_tc.values)
normalized_values_OHE_log_features_dataframe = pd.DataFrame(normalized_values_OHE_log_features,columns=listOfColumnNamesForScaled)
normalized_values_OHE_log_features_dataframe['total_vol_log_features_after_scaled'] = 0
for eachColumnName in listOfColumnNamesForScaled:
    normalized_values_OHE_log_features_dataframe['total_vol_log_features_after_scaled'] =normalized_values_OHE_log_features_dataframe['total_vol_log_features_after_scaled']  + normalized_values_OHE_log_features_dataframe[eachColumnName]
#feature =>add all features volumes 
normalized_values_OHE_log_features_dataframe['id'] = all_ids

log_feature_grouped_final = pd.merge(log_feature_grouped_final,normalized_values_OHE_log_features_dataframe,on=['id'])
'''

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
location
'''
location_f = []
#OHE
for l in range(1,1127):
    location_f.append('loc'+str(l))
#SVD
#for l in range(1,noOfDimensions+1):
#    location_f.append('svd_loc_'+str(l))  
'''
severity
'''
severity_f = []
for s in range(1,6):
    severity_f.append('severity_'+str(s))
'''
event type
'''
event_f = ['event_type_count'] 
for e in range(1,55):
    event_f.append('e'+str(e))
'''
log feature
'''
log_f = ['total_number_of_log_feature_appeared','total_volume_sum_of_log_feature']
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

features_to_be_considered = location_f + severity_f + event_f + log_f + resource_f

dataset_train_sel_fea = dataset_train[features_to_be_considered]
#log(1+x)
dataset_train_sel_fea = dataset_train_sel_fea.apply(lambda x: x+1).apply(np.log10)

dataset_train_sel_fea_val = dataset_train_sel_fea.values
dataset_train_index_val = dataset_train['id'].values
dataset_train_ans = dataset_train['fault_severity']
dataset_train_ans_val = dataset_train_ans.values
dataset_train_id_sevarity_Frame = dataset_train[['id','fault_severity']]

dataset_test_sel_fea = dataset_test[features_to_be_considered]
#log(1+x)
dataset_test_sel_fea = dataset_test_sel_fea.apply(lambda x: x+1).apply(np.log10)


dataset_test_sel_fea_val = dataset_test_sel_fea.values
dataset_test_index_val = dataset_test['id'].values
print "Done With making data"

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
lasagneStorer = []
randomForestStorer = []
MultinomialNBStorer = []
KNNStorer = []

#loop count
loopcount = 0 
for train_index, test_index in skf:
    #indexes on which working
    X_train_index, X_test_index = dataset_train_index_val[train_index], dataset_train_index_val[test_index]
    
    X_train, X_test = dataset_train_sel_fea_val[train_index], dataset_train_sel_fea_val[test_index]
    Y_train, Y_test = dataset_train_ans_val[train_index], dataset_train_ans_val[test_index]

    #===================== Algorithm to apply related (testing and tunning params)===============================
    '''
    #KNN
    KNN.setTrainTestDataAndCheckModel(X_train,Y_train,X_test,Y_test)
    
    #Random forest classifier    
    randomforestclassifier.setTrainTestDataAndCheckModel(X_train,Y_train,X_test,Y_test)
    
    #XGBOOST
    xgboostmodel.setTrainTestDataAndCheckModel(X_train,Y_train,X_test,Y_test)
    '''
    #====================== LEVEL 0 Models =======================
    print "====================== LEVEL 0 Models on X data loopcount: "+ str(loopcount) + " ======================="
    loopcount = loopcount + 1  
    
    print "XGBOOST"
    yprob = xgboostmodel.setTrainDataAndMakeModel(X_train, Y_train, X_test)
    print "%.4f" % log_loss(Y_test, yprob, eps=1e-15, normalize=True)
    outFrame = makeOutPutFrame(yprob,X_test_index,"XGBOOST")
    xgboostStorer.append(outFrame)

    print "LASAGNE"
    yprob = lasange.setTrainDataAndMakeModel(X_train, Y_train, X_test)
    print "%.4f" % log_loss(Y_test, yprob, eps=1e-15, normalize=True)
    outFrame = makeOutPutFrame(yprob,X_test_index,"LASAGNE")
    lasagneStorer.append(outFrame)

    print "RandomForest"
    yprob = randomforestclassifier.setTrainDataAndMakeModel(X_train, Y_train, X_test)
    print "%.4f" % log_loss(Y_test, yprob, eps=1e-15, normalize=True)
    outFrame = makeOutPutFrame(yprob,X_test_index,"RANDOM_FOREST")
    randomForestStorer.append(outFrame)
    
    print "KNN"
    yprob = KNN.setTrainDataAndMakeModel(X_train, Y_train, X_test)
    print "%.4f" % log_loss(Y_test, yprob, eps=1e-15, normalize=True)
    outFrame = makeOutPutFrame(yprob,X_test_index,"KNN")
    KNNStorer.append(outFrame)

    print "MultiNomialNB"
    yprob = MultinomialNB.setTrainDataAndMakeModel(X_train, Y_train, X_test)
    print "%.4f" % log_loss(Y_test, yprob, eps=1e-15, normalize=True)
    outFrame = makeOutPutFrame(yprob,X_test_index,"MultiNomialNB")
    MultinomialNBStorer.append(outFrame)

    
#====================== Between 0 and 1 Models (join all X frames) =======================
print "#====================== Between 0 and 1 Models (join all X frames) ======================="
#all chounks of frames concat for respective.
xgboostMetaFeaturesOfX = pd.concat(xgboostStorer) 
lasangeMetaFeaturesOfX = pd.concat(lasagneStorer)
randonforestMetaFeaturesOfX = pd.concat(randomForestStorer)
knnMetaFeaturesOfX = pd.concat(KNNStorer)
multinomialNBMetaFeaturesOfX = pd.concat(MultinomialNBStorer)
 
##join all above frames
allMetaFeaturesOfX = pd.merge(xgboostMetaFeaturesOfX,lasangeMetaFeaturesOfX,on=['id']).merge(randonforestMetaFeaturesOfX,on=['id']).merge(knnMetaFeaturesOfX,on=['id']).merge(multinomialNBMetaFeaturesOfX,on=['id']).merge(dataset_train_id_sevarity_Frame,on=['id'])
##save to csv (safe side)
allMetaFeaturesOfX.set_index('id',inplace=True)
allMetaFeaturesOfX.to_csv("allMetaFeaturesOfX.csv")

print "====================== LEVEL 0 Models on Y data======================="
outputFramesStore = []

print "XGBOOST"
yprob = xgboostmodel.setTrainDataAndMakeModel(dataset_train_sel_fea_val, dataset_train_ans_val,dataset_test_sel_fea_val)
outFrame = makeOutPutFrame(yprob,dataset_test_index_val,"XGBOOST")
outputFramesStore.append(outFrame)
        
print "LASAGNE"
yprob = lasange.setTrainDataAndMakeModel(dataset_train_sel_fea_val, dataset_train_ans_val,dataset_test_sel_fea_val)
outFrame = makeOutPutFrame(yprob,dataset_test_index_val,"LASAGNE")
outputFramesStore.append(outFrame)

print "RandomForest"
yprob = randomforestclassifier.setTrainDataAndMakeModel(dataset_train_sel_fea_val, dataset_train_ans_val,dataset_test_sel_fea_val)
outFrame = makeOutPutFrame(yprob,dataset_test_index_val,"RANDOM_FOREST")
outputFramesStore.append(outFrame)

print "KNN"
yprob = KNN.setTrainDataAndMakeModel(dataset_train_sel_fea_val, dataset_train_ans_val,dataset_test_sel_fea_val)
outFrame = makeOutPutFrame(yprob,dataset_test_index_val,"KNN")
outputFramesStore.append(outFrame)

print "MultiNomialNB"
yprob = MultinomialNB.setTrainDataAndMakeModel(dataset_train_sel_fea_val, dataset_train_ans_val,dataset_test_sel_fea_val)
outFrame = makeOutPutFrame(yprob,dataset_test_index_val,"MultiNomialNB")
outputFramesStore.append(outFrame)

#====================== Between 0 and 1 Models (join all Y frames) =======================
print "====================== Between 0 and 1 Models (join all Y frames) ======================="

outputFinalFrameForLEVEL2 = pd.DataFrame({'id': dataset_test_index_val})
for eachFrame in outputFramesStore :
    outputFinalFrameForLEVEL2 = pd.merge(outputFinalFrameForLEVEL2,eachFrame,on=['id'])
outputFinalFrameForLEVEL2.set_index('id',inplace=True)
outputFinalFrameForLEVEL2.to_csv("allMetaFeaturesOfY.csv")

#====================== LEVEL 2 Models =======================
'''
print "Final Model"
yprob = xgboostmodel.setTrainDataAndMakeModel( dataset_train_sel_fea_val, dataset_train_ans_val,dataset_test_sel_fea_val)
outFrame = makeOutPutFrame(yprob,"XGBOOST")
outputFramesStore.append(outFrame)
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
https://github.com/christophebourguignat/notebooks/blob/master/Calibration.ipynb
https://github.com/christophebourguignat/notebooks/blob/master/Tuning%20Neural%20Networks.ipynb

What is bagging/boosting/stacking and how to do it?

How to tune?
=>https://www.youtube.com/watch?v=Og7CGAfSr_Y

#Scaling features 
Standard Scalar in scikit learn

https://medium.com/@chris_bour/6-tricks-i-learned-from-the-otto-kaggle-challenge-a9299378cd61#.vjhd0pgfc

sklearn.manifold.TSNE or (PCA for dense data) or (TruncatedSVD for sparse data) 
'''

'''
2.TWO layer model
ADABOOST
3.More Parameters
'''