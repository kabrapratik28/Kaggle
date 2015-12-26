# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 12:37:40 2015

@author: pshrikantkab
"""
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

#read sfcc train set
train = pd.read_csv('train.csv')

#read sfcc test set
test = pd.read_csv('test.csv')

#unique categories of crimes #39
#train['Category'].unique()
#train['PdDistrict'].unique()

#drop not required columns
train.drop(['Descript','Resolution','Address'],axis=1,inplace=True)

#convert to datetime format
train['Dates'] =  pd.to_datetime(train['Dates'])
test['Dates'] =  pd.to_datetime(test['Dates'])

train = train.set_index('Dates')
test = test.set_index('Dates')


'''
Data Care about [DayOfWeek, PdDistrict, Hour] => [ Category]
'''


'''
Encoding of text data into numbers. 
'''
#label encoding for category
le = preprocessing.LabelEncoder()
CategoryEncoded = le.fit_transform(train['Category'].values)
train['CategoryEncoded'] = CategoryEncoded


#day Of week encoding
le2 = preprocessing.LabelEncoder()
DayEncoder = le2.fit_transform(train['DayOfWeek'].values)
train['DayEncoded'] = DayEncoder
DayEncoder = le2.transform(test['DayOfWeek'].values)
test['DayEncoded'] = DayEncoder


#PdDistrict encoding
le3 = preprocessing.LabelEncoder()
PdDistrictEncoder = le3.fit_transform(train['PdDistrict'].values)
train['PdDistrictEncoded'] = PdDistrictEncoder
PdDistrictEncoder = le3.transform(test['PdDistrict'].values)
test['PdDistrictEncoded'] = PdDistrictEncoder


'''
Normalize data
'''
features = ['DayEncoded','PdDistrictEncoded']
for eachFea in features :
    sc = preprocessing.StandardScaler()
    train[eachFea] = sc.fit_transform(train[eachFea].values)
    test[eachFea] = sc.transform(test[eachFea].values)

sc = preprocessing.StandardScaler()
train['Hour'] = sc.fit_transform(train.index.hour)
test['Hour'] = sc.transform(test.index.hour)
    

'''
#plot no of categories happened as per day time
train['count'] = 1
plotFrame = pd.DataFrame({'hour':train.index.hour,'category':train['CategoryEncoded'],'noofcrime':train['count']})
plotFrame = plotFrame.groupby(['hour','category']).sum()
plotFrame.plot()  #each hour vise and each catogory
'''


'''
KNN classifier used 
trained by hour, pddistrict and day of week.
'''
neigh = KNeighborsClassifier(n_neighbors=100, weights='uniform', algorithm='auto', leaf_size=100, p=10, metric='minkowski')

neigh.fit(train[['Hour','DayEncoded','PdDistrictEncoded']].values,train['CategoryEncoded'].values)

answer = neigh.predict(test[['Hour','DayEncoded','PdDistrictEncoded']].values)