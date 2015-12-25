# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 00:16:17 2015

@author: pshrikantkab
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn import metrics

#read titanic train set
train = pd.read_csv('../train.csv')
#print train.head()

#read titanic test set
test = pd.read_csv('../test.csv')

#read gender model answer set
gtest = pd.read_csv('../gendermodel.csv')

#read gender class model answer set
gctest =  pd.read_csv('../genderclassmodel.csv')

#sexOfPeople = train.Sex
#print type(sexOfPeople)

# male =1 female =0
train['Sex'] = train['Sex'].map(lambda x: 0 if x=='female' else 1 )

listOfClmToBeConsidered = ['Sex','Pclass','Fare']

clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(train[listOfClmToBeConsidered].values, train['Survived'].values)

#sex is not missing so converted.
test['Sex'] = test['Sex'].map(lambda x: 0 if x=='female' else 1 )

#test contains nan value in fair
#use Imputer
#more info => row no. 152
#replace nan => 35.6271884892
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(test[listOfClmToBeConsidered].values)
testingData = imp.transform(test[listOfClmToBeConsidered].values)


predication = clf.predict(testingData)

print metrics.accuracy_score(predication, gtest['Survived'].values)

print metrics.accuracy_score(predication, gctest['Survived'].values)
