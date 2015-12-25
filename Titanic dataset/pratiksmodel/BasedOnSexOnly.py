"""
Created on Fri Dec 25 21:12:32 2015

@author: pshrikantkab
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

train['Sex'] = train['Sex'].map(lambda x: 0 if x=='female' else 1 )


clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(train[['Sex']].values, train['Survived'].values)

test['Sex'] = test['Sex'].map(lambda x: 0 if x=='female' else 1 )
predication = clf.predict(test[['Sex']].values)

print metrics.accuracy_score(predication, gtest['Survived'].values)

print metrics.accuracy_score(predication, gctest['Survived'].values)
