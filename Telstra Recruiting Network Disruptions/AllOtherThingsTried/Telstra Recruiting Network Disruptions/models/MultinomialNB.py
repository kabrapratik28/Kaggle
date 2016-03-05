from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier 
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

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

def setTrainDataAndMakeModel(X_train,Y_train,X_test):
    clf = MultinomialNB(alpha=125535, class_prior=None, fit_prior=True)
    calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=5)
    calibrated_clf.fit(X_train, Y_train)
    ypreds = calibrated_clf.predict_proba(X_test)    
    return ypreds
    