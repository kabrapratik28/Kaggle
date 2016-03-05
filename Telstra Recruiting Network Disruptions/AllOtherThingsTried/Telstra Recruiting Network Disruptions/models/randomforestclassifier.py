from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier 
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

def setTrainTestDataAndCheckModel(X_train,Y_train,X_test,Y_test):
    model = RandomForestClassifier(125)
    model.fit(X_train,Y_train)
    '''
    clf = GridSearchCV(model,{'n_estimators':[100,125,150]},verbose=1)
    
    clf.fit(X_train,Y_train)
    print(clf.best_score_)
    print(clf.best_params_)    
    
    output = model.predict(X_test)
    print "-------------------RFC-----------------------"
    #print accuracy_score(Y_test,output)
    #print "%.2f" % log_loss(Y_test,output, eps=1e-15, normalize=True)
    
    ypreds = model.predict_proba(X_test)
    print "%.2f" % log_loss(Y_test,ypreds, eps=1e-15, normalize=True)

    
    clfbag = BaggingClassifier(model, n_estimators=5)
    clfbag.fit(X_train, Y_train)
    ypreds = clfbag.predict(X_test)    
    #print accuracy_score(Y_test,ypreds)    
    
    ypreds = clfbag.predict_proba(X_test)
    print "%.2f" % log_loss(Y_test,ypreds, eps=1e-15, normalize=True)
    '''
    calibrated_clf = CalibratedClassifierCV(model, method='isotonic', cv=5)
    calibrated_clf.fit(X_train, Y_train)
    #ypreds = calibrated_clf.predict(X_test)
    #print accuracy_score(Y_test,ypreds)
    
    ypreds = calibrated_clf.predict_proba(X_test)
    print "%.2f" % log_loss(Y_test, ypreds, eps=1e-15, normalize=True)

def setTrainDataAndMakeModel(X_train,Y_train,X_test):
    model =  RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
            max_depth=27, max_features='log2', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=5,
            min_weight_fraction_leaf=0.0, n_estimators=25, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)       

    model.fit(X_train,Y_train)
    calibrated_clf = CalibratedClassifierCV(model, method='isotonic', cv=5)
    calibrated_clf.fit(X_train, Y_train)
    ypreds = calibrated_clf.predict_proba(X_test)
    return ypreds
    