from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier 
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

def setTrainTestDataAndCheckModel(X_train,Y_train,X_test,Y_test):
    model = RandomForestClassifier(100)
    model.fit(X_train,Y_train)

    clf = GridSearchCV(model,{'n_estimators':[50,100,200]},verbose=1)
    
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

    calibrated_clf = CalibratedClassifierCV(model, method='isotonic', cv=5)
    calibrated_clf.fit(X_train, Y_train)
    ypreds = calibrated_clf.predict(X_test)
    #print accuracy_score(Y_test,ypreds)
    
    ypreds = calibrated_clf.predict_proba(X_test)
    print "%.2f" % log_loss(Y_test, ypreds, eps=1e-15, normalize=True)