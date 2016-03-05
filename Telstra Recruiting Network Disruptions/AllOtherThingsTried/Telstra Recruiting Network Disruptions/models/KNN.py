from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.grid_search import GridSearchCV

def setTrainTestDataAndCheckModel(X_train,Y_train,X_test,Y_test):
    print "called ....."
    knn = KNeighborsClassifier(n_neighbors=2,algorithm='ball_tree')
    knn.fit(X_train,Y_train)
        
    clf = GridSearchCV(knn,{'n_neighbors':[2,4,8,16,32,64,128,256],'algorithm':['ball_tree']},verbose=1)        
    clf.fit(X_train,Y_train)
    print(clf.best_score_)
    print(clf.best_params_)   
    
    output = clf.predict_proba(X_test)
    print log_loss(Y_test, output, eps=1e-15, normalize=True)
    print knn.score(X_train,Y_train)
    print "---------"
    
    
def setTrainDataAndMakeModel(X_train,Y_train,X_test):
    knn = KNeighborsClassifier(algorithm='auto', leaf_size=50, metric='euclidean',
           metric_params=None, n_jobs=1, n_neighbors=128, p=2,
           weights='uniform')
    knn.fit(X_train,Y_train)
    ##NO USE OF CALIBRATED CV OR BAGGING
    output = knn.predict_proba(X_test)
    return output
    