import numpy as np
import GlobalVariables
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle


def learn(X, y, n_fold=10, kernel='linear'):
    
    X,y = shuffle(X,y)
    skf = StratifiedKFold(n_splits=n_fold)
    accuracies = []
    
    for train_index, test_index in skf.split(X, y):
        
        X_train = np.array(X)[train_index.astype(int)]
        X_test = np.array(X)[test_index.astype(int)]
        y_train = np.array(y)[train_index.astype(int)]
        y_test = np.array(y)[test_index.astype(int)]
        
        # define grid parameters
        params = None
        lin_C_params = [2**i for i in [-12,-8,-5,-3,-1,1,3,5,8,12]]
        lin_params = {'C': lin_C_params} 
        rbf_C_params = [2**i for i in [-12,-8,-5,-3,-1,1,3,5,8,12]]
        rbf_gamma_params = [2**i for i in [-12,-8,-5,-3,-1,1,3,5,8,12]]
        rbf_params = {'C': rbf_C_params, 'gamma': rbf_gamma_params}
        if kernel == 'rbf': params = rbf_params
        elif kernel == 'linear': params = lin_params
        
        # setup svm and perform classification
        svc = SVC(kernel=kernel, max_iter=1000000)
        clf = GridSearchCV(svc, params, scoring='accuracy', cv=3, n_jobs=GlobalVariables.threads) 
        clf.fit(X_train, y_train)
        
        # predict
        y_pred = clf.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        
    # return    
    accs = np.array(accuracies)
    return accs, np.mean(accs), np.std(accs)    
    
    
    
    
    
    
    
