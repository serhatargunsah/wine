import numpy as np
from nbc import NaiveBayesDiscrete

def read_wine_data(filepath):
    D = np.genfromtxt(filepath, delimiter=",")
    y = D[:, -1].astype(np.int8)
    X = D[:, :-1]
    return X, y

def train_test_split(X, y, train_fraction=0.75, seed=0):
    np.random.seed(seed)
    m = X.shape[0] # m - no. of examples
    indexes = np.random.permutation(m)    
    X = X[indexes]
    y = y[indexes]
    i = int(np.round(train_fraction * m))
    X_train = X[:i]
    y_train = y[:i]
    X_test = X[i:]
    y_test = y[i:]
    return X_train, y_train, X_test, y_test

def discretize(X, bins=5, mins_ref=None, maxes_ref=None):
    if mins_ref is None:
        mins_ref = np.min(X, axis=0)
        maxes_ref = np.max(X, axis=0)
    X_d = np.clip(np.floor((X - mins_ref) / (maxes_ref - mins_ref) * bins), 0, bins - 1).astype(np.int8)
    return X_d, mins_ref, maxes_ref 
            
if __name__ == '__main__':
    laplace_control = bool(int(input("enter laplace value (1(True)/0(False))")))
    log_control = bool(int(input("Do you want to use log calculation (1(Yes)/0(No))")))
    BINS = int(input("enter BINS VALUE: "))
    X, y = read_wine_data("waveform.data") 
    #X=np.tile(X,(1,50)) #I added tile
    n = X.shape[1]
    X_train, y_train, X_test, y_test = train_test_split(X, y, train_fraction=0.75, seed=2)
    X_train_d, mins_ref, maxes_ref = discretize(X_train, bins=BINS)    
    X_test_d, _, _ = discretize(X_test, bins=BINS, mins_ref=mins_ref, maxes_ref=maxes_ref)
    domain_sizes = BINS * np.ones(n, dtype=np.int8) # just for "wine" data
    clf = NaiveBayesDiscrete(domain_sizes, laplace=laplace_control, log_choice=log_control) 
    clf.fit(X_train_d, y_train)
    #print(f"ACC TRAIN: {np.mean(y_train == clf.predict(X_train_d))}")
    print(f"ACC TRAIN: {clf.score(X_train_d, y_train)}")
    print(f"ACC TEST: {clf.score(X_test_d, y_test)}")
    print(clf.PY_)
    
    
    