from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class NaiveBayesDiscrete(BaseEstimator, ClassifierMixin):

    def __init__(self, domain_sizes, laplace, log_choice): #log on of ekle #I added a new arg to control log
        self.class_labels_ = None # to memorize actual names/labels of classes
        self.PY_ = None # a 1-D array with a priori probabilities of classes
        self.P_ = None # a 3-D array with all conditional probabilities; self.P_[1, 7, 2] = Pr(X_7 = 2 | Y = 1)
        self.domain_sizes_ = domain_sizes
        self.laplace_ = laplace
        self.log_choice_ = log_choice
    
    def fit(self, X, y):
        self.class_labels_ = np.unique(y)
        K = self.class_labels_.size # no. of classes  
        m, n = X.shape # m - no. of examples, n - no. of vars
        yy = np.zeros(m, dtype=np.int8) # class labels mapped to numbers: 0, 1, 2, ...
        self.PY_ = np.zeros(K)
        for index, label in enumerate(self.class_labels_):
            indexes = y == label 
            yy[indexes] = index
            self.PY_[index] = np.mean(indexes)
        q = np.max(self.domain_sizes_) # q - max_domain_size     
        self.P_ = np.zeros((K, n, q))
        for i in range(m):
            for j in range(n):
                self.P_[yy[i], j, X[i, j]] += 1        
        for k in range(K):
            if not self.laplace_:
                self.P_[k] /= self.PY_[k] * m
            else:
                for j in range(n):
                    self.P_[k, j] = (self.P_[k, j] + 1) / (self.PY_[k] * m + self.domain_sizes_[j])  
        print(self.P_)
            
    def predict(self, X):
        return self.class_labels_[np.argmax(self.predict_proba(X), axis=1)]
    
    def predict_proba(self, X):
        m, n = X.shape
        K = self.class_labels_.size
        probas = np.zeros((m, K))
        for i in range(m):            
            for k in range(K):
                if not self.log_choice_:
                    probas[i, k] =self.PY_[k] #log ekle 
                elif self.log_choice_:
                    probas[i, k] =np.log(self.PY_[k]) #log ekle #I added log
                for j in range(n):
                    if not self.log_choice_:
                        probas[i, k] *=self.P_[k, j, X[i, j]]#log ekle ve * yı + yap
                    elif self.log_choice_:
                        probas[i, k] +=np.log(self.P_[k, j, X[i, j]])#log ekle ve * yı + yap # I added log
        return probas