""" 
    Udacity Lesson 1 (Naive Bayes) mini-project. 
    Also Updating the base code to work with Python 3 , updating changed dependencies 
    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


sys.path.append("/Users/vipul/Work/Github/ml-projects/udacity/tools")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

nb = GaussianNB()
t0 = time()
nb.fit(features_train, labels_train)
print(f"training time : {round(time() - t0 ,3)}s")

y_predict = nb.predict(features_test)

accuracy = accuracy_score(y_true=labels_test, y_pred=y_predict)
print(f"Accuracy score is {accuracy}")
