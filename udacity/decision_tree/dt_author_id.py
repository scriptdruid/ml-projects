""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


sys.path.append("/Users/vipul/Work/Github/ml-projects/udacity/tools")
from email_preprocess import preprocess

features_train, features_test, labels_train, labels_test = preprocess()

clf = DecisionTreeClassifier(min_samples_split=40)

print(len(features_train[0]))
t0 = time()
clf.fit(features_train, labels_train)

print(f"training time : {round(time() - t0 ,3)}s")

pred = clf.predict(features_test)

accuracy = accuracy_score(labels_test, pred)

print(f"Accuracy of svm is : {accuracy}")

# When using percentile = 10 or 3785 features

# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# training time : 68.044s
# Accuracy of svm is : 0.9783845278725825

# When using percentile = 1 or only 379 features

# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# 379
# training time : 4.284s
# Accuracy of svm is : 0.9670079635949943
