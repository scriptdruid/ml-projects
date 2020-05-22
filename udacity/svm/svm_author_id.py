""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
import collections

sys.path.append("/Users/vipul/Work/Github/ml-projects/udacity/tools")
from email_preprocess import preprocess

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# Slicing the data to make it run faster
features_train = features_train[: int(len(features_train) // 100)]
labels_train = labels_train[: int(len(labels_train) // 100)]

clf = SVC(kernel="rbf", C=10000)

t0 = time()

clf.fit(features_train, labels_train)

print(f"training time : {round(time() - t0 ,3)}s")

pred = clf.predict(features_test)

# Getting no of both 0 and 1 in output
print(collections.Counter(pred))

accuracy = accuracy_score(labels_test, pred)

print(f"Accuracy of svm is : {accuracy}")


#  ~/W/G/ml-projects/udacity/naive_bayes | master !2 ?2  /Users/vipul/opt/anaconda3/envs/ml/bin/python /Users/vipul/Work/Github/ml-projects/udacity/naive_bayes/naive_bayes_author.py
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# training time : 1.715s
# Accuracy of NB  is 0.9732650739476678
#  ~/W/G/ml-projects/u/naive_bayes | master !2 ?2  /Users/vipul/opt/anaconda3/envs/ml/bin/python /Users/vipul/Work/Github/ml-projects/udacity/svm/svm_author_id.py
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# Accuracy of svm is : 0.9840728100113766 #too slow
