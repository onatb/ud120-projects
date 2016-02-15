#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

from sklearn import tree
import numpy as np
from sklearn import cross_validation
#from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features,labels,test_size=0.30,random_state=42)

clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

#print accuracy_score(pred, labels_test)

tp = np.zeros(29)
for i in range(0,29):
    if pred[i]==1 and features_test[i]==1:
        tp[i] = 1


print "Recall: ", recall_score(tp,labels_test)
print "Precision: ", precision_score(tp,pred)
