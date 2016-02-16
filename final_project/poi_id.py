#!/usr/bin/python

import sys
import pickle
from time import time
import numpy as np
from collections import Counter
from pprint import pprint

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features

# More features will be added to 'features list' in the following parts
features_list = ['poi'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Store to my_dataset for easy export below.
my_dataset = data_dict


# Data exploration part

# Total number of data points
print "Total # Data Points: ", len(my_dataset.values())

# Allocation across classes (POI/non-POI)
num_poi = 0
for name in my_dataset:
    if my_dataset[name]['poi'] == True:
        num_poi += 1

print "# POI's: ", num_poi
print "# Non-POI's: ", len(my_dataset.values()) - num_poi

# Get all features of 'my_dataset'
all_features = my_dataset.values()[0].keys()

# Number of features
print "Total # Features: ", len(all_features)
    
# Remove 'email_address' feature from 'all_features'
# since it doesn't carry any information about poi's
all_features.pop(all_features.index('email_address'))

# It is stated that "The first feature must be 'poi'"
ix_poi = all_features.index('poi')
all_features[0], all_features[ix_poi] = all_features[ix_poi], all_features[0]

# Get numpy multidimensional array using 'all_features'
data = featureFormat(my_dataset, all_features, sort_keys = True)


### Task 2: Remove outliers

print "\n*******************************************************************"
print "\t\t Outlier Candidates\n"

# Investigate outliers by looking at top 1% of each feature
# Dismiss 'poi' and loop starting from the second feature
for i in range(1,len(all_features)):
    feature_percentile = np.percentile(data[:,i],99)
    # Print name of the feature and its 99% percentile
    print all_features[i], ": ", feature_percentile, "\n"
    
    for name in my_dataset:
        # For each 'name' in 'my_dataset', if i'th feature of 'name' is not NaN
        # and if it is in the top 1 percentile, print that 'name'
        if my_dataset[name][all_features[i]]!='NaN':
            if my_dataset[name][all_features[i]]>feature_percentile:
                print name
                print my_dataset[name], "\n"

# Investigate features with many missing values
# Dictionary keys with many NaN values will be kept in this array
NaNs = []

# efp_all stores the treshold value, 85% of 'all_features' 
# i.e. if there are 20 features, efp_all = 17, which is equal to 85% of 20
efp_all = len(all_features)/(20/17.0)

for name in my_dataset:
    # If the most common value of 'name' is NaN, store element for printing
    # p.s. Counter method returns list of tuples,
    # (i,j), where i->Most common values in decresing order and j->i's frequency
    if Counter(my_dataset[name].values()).most_common()[0][0] == 'NaN':
        nan_freq = Counter(my_dataset[name].values()).most_common()[0][1]
        # If more than 85% of the features are having 'NaN' value
        if nan_freq > efp_all:
            NaNs.append((name, nan_freq))
            
print 'Names with many NaNs \n'
for nan in NaNs:
    print nan[0], " has ", nan[1], " NaN's"
    print my_dataset[nan[0]]
    
print "\n*******************************************************************\n"

# After visually investigating the output, there seems to be three outliers.
# Some can dispute 'WHALEY DAVID A', 'WROBEL BRUCE' and 'GRAMM WENDY L'
# are also outliers, where they both have 18 NaN values, but with our limited 
# dataset these data points can be left as valid data points.
Outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']

# Remove outliers from the dataset. 
# First two of them are spreadsheet quirks
# and the third one is full of NaN values
for outlier in Outliers:
    my_dataset.pop(outlier, 0)


### Task 3: Create new feature(s)

# Create new features using 
# 'from_this_person_to_poi' / 'to_messages'; 
# 'from_poi_to_this_person' / 'from_messages'
# These features gives some numbers that doesn't makes any sense
# but when we find a ratio, it will give us a percent 
# and let us understand at what percent
# a person gets/sends emails from or to POI's

def computeFraction( poi_messages, all_messages ):
    """ 
    Given a number messages to/from POI (numerator) 
    and number of all messages to/from a person (denominator),
    return the fraction of messages to/from that person
    that are from/to a POI
    """
   
    fraction = 0.    
    if poi_messages!='NaN' and all_messages!='NaN' and all_messages!=0:
        fraction = poi_messages/float(all_messages)

    return fraction

for name in my_dataset:
    from_poi_to_this_person = my_dataset[name]['from_poi_to_this_person']
    to_messages = my_dataset[name]['to_messages']
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    my_dataset[name]['fraction_from_poi'] = fraction_from_poi 
    
    from_this_person_to_poi = my_dataset[name]['from_this_person_to_poi']
    from_messages = my_dataset[name]['from_messages']
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    my_dataset[name]['fraction_to_poi'] = fraction_to_poi
    
    
# Add features to 'features_list'

pop_items = ['from_this_person_to_poi', 'to_messages',
             'from_poi_to_this_person', 'from_messages']
for item in pop_items:
    all_features.pop(all_features.index(item))

all_features += ['fraction_from_poi'] + ['fraction_to_poi']
    

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, all_features, sort_keys = True)

# Get correlation coefficients between features
coefs = np.corrcoef(data, rowvar=0)

print "Correlations between features..."
print "Mean Correlation: ", np.mean(np.absolute(coefs))
print "Median Correlation: ", np.median(np.absolute(coefs)), "\n"

labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier


def optimizer(clf, params, features, labels):
    """
    This function gets classifier and its params
    also gets features and labels, 
    then tries to optimize the parameters 
    to find the best estimator having best f1 score
    """
    # In this function feature selection can be done in 3 ways and
    # this loop finds the best f1 score across these 3 methods
    best_score = 0.0
    best_estimator = []
    for method in ['SelectKBest', 'PCA', 'PCA,SelectKBest']:
        
        # Grid Search CV parameters will be stored in this dict
        param_grid = {}
    
        # Pipeline parameters will be stored in this list
        pipe_list = []        
        
        # Not equal feature ranges, some features having [0,1] values 
        # where some features are in millions therefore scaling is needed
        pipe_list.append(('scaler', MinMaxScaler(copy=False)))
        
        # Use only SelectKBest method for feature selection
        if method == "SelectKBest":
            pipe_list.append(('selection', SelectKBest()))            
            param_grid['selection__k'] = [5, 6, 7, 8]

        # Use only PCA method for feature selection   
        elif method == "PCA":
            pipe_list.append(('pca', PCA()))            
            param_grid['pca__n_components'] = [2, 3, 4]
            param_grid['pca__whiten'] = [True, False]
        
        # Use both PCA and SeelectKBest for feature selection
        elif method == "PCA,SelectKBest":
            combined_features = FeatureUnion([('pca', PCA()), ('selection', SelectKBest())])
            pipe_list.append(('combined_features', combined_features))            
            param_grid['combined_features__selection__k'] = [1,2]
            param_grid['combined_features__pca__n_components'] = [2,3]
            param_grid['combined_features__pca__whiten'] = [True, False]
        
        pipe_list.append(('clf', clf))
        pipeline = Pipeline(pipe_list)
        
        param_grid.update(params)
        cv = StratifiedShuffleSplit(labels, 100, random_state = 42)
    
        grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring="f1", verbose=5)
        grid_search.fit(features, labels)
        
                
        
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_estimator = [grid_search.best_estimator_, grid_search.best_score_]
            
    return best_estimator

# Give classifier name and its parameters to 'optimizer' method to find the best estimator
# p.s. use 'clf__<parameter_name>' syntax
 
# Uncomment this part to calculate calssifier results again
# p.s. It takes approximately 2 hours to run all 5 classifiers
"""
t0 = time()

results = []

#Naive bayes
results.append(optimizer(GaussianNB(),
                         {},
                         features, labels))

# Logistic Regression
results.append(optimizer(LogisticRegression(),
                         {'clf__random_state' : [42],
                          'clf__C' : [1000, 10000, 100000],
                          'clf__tol' : [1e-5, 1e-7, 1e-9]},
                         features, labels))
                         
# Random Forest
results.append(optimizer(RandomForestClassifier(),
                         {'clf__criterion' : ['gini', 'entropy'],
                          'clf__min_samples_split' : [2, 5, 8],
                          'clf__max_features' : ['sqrt', 'log2']},
                         features, labels))



# K Nearest Neighbors
results.append(optimizer(KNeighborsClassifier(),
                         {'clf__n_neighbors' : [2, 3, 5],
                          'clf__weights' : ['distance']},
                         features, labels))
           
# Adaboost
results.append(optimizer(AdaBoostClassifier(),
                         {'clf__n_estimators' : [40, 60],
                          'clf__learning_rate' : [0.4, 0.6]},
                         features, labels))
                  
with open("optimizer_results.pkl", "w") as results_file:
    pickle.dump(results, results_file)

print "Grid Search CV time:", round((time()-t0) / 60.0, 3), "mins"

"""
nb = 0 
with open("optimizer_results.pkl", "r") as results_file:
    results = pickle.load(results_file)
    for result in results:
        pprint(result)
        # For naive bayes, print the selected features
        if nb == 0:
            support = result[0].named_steps['selection'].get_support()
            scores = result[0].named_steps['selection'].scores_
            print "Selected Features:"
            for i in range(1,17):
                if support[i]:
                    print all_features[i+1], scores[i]
                result[0].named_steps['selection']
            nb+=1

                    
clf = Pipeline([('scaler', MinMaxScaler(copy=False)),
                ('selection', SelectKBest(k=5)),
                ('clf', GaussianNB())])
                
# Pipeline does the whole job step by step by fitting and transforming each step
# no need to send the selected features
features_list = all_features


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)