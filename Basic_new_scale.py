from __future__ import division
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#Lab 3
# ---- Basic Classification with Scikit-Learn:
# Load Data and Set Up Training and Testing Data
#Import Pandas, Numpy, and Matplotlib Python Libraries

# Read the Police Binned dataset into a Pandas data frame
cover = pd.read_csv('C:/Users/whatch2/Desktop/ClassificationCompetition/train_data.csv')


# View the structure of the Iris data frame
print(cover)


# Use the hold out method to create training data (70% random sample) and testing data (30% random sample)
train=cover.sample(frac=0.7,random_state=1234)
test=cover.drop(train.index)



# Separate the observations from the class/target variable in both the training and testing data.  
# Use the ravel() function to flatten the 1D array for the class variable.  
# This is necessary for some of methods used to classify and assess accuracy.

#First attempt - Class Variable to train for is Signs_Of_Mental_Illness
#obs_orig = ['date','manner_of_death','armed','age','gender','race','city','state','threat_level','flee','body_camera']

obs_bin = ['ID',
'Elevation',
'Aspect',
'Slope',
'Horizontal_Distance_To_Hydrology',
'Vertical_Distance_To_Hydrology',
'Horizontal_Distance_To_Roadways',
'Hillshade_9am','Hillshade_noon',
'Hillshade_3pm',
'Horizontal_Distance_To_Fire_Points',
'Wilderness_Area_1',
'Wilderness_Area_2',
'Wilderness_Area_3',
'Wilderness_Area_4',
'2702','2703','2704','2705','2706','2717',
'3501','3502','4201','4703','4704','4744',
'4758','5101','5151','6101','6102','6731',
'7101','7102','7103','7201','7202','7700',
'7701','7702','7709','7710','7745','7746',
'7755','7756','7757','7790','8703','8707',
'8708','8771','8772','8776'
#,'Cover_Type'
]

labs = cover['Cover_Type']
labs = list(set(labs))
print("labs:")
print(labs)

#obs_all = ['ID','Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points','Wilderness_Area_1','Wilderness_Area_2','Wilderness_Area_3','Wilderness_Area_4','2702','2703','2704','2705','2706','2717','3501','3502','4201','4703','4704','4744','4758','5101','5151','6101','6102','6731','7101','7102','7103','7201','7202','7700','7701','7702','7709','7710','7745','7746','7755','7756','7757','7790','8703','8707','8708','8771','8772','8776','Cover_Type']

cls = ['Cover_Type']
trainObs = train.as_matrix(obs_bin)
trainCls = train.as_matrix(cls).ravel()
testObs = test.as_matrix(obs_bin)
testCls = test.as_matrix(cls).ravel()

scaler = MinMaxScaler()
# Fit only to the training data
scaler.fit(trainObs)
trainObs = scaler.transform(trainObs)
testObs = scaler.transform(testObs)

'''
# ----  K Nearest Neighbor Classification
print("---- KNN ----")


# Set up a K Nearest Neighbor Classifier with the number of neighbors = 3 and weights based on Euclidean distance
knn = KNeighborsClassifier(n_neighbors=3, weights='distance')

# Fit the K Nearest Neighbor classifier to the training data and use the resulting classifier to predict the class values for the test dataset
knn.fit(trainObs, trainCls)
knn_pred = knn.predict(testObs)
print(knn_pred)

# Calculate the accuracy of the classifier.
print("KNN Accuracy:")
print((sum(testCls==knn_pred))/len(knn_pred))
# Create a confusion matrix using Scikit-Learn confusion_matrix
knn_tab = confusion_matrix(testCls, knn_pred, labels=labs)
print(knn_tab)
# Create a classification report for the result including precision, recall, and f measure.
print(metrics.classification_report(testCls, knn_pred))

# Exercise 1: Now go back and experiment with different values of k.  What happened?


# ---- Decision Tree Classification
print("---- Decision Tree ----")

# Create a decision tree classifier and fit it to the training dataset.  This Scikit-Learn decision tree is based on the CART algorithm.  The default parameters use the GINI index as the metric for finding the best attribute split.
clf = tree.DecisionTreeClassifier(criterion="entropy", min_samples_leaf=5)
clf = clf.fit(trainObs, trainCls)

# Export the resulting tree in GraphVis format.  You can open the resulting file "tree.dot" in the graphviz Python library or at the graphviz website located at: http://www.webgraphviz.com/
#tree.export_graphviz(clf, out_file='W:/Documents/SCHOOL/Towson/2018-2022 -- DSc - Computer Security/6_Fall 2018/COSC 757 - Data Mining/Assignments/Assignment 2 - 10-22/state.dot', feature_names= ['manner_of_death_bin','armed_bin','armed_categories1','armed_categories2','age','gender_bin','race_bin','signs_of_mental_illness','threat_level_bin','flee_bin','body_camera'],  
#                         class_names=['True','False'])   



dt_pred = clf.predict(testObs)
print(dt_pred)

# Calculate the accuracy of the classifier.
print("DT Accuracy:")
print((sum(testCls==dt_pred))/len(dt_pred))
# Create a confusion matrix using Scikit-Learn confusion_matrix
dt_tab = confusion_matrix(testCls, dt_pred, labels=labs)
print(dt_tab)
# Create a classification report for the result including precision, recall, and f measure.
print(metrics.classification_report(testCls, dt_pred))



# Exercise 2: Use the evaluation metric code from the KNN example to assess the quality of your decision tree classifier.  Did you find any differences?

# You can also use a measure of entropy as the split criteria by including the parameter criterion="entropy".
clf = tree.ExtraTreeClassifier(criterion="entropy")
clf = clf.fit(trainObs, trainCls)
print(clf)

dt_pred = clf.predict(testObs)
print(dt_pred)

# Calculate the accuracy of the classifier.
print("DT Entropy Accuracy:")
print((sum(testCls==dt_pred))/len(dt_pred))
# Create a confusion matrix using Scikit-Learn confusion_matrix
dt_tab = confusion_matrix(testCls, dt_pred, labels=labs)
print(dt_tab)
# Create a classification report for the result including precision, recall, and f measure.
print(metrics.classification_report(testCls, dt_pred))

'''

# ---- Random Forest Classifier
print("---- Random Forest ----")


# Set up a random forest classifier with the number of estimators (trees) = 10
clf = RandomForestClassifier(n_estimators=100, criterion="entropy")
clf = clf.fit(trainObs, trainCls)
print(clf)

rf_pred = clf.predict(testObs)
print(rf_pred)

# Calculate the accuracy of the classifier.
print("RF Accuracy:")
print((sum(testCls==rf_pred))/len(rf_pred))
# Create a confusion matrix using Scikit-Learn confusion_matrix
rf_tab = confusion_matrix(testCls, rf_pred, labels=labs)
print(rf_tab)
# Create a classification report for the result including precision, recall, and f measure.
print(metrics.classification_report(testCls, rf_pred))

#try entropy later

print("---- Random Forest ----")
# Set up a random forest classifier with the number of estimators (trees) = 10
clf = RandomForestClassifier(n_estimators=250)
clf = clf.fit(trainObs, trainCls)
print(clf)

rf_pred = clf.predict(testObs)
print(rf_pred)

# Calculate the accuracy of the classifier.
print("RF Accuracy:")
print((sum(testCls==rf_pred))/len(rf_pred))
# Create a confusion matrix using Scikit-Learn confusion_matrix
rf_tab = confusion_matrix(testCls, rf_pred, labels=labs)
print(rf_tab)
# Create a classification report for the result including precision, recall, and f measure.
print(metrics.classification_report(testCls, rf_pred))

#try entropy later


# ---- Random Forest Classifier
print("---- Random Forest ----")

# Set up a random forest classifier with the number of estimators (trees) = 10
clf = RandomForestClassifier(n_estimators=500)
clf = clf.fit(trainObs, trainCls)
print(clf)

rf_pred = clf.predict(testObs)
print(rf_pred)

# Calculate the accuracy of the classifier.
print("RF Accuracy:")
print((sum(testCls==rf_pred))/len(rf_pred))
# Create a confusion matrix using Scikit-Learn confusion_matrix
rf_tab = confusion_matrix(testCls, rf_pred, labels=labs)
print(rf_tab)
# Create a classification report for the result including precision, recall, and f measure.
print(metrics.classification_report(testCls, rf_pred))

#try entropy later


# ---- Random Forest Classifier
print("---- Random Forest ----")

# Set up a random forest classifier with the number of estimators (trees) = 10
clf = RandomForestClassifier(n_estimators=1000)
clf = clf.fit(trainObs, trainCls)
print(clf)

rf_pred = clf.predict(testObs)
print(rf_pred)

# Calculate the accuracy of the classifier.
print("RF Accuracy:")
print((sum(testCls==rf_pred))/len(rf_pred))
# Create a confusion matrix using Scikit-Learn confusion_matrix
rf_tab = confusion_matrix(testCls, rf_pred, labels=labs)
print(rf_tab)
# Create a classification report for the result including precision, recall, and f measure.
print(metrics.classification_report(testCls, rf_pred))

#try entropy later


'''
# Exercise 3: Assess the quality of your random forest classifier.  What did you find?  Now change the n_estimators parameter to 100, 1000, and 10,000.  What happened?

# ---- Naive Bayes Classifier
print("---- Naive Bayes ----")

# Create a Naive Bayes classifier
gnb = MultinomialNB()
gnb = gnb.fit(trainObs, trainCls)
print(gnb)

nb_pred = gnb.predict(testObs)
print(nb_pred)

# Calculate the accuracy of the classifier.
print("NB Accuracy:")
print((sum(testCls==nb_pred))/len(nb_pred))
# Create a confusion matrix using Scikit-Learn confusion_matrix
nb_tab = confusion_matrix(testCls, nb_pred, labels=labs)
print(nb_tab)
# Create a classification report for the result including precision, recall, and f measure.
print(metrics.classification_report(testCls, nb_pred))



# Exercise 4: Assess the quality of your Naive Bayes classifer.  How does it compare?
'''