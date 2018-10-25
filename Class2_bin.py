from __future__ import division
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
import pandas as pd

# Lab 4
# ----Classification - More Models and Ideas

# Load Data and Set Up Training and Testing Data
# Read the breast cancer dataset from SciKit Learn datasets
'''
cancer = load_breast_cancer()
print("Cancer:")
print(cancer)
'''

# Read the Police Binned dataset into a Pandas data frame
cover = pd.read_csv('W:/Documents/SCHOOL/Towson/2018-2022 -- DSc - Computer Security/6_Fall 2018/COSC 757 - Data Mining/Assignments/Classification Competition - 11-1/train_data.csv')
# Use the hold out method to create training data (70% random sample) and testing data (30% random sample)
train=cover.sample(frac=0.7,random_state=1234)
test=cover.drop(train.index)

obs_bin = ['ID',
#'Elevation',
#'Aspect',
#'Slope',
#'Horizontal_Distance_To_Hydrology',
#'Vertical_Distance_To_Hydrology',
#'Horizontal_Distance_To_Roadways',
#'Hillshade_9am','Hillshade_noon',
#'Hillshade_3pm',
#'Horizontal_Distance_To_Fire_Points',
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

'''
# Set X = the attributes and y = the target variable
X = cancer['data']
print("X:")
print(X)

y = cancer['target']
print("y:")
print(y)
'''

X = cover.as_matrix(obs_bin)
y = cover.as_matrix(cls).ravel()

'''
X_train = train.as_matrix(obs_bin)
y_train = train.as_matrix(cls).ravel()
X_test = test.as_matrix(obs_bin)
y_test = test.as_matrix(cls).ravel()
'''

# Use train_test_split to split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y)


# Rescale the data to values between 1 and 0 (this gives each attribute equal weight)
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# ---- Logistic Regression
print("---- Logistic Regression ----")

# Set up the logistic regression classifier with 
logreg = linear_model.LogisticRegression(C=1e5)

# Fit the logistic regression model to the training data and use the resulting classifier to predict the class values for the test dataset
logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)
print(logreg)

# Calculate the accuracy of the classifier.
print("LogReg Accuracy:")
print((sum(y_test==logreg_pred))/len(logreg_pred))
# Create a confusion matrix using Scikit-Learn confusion_matrix
logreg_tab = confusion_matrix(y_test, logreg_pred)
print(logreg_tab)
# Create a classification report for the result including precision, recall, and f measure.
print(metrics.classification_report(y_test, logreg_pred))


# ---- Neural Networks
print("---- Neural Networks ----")

# Create a multilayer perceptron classifier and fit it to the training dataset.  The classifier will use the 
mlp = MLPClassifier(solver='sgd', hidden_layer_sizes=(30,30,30))
mlp.fit(X_train,y_train)

# Use the neural network to predict the test set and calculate the accuracy.
mlp_pred = mlp.predict(X_test)

print("MLP1 Accuracy:")
print((sum(y_test==mlp_pred))/len(mlp_pred))
# Create a confusion matrix using Scikit-Learn confusion_matrix
mlp_tab = confusion_matrix(y_test, mlp_pred)
print(mlp_tab)
# Create a classification report for the result including precision, recall, and f measure.
print(metrics.classification_report(y_test, mlp_pred))


# Now try a different solver.  Did you get different results?
mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(30,30,30))
mlp.fit(X_train,y_train)

# Use the neural network to predict the test set and calculate the accuracy.
mlp_pred = mlp.predict(X_test)

print("MLP2 Accuracy:")
print((sum(y_test==mlp_pred))/len(mlp_pred))
# Create a confusion matrix using Scikit-Learn confusion_matrix
mlp_tab = confusion_matrix(y_test, mlp_pred)
print(mlp_tab)
# Create a classification report for the result including precision, recall, and f measure.
print(metrics.classification_report(y_test, mlp_pred))


# ---- Support Vector Machines
print("---- SVM ----")

# Set up a SVM classifier using the radial basis function kernel
svm_clf = svm.SVC(kernel="rbf")
svm_clf.fit(X_train,y_train)

# Use the neural network to predict the test set and calculate the accuracy.
svm_pred = svm_clf.predict(X_test)

print("SVM Accuracy:")
print((sum(y_test==svm_pred))/len(svm_pred))
# Create a confusion matrix using Scikit-Learn confusion_matrix
svm_tab = confusion_matrix(y_test, svm_pred)
print(svm_tab)
# Create a classification report for the result including precision, recall, and f measure.
print(metrics.classification_report(y_test, mlp_pred))



# ---- Ensemble Methods
print("---- Ensemble Learning ----")

# Bagging
bagging = BaggingClassifier(GaussianNB(), max_samples=0.5, max_features=0.5)
bagging.fit(X_train,y_train)
bag_pred = bagging.predict(X_test)

print("Bagging Accuracy:")
print((sum(y_test==bag_pred))/len(bag_pred))
bag_tab = confusion_matrix(y_test,bag_pred)
print(bag_tab)
print(metrics.classification_report(y_test, bag_pred))

# Use cross validation to find the overall accuracy of the bagged classifier
scores = cross_val_score(bagging, X, y)
print(scores.mean())


# Boosting
boosting = AdaBoostClassifier(n_estimators=100)
boosting.fit(X_train, y_train)
boost_pred = boosting.predict(X_test)

print("Boosting Accuracy:")
print((sum(y_test==boost_pred))/len(boost_pred))
boost_tab = confusion_matrix(y_test, boost_pred)                     
print(boost_tab)
print(metrics.classification_report(y_test, boost_pred))

# Use cross validation to find the overall accuracy of the boosted classifier
scores = cross_val_score(boosting, X, y)
print(scores.mean())


# ---- Cross Validation Sampling
print("---- Cross Validation ----")

# K-fold Cross Validation Sample
svm_clf = svm.SVC(kernel="rbf")
kf = KFold(n_splits=10)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    svm_clf.fit(X_train,y_train)
    svm_pred = svm_clf.predict(X_test)
    print((sum(y_test==svm_pred))/len(svm_pred))

