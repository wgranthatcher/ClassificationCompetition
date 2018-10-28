from __future__ import division
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

import numpy as np
import timeit
import pandas as pd
import datetime
import argparse
import sys
import keras.utils

from functools import partial
from nltk.tokenize.regexp import regexp_tokenize
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from keras_models import create_one_layer, create_dualInputSimple, create_dualInputLarge

from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import merge, Dense, Dropout, Input, concatenate
from keras.models import Model
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.optimizers import Nadam



def main():

    args = parse_arguments()

    final_test(args)

    return


def final_test(args):
    '''
    performs final test and validation across each train ratio on hardcoded
    hyperparemeter values established by the gridsearch
    '''
    # Read the Police Binned dataset into a Pandas data frame
    #cover = pd.read_csv('W:/Documents/SCHOOL/Towson/2018-2022 -- DSc - Computer Security/6_Fall 2018/COSC 757 - Data Mining/Assignments/Classification Competition - 11-1/train_data.csv')
    #cover = pd.read_csv('C:/Users/whatch2/Desktop/ClassificationCompetition/train_data.csv')
    cover = pd.read_csv('/home/grant309/ClassificationCompetition/train_data.csv')
    
    # Use the hold out method to create training data (70% random sample) and testing data (30% random sample)
    train=cover.sample(frac=0.7,random_state=1234)
    test=cover.drop(train.index)
    
    obs_bin = [ #'ID',
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
    
    X = cover.as_matrix(obs_bin)
    y = cover.as_matrix(cls).ravel()
    #y = cover.as_matrix(cls)

    print("y:")
    print(y)

    '''
    for i in y:
        if i==7:
            i==0

    print(y)
    '''
    y = y-1

    y = keras.utils.to_categorical(y,num_classes = 7)
    
    '''
    X_train = train.as_matrix(obs_bin)
    y_train = train.as_matrix(cls).ravel()
    X_test = test.as_matrix(obs_bin)
    y_test = test.as_matrix(cls).ravel()
    '''

    # Use train_test_split to split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7)

    '''
    # Rescale the data to values between 1 and 0 (this gives each attribute equal weight)
    scaler = StandardScaler()
    # Fit only to the training data
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    '''

    labels = y
    feat_inputs = X
    feat_width = int(len(X[0]))

    print 'feat width: ' + str(feat_width)

    input_ratios = args["input_ratio"]
    size = 32

    #models = {'oneLayer_comb':oneLayer_comb, 'oneLayer_perm':oneLayer_perm, \
    #'oneLayer_feat':oneLayer_feat, 'dual_simple':dual_simple, 'dual_large':dual_large}
    #models = ('oneLayer_comb', 'oneLayer_feat', 'oneLayer_perm', 'dual_simple', 'dual_large')

    m = "create_thousand_layer"
	
    data = []
    for r in args["train_ratio"]:
        percent=float(r)/100
        #stratified shuffle split used for cross validation
        sss = StratifiedShuffleSplit(n_splits=5, random_state=0, test_size=1-percent)
        cm = np.zeros([len(labs),len(labs)], dtype=np.int64)
        train_time = 0.0
        test_time = 0.0
        ir = 0
        for train_index, test_index in sss.split(feat_inputs, labels):
            feat_train, feat_test = feat_inputs[train_index], feat_inputs[test_index]
            labels_train, labels_test = labels[train_index], labels[test_index]

        scaler = StandardScaler()
        scaler.fit(feat_train)
        feat_train = scaler.transform(feat_train)
        feat_test = scaler.transform(feat_test)

        model = create_thousand_layer(optimizer='nadam', data_width=feat_width, neurons=32)
        batch = 16
        epoch = 32
        time0 = timeit.default_timer()
        model.fit(feat_train, labels_train, epochs=epoch, batch_size=batch)
        time1 = timeit.default_timer()
        labels_pred = model.predict(feat_test, batch_size = batch)
        time2 = timeit.default_timer()



        train_time += time1-time0
        test_time += time2-time1
        labels_pred = (labels_pred > 0.5)
    
	cm = cm + confusion_matrix(labels_test, labels_pred)
    acc = calc_accuracy(cm)
    prec = calc_precision(cm)
    rec = calc_recall(cm)
    f1 = calc_f1(prec, rec)
    avg_train_time = train_time/5
    avg_test_time = test_time/5

    data.append(dict(zip(["model_name", "neurons", "train_ratio", "input_ratio", \
    "epochs", "batch_size", "accuracy", "precision", "recall", "f1_score", \
    "avg_train_time", "avg_test_time"], \
    [m, size, r, ir, epoch, batch, acc, prec, rec, f1, avg_train_time, avg_test_time])))


    print 'saving results for model: ' + str(m)
    save_results(data, m, model, args["save"])


def save_results(data, modelName, model, save):
    d = datetime.datetime.today()
    month = str( '%02d' % d.month)
    day = str('%02d' % d.day)
    hour = str('%02d' % d.hour)
    year = str('%02d' % d.year)
    min = str('%02d' % d.minute)

    df = pd.DataFrame(data)
    try:
        path1 = '/home/grant309/ClassificationCompetition/Results' + modelName + month + day + year + '-' + hour + min + '.csv'
        file1 = open(path1, "w+")
    except:
        path1 = "gridSearch" + modelName + ".csv"
        file1 = open(path1, "w+")
    df.to_csv(file1, index=False)
    file1.close()

    if save==True:
        model.save('/home/grant309/ClassificationCompetition/Models/'+ modelName + month + day + year + '-' + hour + min + '.h5')

    return 0

def calc_accuracy(cm):
    TP = float(cm[1][1])
    TN = float(cm[0][0])
    n_samples = cm.sum()
    return (TP+TN)/n_samples

def calc_precision(cm):
    TP = float(cm[1][1])
    FP = float(cm[0][1])
    return TP/(TP+FP)

def calc_recall(cm):
    TP = float(cm[1][1])
    FN = float(cm[1][0])
    return TP/(FN+TP)

def calc_f1(precision, recall):
    return 2*((precision*recall)/(precision+recall))

#1000-500-250-100-10-1 Model  -WGH
def create_thousand_layer(data_width, neurons=25, optimizer='adam', dropout_rate=0.1, weight_constraint=0):
    #baseline Model
    model = Sequential()
    #The first param in Dense is the number of neurons in the first hidden layer
    #model.add(Dense(neurons, input_dim=22300, kernel_initializer='normal', activation='relu',kernel_constraint=maxnorm(weight_constraint) ))
    #model.add(Dense(1000, input_dim=data_width, kernel_initializer='normal', activation='relu'))
    #model.add(Dropout(dropout_rate))
    #model.add(Dense(500, input_dim=data_width, kernel_initializer='normal', activation='relu'))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(250, input_dim=data_width, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(7, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", help="Number of Epochs, can be list for grid search", type=int, nargs="*")
    parser.add_argument("-tr", "--train_ratio", nargs="*", type=int,
                        help="Set Test Ratios. Enter as a percent (20,40,60,80). Can be a list space delimited")
    parser.add_argument("-bs", "--batch_size", nargs="*", type=int,
                        help="Batch size. Can be a list space delimited")
    parser.add_argument("-n", "--neurons", nargs="*", type=int,
                        help="Number of Neurons. Can be a list space delimited")
    parser.add_argument("-o", "--optimizer", nargs="*",
                        help="Optimizers. Can be a list space delimited")
    parser.add_argument("-w", "--weight_constraint", nargs="*", type=int,
                        help="Weight Constraint. Can be a list space delimited")
    parser.add_argument("-d", "--dropout", nargs="*", type=int,
                        help="Dropout. Enter as percent (10,20,30,40...). Can be a list space delimited.")
    parser.add_argument("-model", "--model", help="Select which model to run: all \
    , one_layer, four_decr, four_same")
    parser.add_argument("-s", "--splits", help="Number of Splits for SSS", type=int)
    parser.add_argument("-ir", "--input_ratio", help="ratio of layer width between \
     features and permissions layers", type=float, nargs="*")
    parser.add_argument("--save", help="Saves all models run from final mode")

    args = parser.parse_args()

    arguments = {}

    if args.epochs:
        epochs = args.epochs
    else:
        print("Defaulting to 16 epochs")
        epochs = [16]
    arguments["epochs"] = epochs
    if args.train_ratio:
        train_ratio = args.train_ratio
    else:
        print("Defaulting to testing all ratios")
        train_ratio = [20, 40, 60, 80]
    arguments["train_ratio"] = train_ratio

    if args.batch_size:
        batch_size = args.batch_size
    else:
        print("Defaulting to Batch Size 10")
        batch_size = [10]
    arguments["batch_size"] = batch_size

    if args.neurons:
        neurons = args.neurons
    else:
        print("Defaulting to 32 Neurons")
        neurons = [32]
    arguments["neurons"] = neurons

    if args.optimizer:
        optimizer = args.optimizer
    else:
        print("Defaulting to NADAM Optimizer")
        optimizer = "Nadam"
    arguments["optimizer"] = optimizer

    if args.weight_constraint:
        weight_constraint = args.weight_constraint
    else:
        print("Defaulting to weight constraint 5")
        weight_constraint = [5]
    arguments["weight_constraint"] = weight_constraint

    if args.dropout:
        dropout = args.dropout
    else:
        print("Defaulting to dropout of 10%")
        dropout = [10]
    arguments["dropout"] = dropout

    if args.splits:
        splits = args.splits
    else:
        print("Defaulting to 5 SSS Split")
        splits = [5]
    arguments["splits"] = splits

    if args.input_ratio:
        input_ratio = args.input_ratio
    else:
        print("default to .25 input ratio")
        input_ratio = [.25]
    arguments["input_ratio"] = input_ratio

    if args.save:
        save = True
    else:
        save = False
    arguments["save"] = save

    return arguments


if __name__ == "__main__":
    main()

