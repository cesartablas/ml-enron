#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

def get_scores(cm, i=0):
    '''Get the scores for label i from a confusion matrix

    :param cm: the confusion matrix
    :param i: targeted label's index in the matrix
    :returns: dict of scores
    '''
    accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
    precision = cm[i,i] / np.sum(cm[:,i])
    recall = cm[i,i] / np.sum(cm[i,:])
    f1 = 2.0 * precision * recall / (precision + recall)
    f2 = 5.0 * precision * recall / (4.0 * precision + recall)
    tp = cm[i,i]
    fp = np.sum(cm[:,i]) - cm[i,i]
    fn = np.sum(cm[i,:]) - cm[i,i]
    tn = np.sum(cm) - tp - fp - fn
    total_predictions = np.sum(cm)
    scores = {"Accuracy": accuracy, "Precision": precision,
              "Recall": recall, "F1": f1, "F2": f2,
              "Total Predictions": total_predictions,
              "TP": tp, "TN": tn, "FP": fp, "FN": fn}
    return scores

def classify(features, labels, clf, i=0):
    '''Use estimators to train and test features/labels

    :param features: array of variables/features
    :param labels: array of labels
    :param estimators: one classifier, or pipeline of classifiers
    :param i: targeted label's index in the matrix
    :returns: dict of scores
    '''
    folds = StratifiedShuffleSplit(labels, n_iter=1000, test_size=0.1, random_state=42)
    cm = np.zeros(4).reshape(2,2) # initialize a coufusion matrix
    for train, test in folds:
        features_train, labels_train =  features[train], labels[train]
        features_test, labels_test =  features[test], labels[test]
        clf.fit(features_train, labels_train)
        labels_pred = clf.predict(features_test)
        cm += confusion_matrix(labels_test, labels_pred)
    return get_scores(cm, i)


def main():
    # Load the dictionary containing the dataset
    with open("final_project_dataset.pkl", "r") as f:
        data_dict = pickle.load(f)
   
    # Remove outliers and do needed corrections
    ## Eliminate the TOTAL line
    data_dict.pop("TOTAL", None)
    ## Eliminate the line for "LOCKHART EUGENE E"
    data_dict.pop("LOCKHART EUGENE E", None)
    ## Correct two data points that have shifted columns
    name = "BELFER ROBERT"
    for col in [
            ("deferred_income", -102500),
            ("deferral_payments", 0),
            ("loan_advances", 0),
            ("other", 0),
            ("expenses", 3285),
            ("director_fees", 102500),
            ("total_payments", 3285),
            ("exercised_stock_options", 0),
            ("restricted_stock", 44093),
            ("restricted_stock_deferred", -44093),
            ("total_stock_value", 0)]:
        k, v = col
        data_dict[name][k] = float(v)
    name = "BHATNAGAR SANJAY"
    for col in [
            ("other", 0),
            ("expenses", 137864),
            ("director_fees", 0),
            ("total_payments", 137864),
            ("exercised_stock_options", 15456290),
            ("restricted_stock", 2604490),
            ("restricted_stock_deferred", -2604490),
            ("total_stock_value", 15456290)]:

        k, v = col
        data_dict[name][k] = float(v)

    # New features
    for name in data_dict:
        val = data_dict[name]
        to_messages = val["to_messages"]
        from_messages = val["from_messages"]
        if to_messages != "NaN":
            data_dict[name]["ratio_to"] = \
                    val["from_poi_to_this_person"] / to_messages
            data_dict[name]["ratio_shared"] = \
                    val["shared_receipt_with_poi"] / to_messages
        else:
            data_dict[name]["ratio_to"] = "NaN"
            data_dict[name]["ratio_shared"] = "NaN"
        if from_messages != "NaN":
            data_dict[name]["ratio_from"] = \
                    val["from_this_person_to_poi"] / from_messages
        else:
            data_dict[name]["ratio_from"] = "NaN"

    # Selected Features
    features_list = ["poi",
        "total_stock_value", "exercised_stock_options", "bonus",
        "salary", "ratio_from", "deferred_income",
        "long_term_incentive", "total_payments", "ratio_shared"]
    
    # Store `my_dataset` for easy export below
    my_dataset = data_dict

    # Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    features, labels = map(np.array, (features, labels))

    # Scale all features
    scale = MinMaxScaler()
    scale.fit_transform(features)

    # Classifier
    clf = GaussianNB()

    # Cross Validation
    s = classify(features, labels, clf, 1)

    s1 = (s["Accuracy"], s["Precision"], s["Recall"], s["F1"], s["F2"])
    s2 = (s["Total Predictions"], s["TP"], s["FP"], s["FN"], s["TN"])

    print "Accuracy|Precision|Recall|F1|F2|".replace(
                "|", ": {:0.3f}\t").format(
                *map(float, s1))
    print "Total predictions|TP|FP|FN|TN|".replace(
                "|", ": {}\t").format(
                *map(int, s2))

    # Dump classifier, dataset, and features_list to pickle files 

    dump_classifier_and_data(clf, my_dataset, features_list)

if __name__ == "__main__":
    main()
