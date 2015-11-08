#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import logging
LEVEL = logging.WARN
LEVEL = logging.INFO
LOGFILE = "poi_id.log"
#open(LOGFILE, "w").close()
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(LEVEL)
handler = logging.FileHandler(LOGFILE)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

from pprint import pprint

from collections import OrderedDict as OD
import copy
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier 

from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def prepare_dataset():
    # Load the dictionary containing the dataset
    with open("final_project_dataset.pkl", "r") as f:
        dataset = pickle.load(f)
    # Remove outliers and make necessary corrections
    ## Eliminate the TOTAL line
    dataset.pop("TOTAL", None)
    ## Eliminate the line for "LOCKHART EUGENE E"
    dataset.pop("LOCKHART EUGENE E", None)
    ## Correct two data points that have data in wrong columns
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
        dataset[name][k] = float(v)
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
        dataset[name][k] = float(v)
    return dataset

def create_new_features(dataset):
    """Create new features based on existing email features
    
    the new features created are:
    - "ratio_from" : ratio of emails sent to poi / total emails sent
    - "ratio_shared" : ratio of emails shared received from poi / total emails received
    - "ratio_to" : ratio of emails received from poi / total emails received

    :param dataset: the dictionary containing the dataset
    :returns: dataset including with new features
    """
    for name in dataset:
        val = dataset[name]
        to_messages = val["to_messages"]
        from_messages = val["from_messages"]
        if to_messages != "NaN":
            dataset[name]["ratio_to"] = \
                float(val["from_poi_to_this_person"]) / to_messages
            dataset[name]["ratio_shared"] = \
                float(val["shared_receipt_with_poi"]) / to_messages
        else:
            dataset[name]["ratio_to"] = "NaN"
            dataset[name]["ratio_shared"] = "NaN"
        if from_messages != "NaN":
            dataset[name]["ratio_from"] = \
                float(val["from_this_person_to_poi"]) / from_messages
        else:
            dataset[name]["ratio_from"] = "NaN"
    return dataset
    
def get_scores(cm, i=0):
    """Get the scores for label i from a confusion matrix

    :param cm: the confusion matrix
    :param i: targeted label's index in the matrix
    :returns: dict of scores
    """
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
    scores = OD([ (val, eval(val)) for val in "accuracy precision recall f1 f2 total_predictions tp tn fp fn".split() ])
    return scores

def extract_features_and_labels(dataset, selected_features):
    """Return arrays of features and labels from dataset

    :param dataset: dictionary containing features and labels
    :param selected_features: list of features to extract from dataset
    :returns: np.arrays of features and labels, only with features selected
    """
    if selected_features[0] != "poi":
        selected_features = ["poi"] + selected_features
    # Extract features and labels from dataset for local testing
    data = featureFormat(dataset, selected_features, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    return np.array(features), np.array(labels)

def classify(features, labels, estimator, i=0, scale_features=False):
    """Use estimator and folds to train and test features/labels

    :param features: array of variables/features
    :param labels: array of labels
    :param estimator: classifier or pipeline
    :param i: targeted label's index in the matrix
    :returns: dict of scores
    """
    if scale_features is True:
        # Scale all features
        scale = MinMaxScaler()
        scale.fit_transform(features)
    cm = np.zeros(4).reshape(2,2) # initialize a coufusion matrix
    folds = StratifiedShuffleSplit(labels, n_iter=1000, test_size=0.1, random_state=42)
    for train, test in folds:
        features_train, labels_train =  features[train], labels[train]
        features_test, labels_test =  features[test], labels[test]
        estimator.fit(features_train, labels_train)
        labels_pred = estimator.predict(features_test)
        cm += confusion_matrix(labels_test, labels_pred)
    return get_scores(cm, i)

def forward_stepwise_selection(dataset, available_features, estimator, scoring="precision"):
    """Select the features that contribute more to the ranking

    :param available_features: list of features to consider
    :param estimator: classifier or pipeline
    :param scoring: "accuracy", "precision", "recall" or "f1"
    :returns: ranking, a list of score dicts, ordered by scoring 
    """
    ranking = []
    while True:
        highest_scored = (0, "")
        for f in available_features:
            features_list = ranking + [f]
            #print(features_list)
            features, labels = extract_features_and_labels(dataset, features_list)
            try:
                scores = classify(features, labels, estimator, i=1, scale_features=False)
                if scores[scoring] > highest_scored[0]:
                    highest_scored = (scores[scoring], f, scores)
            except ValueError as err:
                logger.warn("ranking: " + repr(ranking) + \
                        " feature: " + f + "\n" + repr(err))
        feature_selected = highest_scored[1]
        ranking.append(feature_selected)
        idx = available_features.index(feature_selected)
        del(available_features[idx])
        logger.info("By_" + scoring + ": " + \
                    "{:0.3f} ".format(scores[scoring]) + \
                    "Accuracy|Precision|Recall|F1|F2|".replace(
                        "|", ": {:0.3f} ").format(*(scores["accuracy"],
                        scores["precision"], scores["recall"],
                        scores["f1"], scores["f2"])) + \
                    " Features_List: " + repr(ranking) )
        if len(available_features) == 0:
            break
    return ranking

def select_k_best(X, y, X_names, desc=""):
    """Use SelectKBest to get features ordered by score

    :param X: set of features
    :param y: labels
    :param X_names: list, with the names of the features used
    :param desc: string, description of the features set used
    """
    sel = SelectKBest(k="all")
    sel.fit(X, y)
    logger.info("\nSelectKBest scores for feature set: " + desc)
    logger.info("{:30}{:>15}{:>15}".format(*"Feature F_score p_value".split()))
    logger.info("-" * 60)
    for f, s, p  in sorted(zip(X_names, sel.scores_,sel.pvalues_), key=lambda t: t[1], reverse=True):
        logger.info("{:30}{:15.2f}{:15.3e}".format(f, s, p))
    return sel

def grid_best_estimator(X, y, estimator, estimator_name, param_grid, scoring="accuracy"):
    grid_search = GridSearchCV(estimator, param_grid=param_grid,
                    verbose=0, n_jobs=4, scoring=scoring)
    grid_search.fit(X, y)
    logger.info("\nGrid search of best estimator using: " + \
                estimator_name + ", scoring: " + scoring)
    logger.info(grid_search.best_estimator_)
    logger.info(grid_search.best_params_)
    logger.info(grid_search.best_score_)
    logger.info(grid_search.scorer_)

def scores_by_k(X, y, clf_name, clf):
    pass
    logger.info("\nGet scores by k-best for: " + clf_name)
    logger.info("{:>5}{:>15}{:>15}{:>15}{:>15}".format(*"k accuracy precision recall f1".split()))
    for k in range(1,17):
        pipeline = Pipeline([("sel", SelectKBest(k=k)), ("clf", clf)])
        logger.info("{:>5}{accuracy:>15.5f}{precision:>15.5f}{recall:>15.5f}{f1:>15.5f}".format(k,**classify(X, y, pipeline, i=1)))

    
def main():
    # Load dataset, eliminate outliers, make corrections
    data_dict = prepare_dataset()
    # Add new features
    data_dict = create_new_features(data_dict)
    # Store for easy export below
    my_dataset = data_dict 
    # original features in dataset, except "poi" and "email_address"
    original_features = [
        "bonus",
        "deferral_payments",
        "deferred_income",
        "director_fees",
        "exercised_stock_options",
        "expenses",
        "from_messages",
        "from_poi_to_this_person",
        "from_this_person_to_poi",
        "long_term_incentive",
        "other",
        "restricted_stock",
        "restricted_stock_deferred",
        "salary",
        "shared_receipt_with_poi",
        "to_messages",
        "total_payments",
        "total_stock_value"]
    # and new features created in prepare_dataset()
    new_features = [
        "ratio_from",
        "ratio_shared",
        "ratio_to"]
    # include new email features (3) but excludes old email features (5)
    all_features = [
        "bonus",
        "deferral_payments",
        "deferred_income",
        "director_fees",
        "exercised_stock_options",
        "expenses",
        "long_term_incentive",
        "other",
        "ratio_from",
        "ratio_shared",
        "ratio_to",
        "restricted_stock",
        "restricted_stock_deferred",
        "salary",
        "total_payments",
        "total_stock_value"]

    LOG = ["scores", "stepwise"]
    LOG = ["scores"]
    LOG = ["stepwise"]
    if LEVEL == logging.INFO and "scores" in LOG:
        # Get features ordered by score using SelectKBest
        ## from the original features
        X, y = extract_features_and_labels(my_dataset, ["poi"]+original_features)
        select_k_best(X, y, original_features, desc="Original Features")
        # from the original + new email fatures
        X, y = extract_features_and_labels(my_dataset, ["poi"]+all_features)
        select_k_best(X, y, all_features, desc="Original + New Features")
        # Try different classifiers and gridsearch
        X, y = extract_features_and_labels(my_dataset, ["poi"]+all_features)
        scaler = MinMaxScaler()
        Xs = scaler.fit_transform(X)
        ## GaussianNB
        pipeline = Pipeline([("sel", SelectKBest(k=1)), ("bay", GaussianNB())])
        param_grid = {"sel__k" : range(2,16)}
        grid_best_estimator(X, y, pipeline, "Naive Bayes", param_grid, scoring="precision")
        grid_best_estimator(Xs, y, pipeline, "Naive Bayes, scaled features", param_grid, scoring="precision")
        grid_best_estimator(X, y, pipeline, "Naive Bayes", param_grid, scoring="f1")
        grid_best_estimator(Xs, y, pipeline, "Naive Bayes, scaled features", param_grid, scoring="f1")
        ## KNN
        pipeline = Pipeline([("sel", SelectKBest(k=1)), ("knn", KNeighborsClassifier(n_neighbors=5))])
        param_grid = {"sel__k" : range(2,16),
                      "knn__n_neighbors" : range(1,10)}
        grid_best_estimator(X, y, pipeline, "K Neighbors", param_grid, scoring="precision")
        grid_best_estimator(Xs, y, pipeline, "K Neighbors, scaled features", param_grid, scoring="precision")
        grid_best_estimator(X, y, pipeline, "K Neighbors", param_grid, scoring="f1")
        grid_best_estimator(Xs, y, pipeline, "K Neighbors, scaled features", param_grid, scoring="f1")
        ## SVC
        pipeline = Pipeline([("sel", SelectKBest()), ("svm", SVC())])
        param_grid = {"sel__k" : range(2,16),
                      "svm__C" : [10, 20, 25, 30, 50],
                      "svm__kernel": ["rbf", "poly", "sigmoid", "linear"],
                      "svm__gamma": [1., 10., 100.],
                      "svm__max_iter": [1, 5, 10]}
        grid_best_estimator(Xs, y, pipeline, "SVM, scaled features", param_grid, scoring="precision")
        grid_best_estimator(Xs, y, pipeline, "SVM, scaled features", param_grid, scoring="f1")
        ## Decision Tree
        pipeline = Pipeline([("sel", SelectKBest(k=13)), ("tree", DecisionTreeClassifier(min_samples_split=2))])
        param_grid = {"sel__k" : range(2,16),
                      "tree__min_samples_split" : range(1,30) }
        grid_best_estimator(X, y, pipeline, "Decision Tree", param_grid, scoring="precision")
        grid_best_estimator(Xs, y, pipeline, "Decision Tree, scaled features", param_grid, scoring="precision")
        grid_best_estimator(X, y, pipeline, "Decision Tree", param_grid, scoring="f1")
        grid_best_estimator(Xs, y, pipeline, "Decision Tree, scaled features", param_grid, scoring="f1")
        ## AdaBoost
        pipeline = Pipeline([("sel", SelectKBest(k=13)), ("ada", AdaBoostClassifier())])
        param_grid = {"sel__k" : range(2,16),
                      "ada__base_estimator__min_samples_split": range(1,30),
                      "ada__learning_rate": [.1, 1.,10.]}
        grid_best_estimator(X, y, pipeline, "AdaBoost", param_grid, scoring="precision")
        grid_best_estimator(Xs, y, pipeline, "AdaBoost, scaled features", param_grid, scoring="precision")
        grid_best_estimator(X, y, pipeline, "AdaBoost", param_grid, scoring="f1")
        grid_best_estimator(Xs, y, pipeline, "AdaBoost, scaled features", param_grid, scoring="f1")
        ## Random Forest
        pipeline = Pipeline([("sel", SelectKBest(k=13)), ("rfor", RandomForestClassifier())])
        param_grid = {"sel__k" : range(2,16),
                      "rfor__min_samples_split": range(1,30),}
        grid_best_estimator(X, y, pipeline, "Random Forest", param_grid, scoring="precision")
        grid_best_estimator(Xs, y, pipeline, "Random Forest, scaled features", param_grid, scoring="precision")
        grid_best_estimator(X, y, pipeline, "Random Forest", param_grid, scoring="f1")
        grid_best_estimator(Xs, y, pipeline, "Random Forest, scaled features", param_grid, scoring="f1")
    if LEVEL == logging.INFO and "stepwise" in LOG:
        X, y = extract_features_and_labels(my_dataset, ["poi"]+all_features)
        scaler = MinMaxScaler()
        Xs = scaler.fit_transform(X)

        '''
        scores_by_k(X, y, "Naive Bayes", GaussianNB())
        scores_by_k(Xs, y, "Naive Bayes, scaled", GaussianNB())

        scores_by_k(X, y, "KNN(1)", KNeighborsClassifier(n_neighbors=1))
        scores_by_k(Xs, y, "KNN(1), scaled", KNeighborsClassifier(n_neighbors=1))
        scores_by_k(X, y, "KNN(2)", KNeighborsClassifier(n_neighbors=2))
        scores_by_k(Xs, y, "KNN(2), scaled", KNeighborsClassifier(n_neighbors=2))

        scores_by_k(Xs, y, "SVC linear", SVC(C=10, kernel="linear"))
        scores_by_k(Xs, y, "SVC rbf", SVC(C=10, kernel="rbf"))
        scores_by_k(Xs, y, "SVC poly", SVC(C=10, kernel="poly", degree=2))
        
        scores_by_k(X, y, "Decision Tree (1)", DecisionTreeClassifier())
        scores_by_k(X, y, "Decision Tree (2)", DecisionTreeClassifier(min_samples_split=5))
        scores_by_k(X, y, "Decision Tree (5)", DecisionTreeClassifier(min_samples_split=10))
        scores_by_k(X, y, "Decision Tree (10)", DecisionTreeClassifier(min_samples_split=15))

        scores_by_k(X, y, "AdaBoost", AdaBoostClassifier())
        scores_by_k(X, y, "AdaBoost", AdaBoostClassifier(min_samples_split=5))
        scores_by_k(X, y, "AdaBoost", AdaBoostClassifier(min_samples_split=10))
        scores_by_k(X, y, "AdaBoost", AdaBoostClassifier(min_samples_split=15))

        scores_by_k(X, y, "RandomForest", RandomForestClassifier())
        scores_by_k(X, y, "RandomForest", RandomForestClassifier(min_samples_split=5))
        scores_by_k(X, y, "RandomForest", RandomForestClassifier(min_samples_split=10))
        scores_by_k(X, y, "RandomForest", RandomForestClassifier(min_samples_split=15))
        '''
        with open(LOGFILE, "r") as f:
            lines = f.read().strip().split("\n")

        lines_per_graph = 19
        graph_no = 0
        graphs = []
        while lines_per_graph * graph_no <= len(lines):
            graph_name = lines[lines_per_graph * graph_no].split(":")[1].strip()
            graph_data = lines[lines_per_graph * graph_no + 2 : lines_per_graph * graph_no + 19]
            graph_data = ([map(float, l.strip().split()) for l in graph_data])
            graphs.append(OD([("no", graph_no),
                              ("name", graph_name),
                              ("data", graph_data)]))
            graph_no += 1
        graphs = pd.DataFrame(graphs)
        # prepare individual graphs
        # prepare summary graph
            

    # TODO:

    # select the classifier
    # compare with the forward-stepwise that follows
    # chose from it
    # compare with scaling???
    # select the features, dump files
    #for feature_set, desc in [(original_features, "original features"), (all_features, "original plus new email features")]:
    # Select k best original_features

    """
    features, labels = extract_features_and_labels(my_dataset, ["poi"]+all_features)

    logger.info("Get all scores for various k best features")
    logger.info("  k   precision    recall        f1")
    for i in range(2,16):
        pipeline = Pipeline([("sel", SelectKBest(k=i)), ("dtree", KNeighborsClassifier())])
        logger.info("{:3}{precision:12.3f}{recall:10.3f}{f1:10.3f}".format(i,**classify(features, labels, pipeline, i=1)))

    
    logger.info("Get all scores for various k best features")
    logger.info("  k   precision    recall        f1")
    for i in range(2,16):
        pipeline = Pipeline([("sel", SelectKBest(k=i)), ("dtree", DecisionTreeClassifier())])
        logger.info("{:3}{precision:12.3f}{recall:10.3f}{f1:10.3f}".format(i,**classify(features, labels, pipeline, i=1)))


    """
    """
    clf = GaussianNB()
    
    logger.info("\nForward Stepwise Selection with Original Features, scoring by precision")
    af = copy.copy(original_features)
    forward_stepwise_selection(my_dataset, af, clf, scoring="precision")

    logger.info("\nForward Stepwise Selection with All Features, scoring by precision")
    af = copy.copy(all_features)
    forward_stepwise_selection(my_dataset, af, clf, scoring="precision")

    
    # Selected Features
    features_list = ["poi",
        "total_stock_value", "exercised_stock_options", "bonus",
        "salary", "ratio_from", "deferred_income",
        "long_term_incentive", "total_payments", "ratio_shared"]
    
    """
    """
    # Classifier
    # clf = Pipeline([("sel", SelectKBest(k=3)), ("bay", GaussianNB())])
    clf = GausianNB()    
    # Cross Validation
    s = classify(features, labels, clf, 1)

    s1 = (s["accuracy"], s["precision"], s["recall"], s["f1"], s["f2"])
    s2 = (s["total predictions"], s["tp"], s["fp"], s["fn"], s["tn"])

    print "Accuracy|Precision|Recall|F1|F2|".replace(
                "|", ": {:0.3f}\t").format(*map(float, s1))
    print "Total predictions|TP|FP|FN|TN|".replace(
                "|", ": {}\t").format(*map(int, s2))
    

    # Dump classifier, dataset, and features_list to pickle files 
    dump_classifier_and_data(clf, my_dataset, features_list)

    """


if __name__ == "__main__":
    main()
