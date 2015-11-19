#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import logging
LEVEL = logging.WARN
LOGFILE = "poi_id.log"
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(LEVEL)
handler = logging.FileHandler(LOGFILE)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

from collections import OrderedDict as OD
import copy
import numpy as np
import matplotlib.pyplot as plt

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


def classify(features, labels, estimator, i=0, scale_features=False, n_iter=100):
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
        features = scale.fit_transform(features)
    cm = np.zeros(4).reshape(2,2) # initialize a coufusion matrix
    folds = StratifiedShuffleSplit(labels, n_iter=n_iter, test_size=0.1, random_state=42)
    for train, test in folds:
        features_train, labels_train =  features[train], labels[train]
        features_test, labels_test =  features[test], labels[test]
        estimator.fit(features_train, labels_train)
        labels_pred = estimator.predict(features_test)
        cm += confusion_matrix(labels_test, labels_pred)
    return get_scores(cm, i)


def prepare_graph(graph_name, graph_data, ranking=None, prefix=""):
    """Prepare a graph of scores by k, and save it

    :param graph_name: graph title and .png file name
    :param graph_data: data to plot, containing k and scores
    :param ranking: order of features corresponding to the data, default=None (SeleckKBest order)
    :param prefix: prefix for the .png file name, default = ""
    """
    fig = plt.figure(figsize=(10,8))
    fig_ax = fig.add_subplot(1,1,1)
    fig_ax.set_title(graph_name)
    fig_ax.set_xlim(0, 18)
    fig_ax.set_ylim(0.0, 1.0)
    fig_ax.set_xlabel("k (number of features)")
    fig_ax.set_ylabel("Score")
    fig_ax.plot(graph_data["k"], graph_data["precision"], "r-")
    fig_ax.plot(graph_data["k"], graph_data["recall"], "b-")
    fig_ax.plot(graph_data["k"], graph_data["f1"], "g-")
    for x in range(30, 65, 5):
        fig_ax.plot((0, 18), (.01*x, .01*x), "k:")
    fig_ax.legend(["precision", "recall", "f1"])
    if ranking is not None:
        fig_ax.annotate(ranking[:6], xy=(0.5,0.15), xytext=(0.5, 0.15), size=10)
        fig_ax.annotate(ranking[6:12], xy=(0.5,0.10), xytext=(0.5,0.10), size=10)
        fig_ax.annotate(ranking[12:], xy=(0.5,0.05), xytext=(0.5,0.05), size=10)
    fig.savefig(prefix + graph_name.replace(" ",  "-"))


def forward_stepwise_selection(dataset, available_features, clf_name, clf, scaled=False, scoring="accuracy"):
    """Select the features that contribute more to the ranking, log them and produce a graph

    :param available_features: list of features to consider
    :param clf: classifier or pipeline
    :param clf_name: classifier or pipeline name
    :param scaled: boofeatures scaled or not
    :param scoring: "accuracy", "precision", "recall" or "f1"
    :returns: ranking, a list of score dicts, ordered by scoring 
    """
    ranking = []
    logger.info("\nForward Stepwise Selection with All Features, scoring by " + scoring + ", estimator: " + clf_name + ", scaled=" + str(scaled))
    logger.info("{:>5}{:>15}{:>15}{:>15}{:>15}".format(*"k accuracy precision recall f1".split()))
    graph_data = {"k":[], "accuracy": [], "precision": [], "recall": [], "f1": []}
    graph_name = "Forward Stepwise, " + clf_name + ", scoring: " + scoring + ", scaled: " + str(scaled) 
    while True:
        highest_scored = (0, "", None)
        for f in available_features:
            features_list = ranking + [f]
            features, labels = extract_features_and_labels(dataset, features_list)
            try:
                scores = classify(features, labels, clf, i=1, scale_features=scaled, n_iter=1000)
                if scores[scoring] > highest_scored[0]:
                    highest_scored = (scores[scoring], f, scores)
            except ValueError as err:
                logger.warn("ranking: " + repr(ranking) + \
                        " feature: " + f + "\n" + repr(err))
                continue
        feature_selected = highest_scored[1]
        ranking.append(feature_selected)
        idx = available_features.index(feature_selected)
        del(available_features[idx])
        logger.info("{:>5}{accuracy:>15.5f}{precision:>15.5f}{recall:>15.5f}{f1:>15.5f}  {ranking}".format(len(ranking), ranking=ranking, **scores ))
        graph_data["k"].append(len(ranking))
        for s in "accuracy precision recall f1".split():
            graph_data[s].append(scores[s])
        if len(available_features) == 0:
            break
    prepare_graph(graph_name, graph_data, ranking, prefix="fig-1-")
    return ranking


def select_k_best(X, y, X_names, desc=""):
    """Use SelectKBest to get features ordered by score, log them, ang produce a graph

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


def grid_best_estimator(X, y, estimator, estimator_name, param_grid, cv=None, scoring="accuracy"):
    """Perform a grid search, log results

    :param X: features
    :param y: labels
    :param estimator: classifier or pipeline
    :param estimator_name: classifier or pipeline name
    :param para_grid: dict with parameters for grid search
    :param scoring: which score to rank by
    :returns: None
    """
    grid_search = GridSearchCV(estimator, param_grid=param_grid,
                    verbose=0, n_jobs=2, cv=cv, scoring=scoring)
    grid_search.fit(X, y)
    logger.info("\nGrid search of best estimator using: " + \
                estimator_name + ", scoring: " + scoring)
    logger.info(grid_search.best_estimator_)
    logger.info(grid_search.best_params_)
    logger.info(grid_search.best_score_)
    logger.info(grid_search.scorer_)


def scores_by_k(X, y, clf_name, clf):
    logger.info("\nGet scores by k-best for: " + clf_name)
    logger.info("{:>5}{:>15}{:>15}{:>15}{:>15}".format(*"k accuracy precision recall f1".split()))
    graph_name = "Feature Selection: SelectKBest, Scores: " + clf_name
    graph_data = {"k":[], "accuracy": [], "precision": [], "recall": [], "f1": []}
    for k in range(1,17):
        pipeline = Pipeline([("sel", SelectKBest(k=k)), ("clf", clf)])
        scores = classify(X, y, pipeline, i=1)
        logger.info("{:>5}{accuracy:>15.5f}{precision:>15.5f}{recall:>15.5f}{f1:>15.5f}".format(k, **scores))
        graph_data["k"].append(k)
        for s in "accuracy precision recall f1".split():
            graph_data[s].append(scores[s])
    if max(graph_data["precision"]) >= .3 and max(graph_data["recall"]) >= .3:
        prepare_graph(graph_name, graph_data, prefix="fig-2-", ranking="total_stock_value exercised_stock_options bonus salary ratio_from deferred_income long_term_incentive total_payments ratio_shared restricted_stock expenses other ratio_to director_fees restricted_stock_deferred deferral_payments".split())


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

    LOG = ["univariate", "scores", "scores_by_k", "stepwise"]
    LOG = []

    if LEVEL == logging.INFO and "univariate" in LOG:
        # Get features ordered by score using SelectKBest
        ## from the original features
        X, y = extract_features_and_labels(my_dataset, ["poi"]+original_features)
        select_k_best(X, y, original_features, desc="Original Features")
        # from the original + new email fatures
        X, y = extract_features_and_labels(my_dataset, ["poi"]+all_features)
        select_k_best(X, y, all_features, desc="Original + New Features")
        
    if LEVEL == logging.INFO and "scores" in LOG:
        # Try different classifiers and gridsearch
        X, y = extract_features_and_labels(my_dataset, ["poi"]+all_features)
        scaler = MinMaxScaler()
        Xs = scaler.fit_transform(X)
        cv = StratifiedShuffleSplit(y, n_iter=100, test_size=0.1)
        scoring = "recall"
        for scaled, name, clf, param_grid in [
                (0, "Naive Bayes", Pipeline([("sel", SelectKBest()), ("bay", GaussianNB())]), {"sel__k" : range(2,16)}),
                (1, "Naive Bayes scaled", Pipeline([("sel", SelectKBest()), ("bay", GaussianNB())]), {"sel__k" : range(2,16)}),
                (0, "KNN", Pipeline([("sel", SelectKBest()), ("knn", KNeighborsClassifier())]), {"sel__k": range(2,16), "knn__n_neighbors": range(1,10)}),
                (1, "KNN scaled", Pipeline([("sel", SelectKBest()), ("knn", KNeighborsClassifier())]), {"sel__k": range(2,16), "knn__n_neighbors": range(1,10)}),
                (1, "SVM scaled", Pipeline([("sel", SelectKBest()), ("svm", SVC())]), {"sel__k": range(2,16), "svm__C": [10, 20, 25, 30, 50], "svm__kernel": ["rbf", "poly", "linear"]}),
                (0, "DecisionTree", Pipeline([("sel", SelectKBest()), ("tree", DecisionTreeClassifier())]), {"sel__k": range(2,16),  "tree__min_samples_split": [1,5,10,15]}),
                (1, "DecisionTree", Pipeline([("sel", SelectKBest()), ("tree", DecisionTreeClassifier())]), {"sel__k": range(2,16),  "tree__min_samples_split": [1,5,10,15]}),
                (0, "AdaBoost", Pipeline([("sel", SelectKBest()), ("ada", AdaBoostClassifier())]), {"sel__k": range(2,16),  "ada__base_estimator__min_samples_split": [1,5,10,15]}),
                (1, "AdaBoost scaled", Pipeline([("sel", SelectKBest()), ("ada", AdaBoostClassifier())]), {"sel__k": range(2,16),  "ada__base_estimator__min_samples_split": [1,5,10,15]}),
                (0, "Random Forest", Pipeline([("sel", SelectKBest()), ("rfor", RandomForestClassifier())]), {"sel__k": range(2,16), "rfor__min_samples_split": [1,2,15]}),
                (1, "Random Forest scaled", Pipeline([("sel", SelectKBest()), ("rfor", RandomForestClassifier())]), {"sel__k": range(2,16), "rfor__min_samples_split": [1,2,15]}),
            ]:
            grid_best_estimator(Xs if scaled else X, y, clf, name, param_grid, cv, scoring)

    if LEVEL == logging.INFO and "scores_by_k" in LOG:
        X, y = extract_features_and_labels(my_dataset, ["poi"]+all_features)
        scaler = MinMaxScaler()
        Xs = scaler.fit_transform(X)
        for scaled, name, clf in [
                (0, "Naive Bayes", GaussianNB()),
                (1, "Naive Bayes scaled", GaussianNB()),
                (0, "KNN(1)", KNeighborsClassifier(n_neighbors=1)),
                (1, "KNN(1) scaled", KNeighborsClassifier(n_neighbors=1)),
                (0, "KNN(2)", KNeighborsClassifier(n_neighbors=2)),
                (1, "KNN(2) scaled", KNeighborsClassifier(n_neighbors=2)),
                (1, "SVC (C=10,  kernel=linear)", SVC(C=10,  kernel="linear")),
                (1, "SVC (C=10,  kernel=rbf)", SVC(C=10,  kernel="rbf")),
                (1, "SVC (C=10,  kernel=poly 2)", SVC(C=10,  kernel="poly", degree=2)),
                (0, "Decision Tree (1)", DecisionTreeClassifier()),
                (0, "Decision Tree (2)", DecisionTreeClassifier(min_samples_split=5)),
                (0, "Decision Tree (5)", DecisionTreeClassifier(min_samples_split=10)),
                (0, "Decision Tree (10)", DecisionTreeClassifier(min_samples_split=15)),
                (0, "AdaBoost()", AdaBoostClassifier()),
                (0, "AdaBoost(5)", AdaBoostClassifier(base_estimator=DecisionTreeClassifier(min_samples_split=5))),
                (0, "AdaBoost(10)", AdaBoostClassifier(base_estimator=DecisionTreeClassifier(min_samples_split=10))),
                (0, "AdaBoost(15)", AdaBoostClassifier(base_estimator=DecisionTreeClassifier(min_samples_split=15))),
                (0, "RandomForest()", RandomForestClassifier()),
                (0, "RandomForest(5)", RandomForestClassifier(min_samples_split=5)),
                (0, "RandomForest(10)", RandomForestClassifier(min_samples_split=10)),
                (0, "RandomForest(15)", RandomForestClassifier(min_samples_split=15)),
            ]:
            scores_by_k(Xs if scaled else X, y, name, clf)

    if LEVEL == logging.INFO and "stepwise" in LOG:
        X, y = extract_features_and_labels(my_dataset, ["poi"]+all_features)
        scaler = MinMaxScaler()
        Xs = scaler.fit_transform(X)
        for scoring in ("precision", "recall"):
            for scaled, name, clf in [
                (0, "Naive Bayes", GaussianNB()),
                (1, "Naive Bayes", GaussianNB()),
                (0, "KNN(1)", KNeighborsClassifier(n_neighbors=1)),
                (1, "KNN(1)", KNeighborsClassifier(n_neighbors=1)),
                (0, "KNN(2)", KNeighborsClassifier(n_neighbors=2)),
                (1, "KNN(2)", KNeighborsClassifier(n_neighbors=2)),
                (0, "AdaBoost()", AdaBoostClassifier()),
                (1, "AdaBoost()", AdaBoostClassifier()),
                (0, "AdaBoost(5)", AdaBoostClassifier(base_estimator=DecisionTreeClassifier(min_samples_split=5))),
                (1, "AdaBoost(5)", AdaBoostClassifier(base_estimator=DecisionTreeClassifier(min_samples_split=5))),
                (0, "AdaBoost(10)", AdaBoostClassifier(base_estimator=DecisionTreeClassifier(min_samples_split=10))),
                (1, "AdaBoost(10)", AdaBoostClassifier(base_estimator=DecisionTreeClassifier(min_samples_split=10))),
                (0, "AdaBoost(15)", AdaBoostClassifier(base_estimator=DecisionTreeClassifier(min_samples_split=15))),
                (1, "AdaBoost(15)", AdaBoostClassifier(base_estimator=DecisionTreeClassifier(min_samples_split=15))),
                ]:
                forward_stepwise_selection(my_dataset, copy.copy(all_features), name, clf, scaled, scoring)


    # Selected Features
    features_list = [   "poi",
                        "director_fees",
                        "bonus",
                        "restricted_stock_deferred",
                        "expenses",
                        "ratio_shared",
                        "total_stock_value",
                        "deferred_income",
                        "deferral_payments" ]

    features, labels = extract_features_and_labels(my_dataset, features_list)
    clf = AdaBoostClassifier()
    scores = classify(features, labels, clf, i=1, scale_features=False, n_iter=1000)
    print(scores)

    # Dump classifier, dataset, and features_list to pickle files 
    dump_classifier_and_data(clf, my_dataset, features_list)


if __name__ == "__main__":
    main()
    #from itertools import combinations
    #print len(list(combinations(range(16),8)))
