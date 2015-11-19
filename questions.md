---
output: html_document
---
# Enron Scandal
##By César Tablas

***

## Introduction

Enron Corporation had a rocketing success to become one of US' giants in the energy sector. At it's peak Enron was well acclaimed and attracted many investors, customers and employees, and counted with great reviews from many industry and financial analysts. However, by the end of 2001 it was revealed that Enron's success was deceitful, and willfully so: starting from their own CEO down many high ranking employees and counting with their own auditing firm's complicity, Enron had been misrepresenting their real financial condition and commiting fraudulent actions that collectively led to the largest case of bankruptcy in the US at that time.

In this project I try to identify Enron employees who may have committed fraud by applying machine learning algorithms to the Enron financial and email data made public after the legal proceedings.

***

## The Dataset

The data file contains a pickled dictionary with 146 data points mapping the name of an Enron employee to a dictionary of 21 key:value pairs that contains financial and email information. Each employee is labeled as POI / not-POI ("Person Of Interest" in the fraud case) using the key "poi" with boolean values. After the initial exploration of the data I removed the entry "TOTAL" for not being a valid data point but an aggregation. I also removed the data point for the employee "LOCKHART EUGENE E" because it only contained missing values, and also removed the variable "email-address" which is not relevant for the poi-identifier. The number of valid data points is 144, where only 18 are labeled as POI, and the other 126 are labeled as not-POI. The number of variables or features, after removing the "email-address", is 20.

The data for the employees "BELFER ROBERT" and "BHATNAGAR SANJAY" had some values entered in the wrong variable. After comparing with the original table "enron61702insiderpay.pdf" I re-entered their correct values.

Table 1 shows the number of missing values for each variable. In this context I considered appropriate to change all these missing value to zeros. The variable "loan_advances" has only 2 valid values, with only one labeled as POI: it is not useful for classification and I did not consider it in the model.

Feature                     |  valid data    | missing values  
--------------------------- | -------------: | -------------:  
bonus                       |      80        |       64  
deferral_payments           |      37        |      107  
deferred_income	            |      47        |       97  
director_fees               |      15        |      129  
exercised_stock_options	    |     100        |       44  
expenses                    |      93        |       51  
from_messages               |      84        |       60  
from_poi_to_this_person     |      84        |       60  
from_this_person_to_poi     |      84        |       60  
loan_advances               |       2        |      142
long_term_incentive         |      64        |       80
other                       |      91        |       53
poi                         |     144        |        0
restricted_stock            |     108        |       36
restricted_stock_deferred   |      16        |      128
salary                      |      93        |       51
shared_receipt_with_poi     |      84        |       60
to_messages                 |      84        |       60
total_payments              |     123        |       21
total_stock_value           |     124        |       20

#####**Table 1. Number of Valid and Missing Values for each Feature**


***

## Feature Selection

Ordering the features by their univariate F score, we get an idea of which features are more significant. The original email features are ranked in the lower half, which led me to think that it doesn't matter if somene sends/receives 100 emails or 10,000, but if someone sends/receives a high proportion of those emails to/from POI's, it can tell if they are involved. I created 3 ratios of emails sent/received/shared with POI's. These new features rank higher than the old ones. From now on, I will consider only the new email features in the model that now consists of 16 features.

rank | Feature                   |  score    |  p value   | new / old
---: | ------------------------- | --------: | ---------: | ---------
1    | total_stock_value         |  22.78    |  4.461e-06 |
2    | exercised_stock_options   |  22.61    |  4.818e-06 |
3    | bonus                     |  21.06    |  9.702e-06 |
4    | salary                    |  18.58    |  3.034e-05 |
5    | ratio_from                |  16.64    |  7.494e-05 |  <-- new
6    | deferred_income           |  11.56    |  8.743e-04 |
7    | long_term_incentive       |  10.07    |  1.845e-03 |
8    | total_payments            |   9.38    |  2.625e-03 |  
9    | ratio_shared              |   9.30    |  2.740e-03 |  <-- new
10   | restricted_stock          |   8.96    |  3.258e-03 |
--   | shared_receipt_with_poi   |   8.75    |  3.634e-03 |  old -->
11   | expenses                  |   5.55    |  1.984e-02 |
--   | from_poi_to_this_person   |   5.34    |  2.222e-02 |  old -->
12   | other                     |   4.22    |  4.179e-02 |
13   | ratio_to                  |   3.21    |  7.528e-02 |  <-- new
--   | from_this_person_to_poi   |   2.43    |  1.215e-01 |  old -->
14   | director_fees             |   2.11    |  1.483e-01 |
--   | to_messages               |   1.70    |  1.946e-01 |  old -->
15   | restricted_stock_deferred |   0.76    |  3.842e-01 |
16   | deferral_payments         |   0.22    |  6.388e-01 |
--   | from_messages             |   0.16    |  6.860e-01 |  old -->

#####**Table 2. Ranking of Features according to their Univariate F Score related to the Labels**

***

####Parameters Tuning

With a rich library like scikit-learn we have a plethora of algorithms at our disposal to analyze our data and each algorithm has many parameters that influence their outcome in the learning process, so that some parameters need to be adjusted precisely to agree with the observations. The process of Fine Tuning an algorithm consists on determining the correct set of parameters that will render an optimum outcome with our data. This process could be done manually, or it can be automated with a grid search in which every proposed parameter combination is tested and the best result is reported. The process of fine tuning an algorithm depends on understanding how each algorithm works and which parameters influence their outcome, and its importance lies in obtaining a closer match between our observations and our predictions.

I performed a grid search with various classifiers, varying some parameters in each one to tune them, and running them with both the original and scaled data. Scoring by **accuracy** is not appropriate for this dataset because the number of not-POI is not balanced with the number of POI, being easy to classify a lot of not-POI's correctly thus pumping up the accuracy. For this reason I used **precision** and **recall** as the scoring values. Scaling of the features is important for Support Vector Machines, not influencing much the other classifiers used.

Classifier    | Scaled | Scoring   | Parameters                 | Score  
:--------:    | :---:  | :-----:   | :--------:                 | :---:  
Naive Bayes   | no     | precision | k=5                        | 0.415  
""            | yes    | precision | k=12                       | 0.449  
""            | no     | recall    | k=14                       | 0.990  
""            | yes    | recall    | k=15                       | 0.920  
KNN           | no     | precision | k=6, n=1                   | 0.407  
""            | yes    | precision | k=15, n=1                  | 0.371  
""            | no     | recall    | k=6, n=1                   | 0.325  
""            | yes    | recall    | k=12, n=1                  | 0.365  
SVC           | yes    | precision | k=15, C=20, kernel=linear  | 0.430  
""            | yes    | recall    | k=15, C=20, kernel=linear  | 0.240  
Decision Tree | no     | precision | k=4, min_samples_split=5   | 0.310   
""            | yes    | precision | k=12, min_samples_split=5  | 0.269  
""            | no     | recall    | k=3, min_samples_split=1   | 0.325  
""            | yes    | recall    | k=15, min_samples_split=1  | 0.340  
AdaBoost      | no     | precision | k=11, min_samples_split=1  | 0.412  
""            | yes    | precision | min_samples_split=10, k=12 | 0.404  
""            | no     | recall    | k=15, min_samples_split=10 | 0.395  
""            | yes    | recall    | k=14, min_samples_split=1  | 0.340  
Random Forest | no     | precision | k=15, min_samples_split=15 | 0.355  
""            | yes    | precision | k=6, min_samples_split=2   | 0.352   
""            | no     | recall    | k=9, min_samples_split=15  | 0.215   
""            | yes    | recall    | k=11, min_samples_split=15 | 0.230   

#####**Table 3. Summary of Grid Search Results**

***

####Forward Stepwise Selection

To select the features and the classifier for the algorithm I performed a Forward Stepwise Selection using the clasiifiers and parameters obtained in the previous step. Tables 4 and 5 show a comparison between the scores obtained using these classifiers and two different feature selection methods. The order of features for SelectKBest is the same as shown in Table 2. 

***

Classifier | k | precision  | recall | SelectKBest Selected Features
----------- | ----: | ----: | ----: | ------------------  
GaussianNB | 10 | 0.40782 | 0.32850 | ['total_stock_value', 'exercised_stock_options', 'bonus', 'salary', 'ratio_from', 'deferred_income', 'long_term_incentive', 'total_payments', 'ratio_shared', 'restricted_stock']  
KNeighbors (n_neighbors=1) |	5 | 0.43489	| 0.34900 | ['total_stock_value', 'exercised_stock_options', 'bonus', 'salary', 'ratio_from'] 
KNeighbors (n_neighbors=1) scaled features | 13 | 0.38963 |	0.32300	| ['total_stock_value', 'exercised_stock_options', 'bonus', 'salary', 'ratio_from', 'deferred_income', 'long_term_incentive', 'total_payments', 'ratio_shared', 'restricted_stock', 'expenses', 'other', 'ratio_to'] 
AdaBoost | 14 | 0.42515 | 0.32800 | ['total_stock_value', 'exercised_stock_options', 'bonus', 'salary', 'ratio_from', 'deferred_income', 'long_term_incentive', 'total_payments', 'ratio_shared', 'restricted_stock', 'expenses', 'other', 'ratio_to', 'director_fees'] 

#####**Table 4. Precision and Recall for different Classifiers with features selected by SelectKBest Method**

***

Classifier | k | precision | recall | Forward Stepwise Selected Features 
---------- | ----: | ----: | ----: | :-----------------
GaussianNB | 9 | 0.47152 | 0.34350 | ['bonus', 'deferred_income', 'ratio_shared', 'ratio_to', 'long_term_incentive', 'expenses', 'salary', 'total_payments', 'total_stock_value']
KNeighbors (n_neighbors=1) | 13 | 0.43657 | 0.38200 |  ['director_fees', 'restricted_stock_deferred', 'bonus', 'ratio_shared', 'ratio_to', 'other', 'expenses', 'ratio_from', 'salary', 'deferred_income', 'deferral_payments', 'exercised_stock_options', 'total_stock_value']  
KNeighbors (n_neighbors=1) scaled features | 7 | 0.50277 |	0.45300 | ['director_fees', 'restricted_stock_deferred', 'bonus', 'ratio_shared', 'ratio_to', 'exercised_stock_options', 'salary']  
AdaBoost | 8 | 0.58243	| 0.40100 | ['director_fees', 'bonus', 'restricted_stock_deferred', 'expenses', 'ratio_shared', 'total_stock_value', 'deferred_income', 'deferral_payments']  

#####**Table 5. Precision and Recall for different Classifiers with features selected by Forward Stepwise Method**


***

![Forward Stepwise Selection of Features](fig-1.png)

#####**Fig 1. Scores by Number of Features using AdaBoost Classifier, Features selected by Forward Stepwise Method**

***

![Selected K Best Features](fig-2.png)

#####**Fig 2. Scores by Number of Features using AdaBoost Classifier, Features selected by SelectKBest Univariate Method**


***

## Algorithm

#####**Classifier:** AdaBoostClassifier  

#####**Parameters Used:** Default

#####**Features Selected:**  

- director_fees  
- bonus  
- restricted_stock_deferred  
- expenses  
- ratio_shared  
- total_stock_value  
- deferred_income  
- deferral_payments


***

## Validation

The whole point of building and tuning a classification model is to use it when new data comes along and determine which class it belongs to. In our case, the data is a closed chapter: Fortunately, there won't be any more Enron employees that can be POI/not-POI. However, for the sake of being thorough, we need to evaluate the performance of our model. In the case of production datasets when new data is going to appear, we need to evaluate the performace of the model with the data at hand. For this purpose it is a good practice to put aside a portion of the dataset to test the model that has been trained or developed with the rest of the data. The failure of doing so could lead to a false perception of the real performance of our model: we need to have an insight on how the model will generalize on an independent or new dataset.

In our case, the dataset contains 144 valid observations and only 18 of them are labeled POI. Due to the limitation in the number of observations and the low number of POI's, it is difficult to use a conventional validation like setting asside 30% of the data for testing, and training on the 70% left. For this reason, I used a StratifiedShuffleSplit cross-validation, in which for each iteration the dataset is partitioned into a training and testing set and the model performance is averaged from all iterations. Cross-validation is important in guarding against problems like overfitting or testing hypotheses suggested by the data.

***

## Performance

The algorithm has the following average performance:  

- **Precision = 0.62:** Of those predicted as POI, 62% are indeed POI, while the rest are not.  

- **Recall = 0.44:** Of the real POI's, only 44% are predicted correctly. 

***
