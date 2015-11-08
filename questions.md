# Enron Scandal

***

## Introduction

Enron Corporation had a rocketing success to become one of US' giants in the energy sector. At it's peak Enron was well acclaimed and attracted many investors, customers and employees, and counted with great reviews from many industry and financial analysts. However, by the end of 2001 it was revealed that Enron's success was deceitful, and willfully so: starting from their own CEO down many high ranking employees and counting with their own auditing firm's complicity, Enron had been misrepresenting their real financial condition and commiting fraudulent actions that collectively led to the largest case of bankruptcy in the US at that time.

In this project I try to identify Enron employees who may have committed fraud by applying machine learning algorithms to the Enron financial and email data made public after the legal proceedings.

***

## The Dataset

The data file contains a pickled dictionary with 146 data points mapping the name of an Enron employee to a dictionary of 21 key:value pairs that contains financial and email information. Each employee is labeled as POI/non-POI ("Person Of Interest" in the fraud case) using the key "poi" with boolean values.

After the initial exploration of the data I removed the entry "TOTAL" for not being a valid data point but an aggregation. I also removed the data point for the employee "LOCKHART EUGENE E" because it only contained missing values, and also the variable "email-address" which is not relevant for the poi-identifier.

The number of valid data points is 144, where only 18 are labeled as POI, and the other 126 are labeled as not-POI. The number of variables or features, after removing the "email-address", is 20.

The table below shows the number of missing values for each variable. In this context I considered appropriate to change all these missing value to zeros. The variable "loan_advances" has only 2 valid values, with only one labeled as POI: it is not useful for classification and I did not consider it in the model.
 
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


The data for the employees "BELFER ROBERT" and "BHATNAGAR SANJAY" had some values entered in the wrong variable. After comparing with the original table "enron61702insiderpay.pdf" I re-entered their correct values.

***

## Feature Selection

Ordering the features by their univariate F score, we get an idea of which features are more significant. The original email features are ranked in the lower half, which led me to think that it doesn't matter if somene sends/receives 100 emails or 10,000, but if someone sends/receives a high proportion of those emails to/from POI's, it can tell if they are involved. I created 3 ratios of emails sent/received/shared with POI's. These new features rank higher than the old ones. For now on, I will consider only the new email features in the model that now consists of 16 features.

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

I performed a grid search with various classifiers, varying some parameters in each to tune them, and running them with both the original and scaled data. Scoring by **accuracy** is not appropriate for this dataset because the number of not-POI is not balanced with the number of POI, being easy to classify a lot of not-POI's correctly thus pumping up the accuracy. For this reason I needed to compare both **precision** and **f1** as the scoring value. Scaling is important for Support Vector Machines, and also gives a slight difference with Decission Trees and ensemble classifiers.

Classifier | Scaled features | Scoring | Parameters | Score | Warnings
:--------: | :-------------: | :-----: | :--------: | :---: | --------
GaussianNB | no | precision | k=5 | 0.44444 | NA  
"" | yes | precision | k=5 | 0.44444 |   
"" | no | f1 | k=5 | 0.39788 |   
"" | yes | f1 | k=14 | 0.41673 |   
KNeighborsClassifier | no | precision | k=10, n=3 | 0.83333 | UserWarning  
"" | yes | precision | k=3, n=4 | 1.00000 | 
"" | no | f1 | k=3, n=1 | 0.44250 | 
"" | yes | f1 | k=3, n=3 | 0.41005 | 
SVC | yes | precision | max_iter=1, k=8, C=10, kernel=linear | 0.83333 | ConvergenceWarning, UserWarning  
"" | yes | f1 | max_iter=10, k=9, C=10, kernel=poly | 0.45679 | 
DecisionTreeClassifier | no | precision | min_samples_split=23, k=4 | 0.80000 | UserWarning
"" | yes | precision | min_samples_split=21, k=4 | 0.80000 | 
"" | no | f1 | min_samples_split=8, k=3 | 0.56068 | 
"" | yes | f1 | min_samples_split=9, k=3 | 0.57778 | 
AdaBoostClassifier | no | precision | min_samples_split=1, k=6, learning_rate=0.1 | 0.83333 | RuntimeWarning, UserWarning
"" | yes | precision | min_samples_split=1, k=3, learning_rate=0.1 | 0.83333 | 
"" | no | f1 | min_samples_split=1, k=12, learning_rate=1.0 | 0.45688 | 
"" | yes | f1 | min_samples_split=7, k=12, learning_rate=1.0 | 0.45688 | 
RandomForestClassifier | no | precision | k=12, min_samples_split=25 | 1.00000 | UserWarning
"" | yes | precision | k=15, min_samples_split=18 | 1.00000 | 
"" | no | f1 | k=5, min_samples_split=8 | 0.50168 | 
"" | yes | f1 | k=5, min_samples_split=4 | 0.52222 | 


I 
Forward Stepwise Selection
loan_advances
2097151




***
## Algorithm

I decided to try the following classifiers: Naive Bayes, Support Vector Machines, and Linear Discriminant Analysis. To tune some of their parameters I used a grid search and after considering all the results, I chose Naive Bayes for being the simplest method that yields satisfactory performance.

Naive Bayes has no additional parameters to tune, but for other classifiers I used GridSearchCV that performs an exhaustive search of the best predictor by trying all combinations of the given parameters. e.g. to tune the SVM, I searched using "linear" and "rbf" kernels, and values of C of .1, 1, or 10. For this case, the best predictor was using a "linear" kernel with a value of C=10.

***
## Validation

The dataset contains 144 valid observations and only 18 of them are labeled POI. Due to the limitation in the number of observations and the low number of POI's, it is difficult to use a conventional validation like setting asside 30% of the data for testing, and training on the 70% left. For this reason, I used a StratifiedShuffleSplit cross validation, in which for each iteration the dataset is partitioned into a training and testing set and the model performance is averaged from all iterations.

## Performance

The algorithm has the following average performace:  
- Precision: 0.435: Of those predicted as POI, 43.5% are indeed POI, while the rest are not.  
- Recall:    0.345: Of the real POI's, only 34.5% are predicted correctly. 


