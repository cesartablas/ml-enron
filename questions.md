# Enron Scandal

## Introduction

Enron Corporation had a rocketing success to become one of US' giants in the energy sector. At it's peak Enron was well acclaimed and attracted many investors, customers and employees, and counted with great reviews from many industry and financial analysts. However, by the end of 2001 it was revealed that Enron's success was deceitful, and willfully so: starting from their own CEO down many high ranking employees and counting with their own auditing firm's complicity, Enron had been misrepresenting their real financial condition and commiting fraudulent actions that collectively led to the largest case of bankruptcy in the US at that time.

In this project I try to identify Enron employees who may have committed fraud by applying machine learning algorithms to the Enron financial and email data made public after the legal proceedings.

## The Dataset

The data file contains a pickled dictionary with 146 data points mapping the name of an Enron employee to a dictionary of 21 key:value pairs that contains financial and email information. Each employee is labeled as POI/non-POI ("Person Of Interest" in the fraud case) using the key "poi" with boolean values.

After the initial exploration of the data I removed the entry "TOTAL" for not being a valid data point but an aggregation. I also noticed that 2 data points had some mistakes in their financial values. I compared with the original [table](enron61702insiderpay.pdf) and corrected their values. Not every employee has every piece of information. For instance, only 86 employees have quantities of sent or received emails, and only 4 employees have data for "loan_advances". In this context I deemed appropriate to change all these missing value to zeros. My complete exploration is shown in this [notebook](exploration.html).

## Feature Selection

A first read to the financial features gives an indication of which variables are associated with privielges, in this case as reward for involvment in the fraud. On the other hand, none of the original email features seemed relevant for classification. I thought that it doesn't matter if somene sends/receives 100 emails or 10,000, but if someone sends/receives a high proportion of those emails to/from POI's, it can tell if they are involved. I created 3 ratios of emails sent/received/shared with POI's.

Due to the big differences between the values of ratios (0.0 - 1.0) and some financial features with values of up to 1e+08, I scaled all the features. However, after choosing the features and classifier, I tested without scaling and the results were not affected. Because simpler is better and scaling is not called for in this algorithm, I didn't scale the features in the final poi identifier.

I used SelectKBest and selected these 9 features with the highest F score for the analysis:

features                | score | p value
------------------------|-------|--------
total_stock_value       |  22.8 |  <.001
exercised_stock_options |  22.6 |  <.001
bonus                   |  21.1 |  <.001
salary                  |  18.6 |  <.001
ratio_from              |  16.6 |  <.001
deferred_income         |  11.6 |  <.001
long_term_incentive     |  10.1 |  .0018
total_payments          |   9.4 |  .0026
ratio_shared            |   9.3 |  .0027

## Algorithm

I decided to try the following classifiers: Naive Bayes, Support Vector Machines, and Linear Discriminant Analysis. To tune some of their parameters I used a grid search and after considering all the results, I chose Naive Bayes for being the simplest method that yields satisfactory performance.

Naive Bayes has no additional parameters to tune, but for other classifiers I used GridSearchCV that performs an exhaustive search of the best predictor by trying all combinations of the given parameters. e.g. to tune the SVM, I searched using "linear" and "rbf" kernels, and values of C of .1, 1, or 10. For this case, the best predictor was using a "linear" kernel with a value of C=10.

## Validation

The dataset contains 144 valid observations and only 18 of them are labeled POI. Due to the limitation in the number of observations and the low number of POI's, it is difficult to use a conventional validation like setting asside 30% of the data for testing, and training on the 70% left. For this reason, I used a StratifiedShuffleSplit cross validation, in which for each iteration the dataset is partitioned into a training and testing set and the model performance is averaged from all iterations.

## Performance

The algorithm has the following average performace:
- Accuracy:  0.853: The label (POI or NOT-POI) is predicted correctly 85.3% of the time.
- Precision: 0.435: Of those predicted as POI, 43.5% are indeed POI, while the rest are not.
- Recall:    0.345: Of the real POI's, only 34.5% are predicted correctly.
