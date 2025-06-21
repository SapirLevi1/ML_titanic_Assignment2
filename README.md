# ML_titanic_Assignment2

Titanic Survival Prediction – Supervised Learning Flow (Assignment 2)

This repository contains my solution for Assignment 2 in the Machine Learning course. The task was to perform a full supervised learning flow on the Titanic dataset, using both train and test sets provided by the course staff. The main objective was to build a classification pipeline that predicts passenger survival, using proper preprocessing, feature engineering, and model selection strategies, while following academic requirements and coding standards.

The Jupyter notebook (Assignment2_supervised_learning_flow.ipynb) includes the following sections:

# Introduction

Dataset: Titanic passengers data (train and test)

Task type: Classification (binary – survived or not)

Evaluation metric: F1 Score (macro average)

Data Preparation & EDA

Loaded both train and test datasets (without re-splitting)

Displayed head of each set

Visualized distributions and relationships of features such as Age, Sex, Class and Survival using seaborn and matplotlib

Discussed data imbalance and correlations

Feature Engineering

Created new features: FamilySize, IsAlone, AgeBin, FareBin

Dropped unnecessary columns (e.g. Name, Ticket, Cabin)

Imputed missing values and applied transformations consistently to both train and test

Applied preprocessing pipeline with ColumnTransformer, StandardScaler, and OneHotEncoder

Model Training & Cross-Validation

Tested three models: Decision Tree, K-Nearest Neighbors, and Naive Bayes

Performed hyperparameter tuning with GridSearchCV using 5-fold stratified cross-validation

Chose the best model and hyperparameters based on macro-F1 score

Displayed a summary table comparing the models’ performance

Final Training & Test Evaluation

Re-trained the best model on the entire training data

Evaluated on the test set

Displayed the first 5 predictions

Reported classification metrics (precision, recall, F1) on test

Compared performance on train and test sets to evaluate overfitting

Plotted confusion matrices for both train and test sets

# Bonus

Applied advanced feature engineering (e.g. binning and group-wise feature statistics)
Included F1_macro scores from validation and testing
Used multiple models and compared their behavior

# Tools & Libraries

Python 3.x
pandas, numpy
matplotlib, seaborn
scikit-learn (GridSearchCV, pipelines, classifiers)
Jupyter Notebook (Colab-compatible)

# Contribution

This project was developed by Sapir Levi (ID suffix: 8413), as part of the HIT Machine Learning course. While I used ChatGPT as an assistant to clarify and improve my code and explanations, the entire pipeline and analysis were implemented, tested, and understood independently. All decisions regarding model design and interpretation were made based on my understanding of the course material.

# License

This project is submitted as an academic assignment. All rights reserved to the student unless otherwise specified by course instructors
