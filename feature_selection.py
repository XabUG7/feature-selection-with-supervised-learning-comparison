# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 11:18:01 2024

@author: hp
"""

import streamlit as st
import pandas as pd


title = """Comparing Supervised Learning Methods for Feature Selection"""

headr = """This exercise compares 8 different methods for feature selection. """

brief = """
The selected 6 supervised learning methods are:
 - [Logistic Regression with L1 regularization](#logistic-regression-with-l1-regularization)
 - [Logistic Regression with L2 regularization](#logistic-regression-with-l2-regularization)
 - [Decision Tree Classifier](#decision-tree-classifier)
 - [Random Forest Classifier](#random-forest-classifier)
 - [Gradient Boosting Classifier](#gradient-boosting-classifier)
 - [Mutual Information Classification](#mutual-information-classification)
 
 Additionally, 2 other methods will be used:
 - [Features with least correlation](#features-with-least-correlation) 
 - [The most common features out of the previous methods](#the-most-common-features-out-of-the-previous-methods)
 
The dataset contains the forest cover type for 30 x 30 meter cells obtained from US Forest Service (USFS) Region 2 Resource Information System (RIS) data and was downloaded from [OpenML website](https://www.openml.org/search?type=data&status=active&qualities.NumberOfFeatures=between_10_100&qualities.NumberOfInstances=between_1000_10000&id=150&sort=runs). It contains 581,012 instances and 54 attributes.
The model to fit the selected features is the Gradient Boosting Classifier and the evaluation score will be the accuracy. The purpose of this exercise is not to get the maximum accuracy so default parameters will be used with the model and the first 100,000 instances of the dataset will be selected only.

"""

results = "Accuracy scores and the most common features selected"

imports = """

from scipy.io import arff
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt 

random_state= 42

"""


dataset_preprocess = """

data = arff.loadarff('covtype-normalized.arff')
df = pd.DataFrame(data[0])

df_work = df.copy().head(100000)


for col, dtype in df_work.dtypes.items():
    if dtype == object:
        df_work[col] = df_work[col].apply(lambda x: x.decode("utf-8")).astype("int")
        
df_work.head()

"""

train_test = """

# All features will be stored in a list
all_method_feats = []
all_scores = []

# Get features: X and get target: y
y = df_work["class"]
X = df_work.drop(["class"],  axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=df_work["class"], random_state=random_state)

# Prepare the scaler
scaler = StandardScaler().fit(X_train)

# Scale features
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Number of top features to select
n_feats = 4


def get_most_pop(n_feats, list_of_lists):

    # count the features and select the most common ones
    feat_count = defaultdict(lambda: 0)
    for item in list_of_lists: 
        feat_count[item] += 1
    sorted_feat_count = sorted(feat_count.items())
    sorted_correlations = sorted(sorted_feat_count, key=lambda x: x[1])
    feats = sorted_correlations[::-1][:n_feats]
    
    final_feats = [x[0] for x in feats]

    return final_feats

"""


code_L1 = """

# feature selection with logistic regression L1 regularization

lr1 = LogisticRegression(max_iter=10000, random_state=random_state,solver="liblinear", penalty="l1").fit(X_train_scaled, y_train)

# Getting most repeated features out of the most relevant ones
df_lr1_dict = {}
for coefs_tup in list(zip(lr1.classes_, lr1.coef_)):
    df_lr1_dict[coefs_tup[0]] = coefs_tup[1]
lr1_df = pd.DataFrame(df_lr1_dict, index=X_train.columns)


feats_lr1 = []
for i in lr1_df.columns:
    temp = lr1_df.sort_values(i, ascending=False)
    feats_lr1.append(list(temp.index[:n_feats]))

# Make a list with all the selected features
all_feats_lr1 = [x for xs in feats_lr1 for x in xs]

# Get the most popular feats from logistic regression
final_feats_lr1 = get_most_pop(n_feats, all_feats_lr1)
all_method_feats.append(final_feats_lr1)

model_lr1 = GradientBoostingClassifier(random_state=random_state).fit(X_train[final_feats_lr1], y_train)

# Score
score_1 = model_lr1.score(X_test[final_feats_lr1], y_test)
all_scores.append(score_1)
print("Accuracy score is: ", score_1)

"""

code_L2 = """

# feature selection with logistic regression L2 regularization

lr2 = LogisticRegression(max_iter=10000, random_state=random_state,solver="liblinear", penalty="l2").fit(X_train_scaled, y_train)

# Getting most repeated features out of the most relevant ones
df_lr2_dict = {}
for coefs_tup in list(zip(lr2.classes_, lr2.coef_)):
    df_lr2_dict[coefs_tup[0]] = coefs_tup[1]
lr2_df = pd.DataFrame(df_lr2_dict, index=X_train.columns)


feats_lr2 = []
for i in lr2_df.columns:
    temp = lr2_df.sort_values(i, ascending=False)
    feats_lr2.append(list(temp.index[:n_feats]))


# Make a list with all the selected features
all_feats_lr2 = [x for xs in feats_lr2 for x in xs]


# Get the most popular feats from logistic regression
final_feats_lr2 = get_most_pop(n_feats, all_feats_lr2)
all_method_feats.append(final_feats_lr2)


model_lr2 = GradientBoostingClassifier(random_state=random_state).fit(X_train[final_feats_lr2], y_train)

# Score
score_2 = model_lr2.score(X_test[final_feats_lr2], y_test)
all_scores.append("Accuracy score is: ", score_2)

"""

decision_tree = """

# Feature selection with decision tree

# Model to get the most significant features
clf =  DecisionTreeClassifier(random_state=random_state).fit(X_train, y_train)

# selecting the best 10 features out of the most important ones
feature_importance = {"names": clf.feature_names_in_, "importance": clf.feature_importances_}
feature_importance_df = pd.DataFrame(feature_importance)
feature_importance_df.sort_values("importance",ascending=False, inplace=True)

top_x_features = feature_importance_df[:n_feats]

# Best 10 features are in feat_tree
feat_tree = list(top_x_features["names"])
all_method_feats.append(feat_tree)

# Take only the most important features to tune the model
X_train1 = X_train[feat_tree]
X_test1 = X_test[feat_tree]

# Building the model
model = GradientBoostingClassifier(random_state=random_state).fit(X_train1, y_train)

# Score
score_3 = model.score(X_test1, y_test)
all_scores.append(score_3)
print("Accuracy score is: ", "Accuracy score is: ", score_3)

"""

random_forest = """

# Feature selection with random forest

# Model to get the most significant features
clf_forest =  RandomForestClassifier(random_state=random_state, n_jobs=-1).fit(X_train, y_train)

# selecting the best 10 features out of the most important ones
feature_importance_forest = {"names": clf_forest.feature_names_in_, "importance": clf_forest.feature_importances_}
feature_importance_forest_df = pd.DataFrame(feature_importance_forest)
feature_importance_forest_df.sort_values("importance",ascending=False, inplace=True)

top_x_features = feature_importance_forest_df[:n_feats]

# Best 10 features are in feat_tree
feat_tree_forest = list(top_x_features["names"])
all_method_feats.append(feat_tree_forest)


# Take only the most important features to tune the model
X_train1 = X_train[feat_tree_forest]
X_test1 = X_test[feat_tree_forest]

# Building the model
model_forest_feats = GradientBoostingClassifier(random_state=random_state).fit(X_train1, y_train)

# Score
score_4 = model_forest_feats.score(X_test1, y_test)
all_scores.append(score_4)
print("Accuracy score is: ", score_4)

"""

gradient_boosting = """

# Feature selection with gradient booster

# Model to get the most significant features
clf_booster =  GradientBoostingClassifier(random_state=random_state).fit(X_train, y_train)

# selecting the best 10 features out of the most important ones
feature_importance_booster = {"names": clf_booster.feature_names_in_, "importance": clf_booster.feature_importances_}
feature_importance_booster_df = pd.DataFrame(feature_importance_booster)
feature_importance_booster_df.sort_values("importance",ascending=False, inplace=True)

top_x_features = feature_importance_booster_df[:n_feats]

# Best 10 features are in feat_tree
feat_booster = list(top_x_features["names"])
all_method_feats.append(feat_booster)


# Take only the most important features to tune the model
X_train1 = X_train[feat_booster]
X_test1 = X_test[feat_booster]

# Building the model
model_booster = GradientBoostingClassifier(random_state=random_state).fit(X_train1, y_train)

# Score
score_5 = model_booster.score(X_test1, y_test)
all_scores.append(score_5)
print("Accuracy score is: ", score_5)
"""

less_corr = """

# Less correlated

train_corr_arr = np.abs(X_train.corr().values)
corr_sum = train_corr_arr.sum(axis=0)
corr_sum_list = list(zip(corr_sum, X_train.columns))

sorted_correlations = sorted(corr_sum_list, key=lambda x: x[0])

less_corr_feats = [feat[1] for feat in sorted_correlations[:n_feats]]
all_method_feats.append(less_corr_feats)

# Take only the most important features to tune the model
X_train1 = X_train[less_corr_feats]
X_test1 = X_test[less_corr_feats]

# Building the model
model_corr = GradientBoostingClassifier(random_state=random_state).fit(X_train1, y_train)

# Score
score_6 = model_corr.score(X_test1, y_test)
all_scores.append(score_6)
print("Accuracy score is: ", score_6)

"""


mutual_information = """

# Mutual information classification

mutual_info_importances = mutual_info_classif(X_train_scaled, y_train, random_state=random_state)
mutual_info_df = pd.DataFrame(mutual_info_importances, index=X.columns)

# Getting a list of the most significant features
mutual_feats = list(mutual_info_df.sort_values(0, ascending=False).index[:n_feats])
all_method_feats.append(mutual_feats)


# Take only the most important features to tune the model
X_train1 = X_train[mutual_feats]
X_test1 = X_test[mutual_feats]

# Building the model
model_mutual_importance = GradientBoostingClassifier(random_state=random_state).fit(X_train1, y_train)

# Score
score_7 = model_mutual_importance.score(X_test1, y_test)
all_scores.append(score_7)
print("Accuracy score is: ", score_7)

"""


top_selected = """

top_selected_feats = get_most_pop(n_feats, [x for xs in all_method_feats for x in xs])
all_method_feats.append(top_selected_feats)

# Take only the most important features to tune the model
X_train1 = X_train[top_selected_feats]
X_test1 = X_test[top_selected_feats]

# Building the model
model_mutual_importance = GradientBoostingClassifier(random_state=random_state).fit(X_train1, y_train)

# Score
score_8 = model_mutual_importance.score(X_test1, y_test)
all_scores.append(score_8)
print("Accuracy score is: ",score_8)

"""


# PANDAS IMPORT DATAFRAMES

df_head = pd.read_csv("forest_cover_dataset_head.csv")
df_results = pd.read_csv("results.csv")
result_scores = df_results.iloc[8]

 
st.set_page_config(layout="wide", page_title="Forest Cover data feature selection", page_icon=":deciduous_tree:")
st.write("#")

st.sidebar.markdown('''
# Sections
- [Overview](#overview)
- [Results](#results)
- [Imports](#imports)
- [Dataset preprocessing](#dataset-preprocessing)
- [Logistic Regression with L1 regularization](#logistic-regression-with-l1-regularization)
- [Logistic Regression with L2 regularization](#logistic-regression-with-l2-regularization)
- [Decision Tree Classifier](#decision-tree-classifier)
- [Random Forest Classifier](#random-forest-classifier)
- [Gradient Boosting Classifier](#gradient-boosting-classifier)
- [Features with least correlation](#features-with-least-correlation) 
- [Mutual Information Classification](#mutual-information-classification)
- [The most common features out of the previous methods](#the-most-common-features-out-of-the-previous-methods)
                                               
''', unsafe_allow_html=True)



st.markdown("<h1 style='text-align: center; color: black;'>Comparing Supervised Learning Methods for Feature Selection</h1>", unsafe_allow_html=True)
st.markdown("""#""")

st.subheader("Overview")
st.markdown(r"$\textsf{\large This exercise compares 8 different methods for feature selection.}$")

st.markdown(brief, unsafe_allow_html=True)

st.subheader("Results")
st.image("features_results.png")

    
st.subheader("Imports")
st.code(imports)

st.subheader("Dataset preprocessing")
st.code(dataset_preprocess)

# Display dataframe
st.dataframe(df_head)
st.code(train_test)

# Display code and accuracy results
st.subheader("Logistic Regression with L1 regularization")
st.code(code_L1)
st.markdown(f"Accuracy score is: {result_scores[0]}")

st.subheader("Logistic Regression with L2 regularization")
st.code(code_L2)
st.markdown(f"Accuracy score is: {result_scores[1]}")

st.subheader("Decision Tree Classifier")
st.code(decision_tree)
st.markdown(f"Accuracy score is: {result_scores[2]}")

st.subheader("Random Forest Classifier")
st.code(random_forest)
st.markdown(f"Accuracy score is: {result_scores[3]}")

st.subheader("Gradient Boosting Classifier")
st.code(gradient_boosting)
st.markdown(f"Accuracy score is: {result_scores[4]}")

st.subheader("Features with least correlation")
st.code(less_corr)
st.markdown(f"Accuracy score is: {result_scores[5]}")

st.subheader("Mutual Information Classification")
st.code(mutual_information)
st.markdown(f"Accuracy score is: {result_scores[6]}")

st.subheader("The most common features out of the previous methods")
st.code(top_selected)
st.markdown(f"Accuracy score is: {result_scores[7]}")
