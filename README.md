# feature-selection-with-supervised-learning



This exercise compares 8 different methods for feature selection. The selected 6 supervised learning methods are:

- Logistic Regression with L1 regularization
- Logistic Regression with L2 regularization
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- Mutual Information Classification
- Additionally, 2 other methods will be used:
  
- Features with least correlation
- The most common features out of the previous methods
  
The dataset contains the forest cover type for 30 x 30 meter cells obtained from US Forest Service (USFS) Region 2 Resource Information System (RIS) data and was downloaded from OpenML website (https://www.openml.org/search?type=data&status=active&qualities.NumberOfFeatures=between_10_100&qualities.NumberOfInstances=between_1000_10000&id=150&sort=runs). It contains 581,012 instances and 54 attributes. The model to fit the selected features is the Gradient Boosting Classifier and the evaluation score will be the accuracy. The purpose of this exercise is not to get the maximum accuracy so default parameters will be used with the model and the first 100,000 instances of the dataset will be selected only.

feature_selection.py contains the streamlit code to display the exercise.
Feature_selection_notebook.ipynb contains the full code to get all the results from the raw dataset
features_results.png contains the results graph
results.csv contains the scores and the features selected
