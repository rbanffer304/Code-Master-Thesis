# Import libraries
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score

def train_fitting(model, X_train, X_test, y_train, y_test, model_name):
    """Train and fit a prediction model"""

    # Fit model on train data
    model.fit(X_train, y_train)

    if model_name == 'logistic_regression':
        # Get model coefficients
        model_coefficients = model.coef_[0]
        print(model.coef_)

        number_model_features = len(set(model_coefficients))

        print('Number of model features:', number_model_features)

    # Get train probabilities
    y_prob_train = model.predict_proba(X_train)

    y_pred_train = model.predict(X_train)

    # Get ths train auc score
    train_auc = roc_auc_score(y_train, y_prob_train[:, 1])

    # Get test probabilities
    y_prob_test = model.predict_proba(X_test)

    y_pred_test = model.predict(X_test)

    # Get ths test auc score
    test_auc = roc_auc_score(y_test, y_prob_test[:, 1])

    # Calculate precision-recall curve
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_prob_test[:, 1])

    # Calculate AUPRC
    auprc_test_score = auc(recall_test, precision_test)

    zeros_list_train = [0] * len(y_pred_train)
    zeros_list_test = [0] * len(y_pred_test)

    print(accuracy_score(y_train, zeros_list_train))
    print(accuracy_score(y_test, zeros_list_test))


    print(roc_auc_score(y_test, zeros_list_test))

    print(accuracy_score(y_train, y_pred_train))
    print(accuracy_score(y_test, y_pred_test))

    return train_auc, test_auc, auprc_test_score
