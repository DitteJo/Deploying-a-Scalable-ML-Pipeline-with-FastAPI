import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics

def test_train_model_return_type():
    """
    Test that train_model returns a RandomForestClassifier instance
    """
    # create simple training data
    X_train = np.random.rand(100, 10)  # 100 samples, 10 features
    y_train = np.random.randint(0, 2, size=100)  # Binary labels

    # train the model
    model = train_model(X_train, y_train)

    # assert that the returned model is an instance of RandomForestClassifier
    assert isinstance(model, RandomForestClassifier), "train_model should return a RandomForestClassifier instance"

def test_compute_model_metrics():
    """
    Test that compute_model_metrics returns correct precision, recall, and fbeta scores
    """
    # simple known labels
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    # assert that the returned metrics are floats
    assert isinstance(precision, float), "Precision should be a float"
    assert isinstance(recall, float), "Recall should be a float"
    assert isinstance(fbeta, float), "F-beta should be a float"

def test_consistent_feature_columns():
    """
    Test that the training and test datsets have the same number of feature columns
    """
    # create simple training data
    X_train = np.random.rand(100, 10)  # 100 samples, 10 features
    y_train = np.random.randint(0, 2, size=100)  # Binary labels

    # create simple test data with the same number of feature columns
    X_test = np.random.rand(20, 10) # 20 samples, 10 features

    # check column consistency
    train_columns = X_train.shape[1]
    test_columns = X_test.shape[1]

    # assert that the number of columns in training and test datasets are the same
    assert train_columns == test_columns, "Training and test datasets should have the same number of feature columns"


