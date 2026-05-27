"""
HurdleRegression.py

Contains the HurdleRegression class, which is a regression model that handles excessive zeros by fitting a two-part model and combining predictions:
    1) binary classifier
    2) continuous regression
Implemented as a valid sklearn estimator, so it can be used in pipelines and GridSearch objects.

Adapted from the HurdleRegression class as implemented by VIEWS here:
https://github.com/prio-data/viewsforecasting/blob/main/Tools/ViewsEstimators.py

which is in turn a derivative of code developed by Geoff Hurdock: 
https://geoffruddock.com/building-a-hurdle-regression-estimator-in-scikit-learn/
"""

from typing import Optional, Union
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from xgboost import XGBRegressor
from xgboost import XGBClassifier


class HurdleRegression(BaseEstimator):
    """ 
    Regression model which handles excessive zeros by fitting a two-part model and combining predictions:
        1) binary classifier
        2) continuous regression
    Implemented as a valid sklearn estimator, so it can be used in pipelines and GridSearch objects.

    Args:
        clf_params: dict of parameters to pass to classifier sub-model when initialized
        reg_params: dict of parameters to pass to regression sub-model when initialized
    """

    def __init__(
            self,
            clf_params: Optional[dict] = None,
            reg_params: Optional[dict] = None
        ):
        self.clf_params = clf_params
        self.reg_params = reg_params
        self.clf_fi = []
        self.reg_fi = []


    def fit(
            self,
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series]
        ) -> 'HurdleRegression':
        """
        Fit the model according to the given training data.
        The classifier is fit on the full data, while the regressor is only fit on the non-zero outcomes.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Training data
            y (Union[np.ndarray, pd.Series]): Target values

        Raises:
            ValueError: If the number of features in X is less than 2.

        Returns:
            HurdleRegression: Returns a fitted instance of self.
        """

        # check that X and y have correct shape and type
        X, y = check_X_y(
            X, y, dtype=None,
            accept_sparse=False,
            accept_large_sparse=False,
            ensure_all_finite='allow-nan' #type: ignore
        )
        if X.shape[1] < 2:
            raise ValueError('Cannot fit model when n_features = 1')

        # fit classifier
        self.clf_ = XGBClassifier(n_estimators=100,learning_rate=0.05,n_jobs=-2)
        if self.clf_params:
            self.clf_.set_params(**self.clf_params)
        self.clf_.fit(X, y > 0)
        self.clf_fi = self.clf_.feature_importances_

        # fit regressor only on non-zero outcomes
        self.reg_ = XGBRegressor(n_estimators=100,learning_rate=0.05,n_jobs=-2)
        if self.reg_params:
            self.reg_.set_params(**self.reg_params)
        self.reg_.fit(X[y > 0], y[y > 0])
        self.reg_fi = self.reg_.feature_importances_

        # return self
        self.is_fitted_ = True
        return self


    def predict(
            self, 
            X: Union[np.ndarray, pd.DataFrame]
        ) -> np.ndarray:
        """ 
        Predict combined response using probabilistic classification outcome and regression prediction.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Input data
        Returns:
            np.ndarray: Combined predictions
        """

        # check that X has correct shape and type, and that model is fitted
        X = check_array(X, accept_sparse=False, accept_large_sparse=False)
        check_is_fitted(self, 'is_fitted_')

        # combine predictions from classifier and regressor
        return self.clf_.predict_proba(X)[:, 1] * self.reg_.predict(X)