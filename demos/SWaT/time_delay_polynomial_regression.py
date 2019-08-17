import numpy as np

from sklearn.linear_model.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error as mse


def rolling_window(X, delay):
        # X for 2d input array, delay for time delay
        X_rol = X[np.arange(X.shape[0] - delay + 1)[:,None] + np.arange(delay)]
        return X_rol.reshape(X_rol.shape[0], -1)


class TimeDelayPolynomialRegression(BaseEstimator):
    """An example of Time Delay Polynomial Regression

       TODO:
        test the functionality

       Args:
        degree (int): order of nonlinearities
        output_indices (list of int or int): indices of the зкувшсеув variables
        delay (int): count of previous timestamps that are used for prediction
        penalty (bool): if True then use L2 regularization
        fit_intercept (bool): if True the intercept is calculated

    """

    def __init__(self, degree=1, delay=1, output_indices=0,
                 penalty=True, fit_intercept=False):
        self.degree = degree
        self.delay = delay
        self.penalty = penalty
        self.output_indices = output_indices
        self.fit_intercept = fit_intercept
        self.model = None

    def get_model(self, penalty, fit_intercept):
        regr_args = {'fit_intercept': fit_intercept,
                     'normalize': True} #todo: move to input parameters
        if penalty:
            return Ridge(**regr_args)
        return LinearRegression(**regr_args)

    def fit(self, X, y=None):
        # todo: check parameters of model
        # train to predict one step in time X(ti) --> X(ti+1)
        Xtrain = rolling_window(X, self.delay)[:-1] # add delayed history
        Ytrain = X[-Xtrain.shape[0]:, self.output_indices]

        # calculate polynomial features and run linear regression
        self.model = make_pipeline(
                            PolynomialFeatures(self.degree, include_bias=False),
                            self.get_model(self.penalty, self.fit_intercept))

        self.model.fit(Xtrain, Ytrain)
        return self

    def predict_one_step_ahead(self, X):
        # predicts one step in time
        if self.model is None:
            raise RuntimeError("Train model first")

        Xtrain = rolling_window(X, self.delay)
        return self.model.predict(Xtrain) # polynomial linear regression

    def predict(self, X):
        # propogates prediction in time
        # prediction for the next time is used as input for the following one
        X_out = X.copy()
        X_out[self.delay:, 0] = 0 # use first samples as input
        for i in range(self.delay, X_out.shape[0]): # propogate predictions
            x = X_out[i-self.delay:i] # last predictions
            y = self.predict_one_step_ahead(x) # get one-step prediction
            X_out[i, self.output_indices] = y # store new prediction
        return X_out[:, self.output_indices]

    def score(self, X, y=None):
        # use negative MSE metric for one-step prediction
        Ypred = self.predict_one_step_ahead(X)
        Ytrue = X[-Ypred.shape[0]:, self.output_indices]
        return -mse(Ypred, Ytrue)