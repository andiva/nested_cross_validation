import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import GridSearchCV

from time_delay_polynomial_regression import TimeDelayPolynomialRegression
from nested_cross_validation.loops import uniform_loop


def load_csv(filename="SWaT_P1.csv"):
    df = pd.read_csv(filename)
    times = pd.to_datetime(df['Timestamp'], format=" %d/%m/%Y %H:%M:%S AM")
    df = df[['LIT101', 'MV101', 'P101']]
    t, X = times.values, df.values.astype(float)
    mv101 = X[:, 1]
    X[mv101==0, 1] = 1
    return t, X


def plot_sensors(t, X):
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(t, X[:, 0], label="LIT101 (sensor)")
    ax1.set_ylabel("LIT101 (sensor)")

    ax2.plot(t, X[:, 1], alpha=0.5, label="MV101 (actuator)")
    ax2.plot(t, X[:, 2], alpha=0.5, label="P101 (actuator)")
    ax2.legend(loc=2)
    return ax1


def get_array_indices(num_samples, num_folds, split_ratio):
    """ Converts start-end indices to indices array """
    for trn, tst in uniform_loop(0, num_samples-1, num_folds, split_ratio):
        yield np.arange(trn[0], trn[1]+1), np.arange(tst[0], tst[1]+1)


def main():
    register_matplotlib_converters()
    times, X = load_csv()

    # todo: filter sensor LIT101, for example
    # from scipy.signal import medfilt
    # X[:,0] = medfilt(X[:,0], 101)

    ax = plot_sensors(times, X)

    # simple fitting with the first 70% of samples
    # and prediction for remaining sample
    train_size = int(0.7*X.shape[0])
    X_train, X_test = X[:train_size], X[train_size:]

    model = TimeDelayPolynomialRegression(degree=2, delay=2)
    model.fit(X_train)
    print('score:', model.score(X_test))
    Y_predict = model.predict(X_test)
    ax.plot(times[train_size:], Y_predict, 'r-', alpha=0.7,
                                                 label="Prediction ahead")

    # estimate model performance with nested cross-validation
    param_grid = {'delay': np.arange(1, 3),
                  'degree': np.arange(1, 3),
                  'penalty': [True, False],
                  'fit_intercept': [True, False],}

    scores = []
    for trn, tst in uniform_loop(0, X.shape[0]-1, num_folds=3, split_ratio=4):
        Xtrn, Xtst = X[trn[0]:trn[1]], X[tst[0]:tst[1]] # outer loop

        # run tuning on inner loop: 2 folds with split_ratio=4
        grid_search = GridSearchCV(TimeDelayPolynomialRegression(), param_grid,
                                   cv=get_array_indices(Xtrn.shape[0], 2, 1),
                                   verbose=0)
        grid_search.fit(Xtrn)
        score = grid_search.best_estimator_.score(Xtst)
        scores.append(score)

    scores = np.array(scores)
    print(f"nested CV scores: {scores}")
    print(f"mean performance: {scores.mean()}")

    # plot results
    ax.legend()
    plt.show()
    return


if __name__ == "__main__":
    main()