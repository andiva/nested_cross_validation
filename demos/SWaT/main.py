import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters


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


def main():
    register_matplotlib_converters()

    times, X = load_csv()

    # todo: filter sensor LIT101, for example
    # from scipy.signal import medfilt
    # X[:,0] = medfilt(X[:,0], 101)

    ax = plot_sensors(times, X)

    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()