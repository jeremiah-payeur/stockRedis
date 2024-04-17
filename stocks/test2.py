import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import yfinance as yf
from collections import defaultdict
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import itertools


def load_data(ticker, period='1y'):
    data = yf.Ticker(ticker)
    data = data.history(period=period)
    return list(zip(list(data["Close"]), list(data["Open"]), list(data["High"]), list(data["Low"])))


def k_means_helper(data, clusters, labels):
    """find mean variance and weight of each relevant cluster"""
    means = []
    variances = []
    weights = []

    # attach labels to data
    labeled_data = defaultdict(list)
    for i in range(len(labels)):
        labeled_data[str(labels[i])].append(data[i])

    # find mean variances and weights for all clusters
    for cluster_label in range(clusters):
        cluster_data = labeled_data[str(cluster_label)]
        mean, variance, weight = np.mean(cluster_data, axis=0), np.var(cluster_data, axis=0), len(cluster_data) / len(
            data)
        means.append(mean)
        variances.append(variance)
        weights.append(weight)

    return means, variances, weights, labeled_data


def k_means(data, clusters):
    # initialize k-means
    kmeans = KMeans(n_clusters=clusters)

    kmeans.fit(data)
    labels = kmeans.labels_
    # find mean variance and weight of each cluster
    return k_means_helper(data, clusters, labels)


def transition(data, clusters):
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(data)

    count = defaultdict(int)
    # find probability of going from one cluster to another
    state0 = 1
    for i in range(len(data)):
        if i == 0:
            state0 = kmeans.predict([data[i]])
        else:
            state1 = kmeans.predict([data[i]])
            count[str(state0) + "," + str(state1)] += 1
            state0 = state1

    T = []
    for i in range(clusters):
        total = 0
        for j in range(clusters):
            total += count[str([i]) + "," + str([j])]

        T.append([count[str([i]) + "," + str([j])] / total for j in range(clusters)])

    T_labels = range(clusters)
    return T, T_labels


def predict(T, means, weights, train, test, clusters):
    """Predict the percentage change"""

    # classify
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(train)
    labels = kmeans.labels_

    # create list of predictions
    predictions = []
    actual = []
    next_label = labels[len(labels) - 1]
    for i in range(len(test)):

        probs = T[next_label]
        idx = np.argmax(probs)

        # prediction = probs[idx] * means[idx][0]

        prediction = 0
        for j in range(len(probs)):
            prediction += probs[j] * means[j][0] * weights[j]

        predictions.append(prediction)
        actual.append(test[i][0])
        next_label = kmeans.predict([test[i]])[0]

    return predictions, actual


def compute_all_possible_outcomes(n_steps_frac_change, n_steps_frac_high, n_steps_frac_low):
    frac_change_range = np.linspace(-0.1, 0.1, n_steps_frac_change)
    frac_high_range = np.linspace(0, 0.1, n_steps_frac_high)
    frac_low_range = np.linspace(0, 0.1, n_steps_frac_low)

    possible_outcomes = np.array(list(itertools.product(frac_change_range, frac_high_range, frac_low_range)))

    return possible_outcomes

def extract(previous_data):
    previous_data= list(zip(list(previous_data["close"]), list(previous_data["open"]), list(previous_data["high"]), list(previous_data["low"])))
    return [((closes - opens) / opens, (highs - opens) / opens, (opens - lows) / opens)
                  for closes, opens, highs, lows in previous_data]

def get_most_probable_outcome(hmm, day_index, latency, test, possible_outcomes):
    previous_data_start_index = max(0, day_index - latency)
    previous_data_end_index = max(0, day_index - 1)
    previous_data = test.iloc[previous_data_start_index: previous_data_end_index+1]
    previous_data_features = extract(previous_data)

    outcome_score = []
    for possible_outcome in possible_outcomes:
        total_data = np.row_stack((previous_data_features, possible_outcome))
        outcome_score.append(hmm.score(total_data))
    most_probable_outcome = possible_outcomes[np.argmax(outcome_score)]

    return most_probable_outcome


def predict_close_price(hmm, day_index, latency, test, possible_outcomes):
    open_price = test.iloc[day_index]['open']
    predicted_frac_change, _, _ = get_most_probable_outcome(hmm, day_index, latency, test, possible_outcomes)

    return open_price * (1 + predicted_frac_change)


def create_model(obs_vector, components=4, latency=10):
    hmm = GaussianHMM(components)
    hmm = hmm.fit(obs_vector)

    return hmm


def predict_close_prices_for_days(days, hmm, latency, test, possible_outcomes, with_plot=False):
    predicted_close_prices = []
    for day_index in range(days):
        predicted_close_prices.append(predict_close_price(hmm, day_index, latency, test, possible_outcomes))

    if with_plot:
        test_data = test[0: days]
        actual_close_prices = list(test_data['close'])
        print(actual_close_prices, predicted_close_prices)
        plt.plot(actual_close_prices)
        plt.plot(predicted_close_prices)

        plt.show()

        return predicted_close_prices


def main():
    n_steps_frac_change = 50
    n_steps_frac_high = 10
    n_steps_frac_low = 10
    latency = 10

    # constants
    hidden_states = 4

    # data
    data = load_data("AAPL", "5y")


    # this should be changed so that we are using roughly the prior 10 days of information
    train_ind = int(len(data) * .7)
    observed = data[:train_ind]

    # find relevant % changes
    obs_vector = [((closes - opens) / opens, (highs - opens) / opens, (opens - lows) / opens)
                  for closes, opens, highs, lows in observed]

    test = data[train_ind:]
    test_dct = {'close': [], 'open': [], 'high': [], 'low': []}
    for (x, y, h, l) in test:
        test_dct['close'].append(x)
        test_dct['open'].append(y)
        test_dct['high'].append(h)
        test_dct['low'].append(l)

    test_df = pd.DataFrame(test_dct)

    hmm = create_model(obs_vector)
    possible_outcomes = compute_all_possible_outcomes(n_steps_frac_change, n_steps_frac_high, n_steps_frac_low)
    predict_close_prices_for_days(90, hmm, latency, test_df, possible_outcomes, with_plot=True)


main()
