import numpy as np
from sklearn.cluster import KMeans
import yfinance as yf
from collections import defaultdict
from hmmlearn import hmm
import matplotlib.pyplot as plt

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
        mean, variance, weight = np.mean(cluster_data, axis=0), np.var(cluster_data, axis=0),len(cluster_data)/len(data)
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
            count[str(state0)+ "," +str(state1)] += 1
            state0 = state1

    T = []
    for i in range(clusters):
        total = 0
        for j in range(clusters):
            total += count[str([i])+","+str([j])]

        T.append([count[str([i])+","+str([j])]/total for j in range(clusters)])

    T_labels = range(clusters)
    return T, T_labels


def emission(data):
    # Initialize and fit HMM based on labels from kmeans
    model = hmm.MultinomialHMM(n_components=2, n_iter=100)
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(data)
    labels = kmeans.labels_
    model.fit([labels])
    emission_matrix = model.emissionprob_

    return emission_matrix

def predict(T, means, weights, train, test, clusters):
    """Predict the percentage change"""

    # classify
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(train)
    labels = kmeans.labels_

    # create list of predictions
    predictions = []
    actual = []
    next_label = labels[len(labels)-1]
    for i in range(len(test)):

        probs = T[next_label]
        idx = np.argmax(probs)

        #prediction = probs[idx] * means[idx][0]

        prediction = 0
        for j in range(len(probs)):
            prediction += probs[j] * means[j][0] * weights[j]

        predictions.append(prediction)
        actual.append(test[i][0])
        next_label = kmeans.predict([test[i]])[0]

    return predictions, actual





# constants
hidden_states = 3
clusters = 5
dimension_obs = 3

# data
data = load_data("AAPL", "120mo")

# this should be changed so that we are using roughly the prior 10 days of information
train_ind = int(len(data) * .7)
observed = data[:train_ind]

# find relevant % changes
obs_vector = [((closes - opens) / opens, (highs - opens) / opens, (opens - lows) / opens)
              for closes, opens, highs, lows in observed]

test = data[train_ind:]
test_vector = [((closes - opens) / opens, (highs - opens) / opens, (opens - lows) / opens)
              for closes, opens, highs, lows in test]

# find initial emission probs using k-means
means, variances, weights, labeled_data = k_means(obs_vector, clusters)

# Transition matrices with labels
T, states = transition(obs_vector, clusters)

# find steady state vector
steady = np.linalg.matrix_power(T, 100)[0]

# given a current state s we look at means variances and weights of where it could go to form a prediction

# we need to improve our prediction algorithm
predictions, actual = predict(T, means, weights, obs_vector, test_vector, clusters)


for i in range(len(predictions)):
    print(f"Actual: {actual[i]}, Predicted: {predictions[i]}")

correct = [1 if (actual[i] > 0 and predictions[i] > 0) or (actual[i] < 0 and predictions[i] < 0 )else 0 for i in range(len(actual))]

plt.bar([0, 1], height=[len(correct)-sum(correct), sum(correct)])
plt.show()

plt.bar([0, 1], height=[len(test)-sum(test), sum(test)])
plt.show()

plt.plot(actual)
plt.plot(predictions)
plt.show()


