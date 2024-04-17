""" Jeremiah Payeur and Madelyn Redick - ForecastMark
        A hidden markov model stock predictor"""

import redis
import yfinance as yf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from hmmlearn import hmm
import itertools
import time
from collections import Counter


def vector_stock(data):
    """data should be a list of tuples that contain daily close, open, highs, and lows over time"""
    return [((closes - opens) / opens, (highs - opens) / opens, (opens - lows) / opens)
            for closes, opens, highs, lows in data]


class StockRedis:

    def __init__(self, host="localhost", port=6379, decode=True):
        """initialize database"""
        self.db = redis.Redis(host=host, port=port, decode_responses=decode)

    def close(self):
        """close connection"""
        self.db.close()

    def clear_db(self):
        """empty database"""
        self.db.flushall()

    def find_tickers(self):
        return list(self.db.smembers("tickers_in_db"))

    def add_ticker(self, ticker, per_change=20, per_high=5, per_low=5, min_change=-.1, max_change=.1, latency=10,
                   train_per=.97, period="120mo", components=4, timing=False):
        """ create a hidden markov model and predictions for a specific ticker

        ticker (str) - ticker symbol
        per_change (int) - number of % change states from open to close
        per_low (int) - number of low % change states
        per_high (int) - number of high % change states

        latency (int) - number of days to look back for Maximum a Posteriori and Balm Welch Algorithms (probability of
        seeing specific states in specific order and what is maximium likely next state)

        train_per (float) - decimal between 0 and 1 that represents the portion of data to initially train

        period (str) - amount of historical data to look at

        components (int) - number of hidden states to train Markov model default 4 for (open, close, high, low)

        Note - period must be long enough to allow for latency period to take place within test set

        Adds a ticker and stores data within redis
        """

        # collect historical data for n months
        data = yf.Ticker(ticker.upper())
        data = data.history(period=f'{period}')

        # set all related data related to percentage changes
        self.db.set(f"{ticker}:steps_change", per_change)
        self.db.set(f"{ticker}:steps_high", per_high)
        self.db.set(f"{ticker}:steps_low", per_low)
        self.db.set(f"{ticker}:min_change", min_change)
        self.db.set(f"{ticker}:max_change", max_change)

        # add the ticker to set of all
        self.db.sadd(f"tickers_in_db", ticker)

        # add latency and components to database
        self.db.set(f"{ticker}:latency", latency)
        self.db.set(f"{ticker}:components", components)

        # create range of values for MAP for ticker
        self.compute_all_possible_outcomes(ticker)

        # store observation vector
        [self.db.rpush(f"{ticker}:obs_vector", ",".join([str(ob) for ob in obs])) for obs in
         list(zip(list(data["Close"]), list(data["Open"]), list(data["High"]), list(data["Low"])))[:int(len(data) *
                                                                                                        train_per)]]

        # store test vector
        [self.db.rpush(f"{ticker}:test_vector", ",".join([str(ob) for ob in obs])) for obs in
         list(zip(list(data["Close"]), list(data["Open"]), list(data["High"]), list(data["Low"])))[int(len(data) *
                                                                                                       train_per):]]

        # train model and predict outcomes
        start = 0
        if timing:
            start = time.time()
        self.predict_close_prices_for_days(ticker)
        if timing:
            print(f"It took {round(time.time() - start, 2)} seconds to load in predicted prices for {ticker}")

    def create_model(self, ticker):
        """Create the hidden markov model using a gaussian distribution with the number of components as hidden states
        """
        model = hmm.GaussianHMM(int(self.db.get(f"{ticker}:components")))

        # fit the model to the relevant percent changes
        model = model.fit([((closes - opens) / opens, (highs - opens) / opens, (opens - lows) / opens)
                           for (closes, opens, highs, lows) in
                           [(float(y) for y in x.split(",")) for x in self.db.lrange(f"{ticker}:obs_vector", 0, -1)]])
        return model

    def predict_close_prices_for_days(self, ticker):
        """Predict closes for test data takes a ticker and predicts close for each day in test set"""

        # initialize gaussian model and number of days to predict for
        model = self.create_model(ticker)
        days = len(self.db.lrange(f"{ticker}:test_vector", 0, -1))

        # iterate though days
        for day in range(days):
            # print(day, days) can leave this turned on to visualize how far along prediction progress
            self.db.rpush(f"{ticker}:predicted_close", self.get_most_probable_outcome(model, day, ticker))

    def get_most_probable_outcome(self, hmm, day, ticker):
        """Finds most probable outcome from discrete set of outcomes
        Performs MAP for a ticker on all test data and find discrete outcome that is most probable
        returns predicted close based on MAP percentage change and known open price
        """
        test = {"close": [], "open": [], "high": [], "low": []}

        # turn our test data into a df
        for (c, o, h, l) in [(float(y) for y in x.split(",")) for x in self.db.lrange(f"{ticker}:test_vector", 0, -1)]:
            test['close'].append(c)
            test['open'].append(o)
            test['high'].append(h)
            test['low'].append(l)

        test = pd.DataFrame(test)

        # find where to start for the test data, start on day one if no test data has been added to outcomes yet

        # get data from 10 days ago
        previous_data_start_index = max(0, day - int(self.db.get(f"{ticker}:latency")))

        # get data from today
        previous_data_end_index = max(0, day - 1)

        # take the section of latency from data from test dataframe
        previous_data = test.iloc[previous_data_start_index: previous_data_end_index + 1]

        previous_data_features = [((closes - opens) / opens, (highs - opens) / opens, (opens - lows) / opens)
                                  for closes, opens, highs, lows in
                                  list(zip(list(previous_data["close"]), list(previous_data["open"]),
                                           list(previous_data["high"]), list(previous_data["low"])))]

        outcome_score = []

        # use MAP algorithm to iterate through possible outcomes and choose the most likey
        for possible_outcome in [tuple([float(y) for y in x.split(",")]) for x in
                                 self.db.lrange(f"{ticker}:possible_outcomes", 0, -1)]:
            # add the test to total data
            total_data = np.row_stack((previous_data_features, possible_outcome))

            # score the outcome and add to outcome scores then find the maximum likelihood outcome
            outcome_score.append(hmm.score(total_data))
        per_change, high_change, low_change = [(float(y) for y in x.split(",")) for x in
                                               self.db.lrange(f"{ticker}:possible_outcomes", 0,
                                                              -1)][np.argmax(outcome_score)]

        # return predicted stock price
        # print(per_change) this is useful to see if we should add more or less clusters
        # it seems like if it is grouping in the same spot repeatedly we should likely take away clusters
        # clusters can be taken away by changing the the amount of steps in linspace
        return test.iloc[day]['open'] * (1 + per_change)

    def compute_all_possible_outcomes(self, ticker):
        """compute all possible outcomes of three states given total staps, and find the array of possible outcomes

        total number of spaces will be steps_change * steps_high * steps_low -- wth all valid combinations"""
        change_range = np.linspace(float(self.db.get(f"{ticker}:min_change")),
                                   float(self.db.get(f"{ticker}:max_change")),
                                   int(self.db.get(f"{ticker}:steps_change")))
        high_range = np.linspace(0, float(self.db.get(f"{ticker}:max_change")),
                                 int(self.db.get(f"{ticker}:steps_high")))
        low_range = np.linspace(0, float(self.db.get(f"{ticker}:max_change")),
                                int(self.db.get(f"{ticker}:steps_low")))

        # find all possible states
        possible_outcomes = np.array(list(itertools.product(change_range, high_range, low_range)))

        # add all possible states to redis of related ticker
        [self.db.rpush(f"{ticker}:possible_outcomes", ",".join([str(out) for out in outcome])) for outcome in
         possible_outcomes]

    def retrieve_tickers(self):
        """Returns to user list of tickers in database"""
        return self.db.lrange("tickers_in_db", 0, -1)

    def transition(self, ticker, rounded=2):
        """Find the estimation of a transition matrix using daily open and closes as well as variances

        rounded is the number of decimals to round percentage change to """
        state0s = []
        state1s = []
        state1 = 0
        # find data and iterate through percent change movements
        data = self.db.lrange(f"{ticker}:obs_vector", 0, -1)
        for i in range(len(data)):

            # set initial state
            if i == 0:

                state0 = 0

            else:
                state0 = state1

            # add data to dict
            day_data = data[i].split(",")
            state1 = round((float(day_data[0]) - float(day_data[1])) / float(day_data[1]), rounded)
            state0s.append(state0)
            state1s.append(state1)

        # make sure that it is not an absorbing matrix
        state0s.append(state1)
        state1s.append(0)

        # sort data
        states = {"state0": [state for state in state0s], "state1": [state for state in state1s]}

        # add to redis to allow future access
        for key, lst in states.items():
            for value in lst:
                self.db.rpush(f"{ticker},state:{key}", value)

        return

    def per_change_transmat(self, ticker):
        """heatmap of transition matrix -- find the relative percentage change and states and plot a transition
        matrix as a heatmap"""

        # retrieve state lists
        state0 = [float(x) for x in self.db.lrange(f"{ticker},state:state0", 0, -1)]

        state1 = [float(x) for x in self.db.lrange(f"{ticker},state:state1", 0, -1)]

        states = (list(set(state0)))
        states.sort()
        dct = {}

        # create transition matrix
        for state in states:
            transitions =[]

            # find all states state0 goes to and count
            for i in range(len(state0)):
                if state == state0[i]:
                    transitions.append(state1[i])
            count = Counter(transitions)
            total = sum(list(count.values()))

            # find the probability of going from state0-state1
            dct[state] = [count[s]/total if s in count.keys() else 0 for s in states]

        # turn dict to transition data frame
        T = pd.DataFrame(dct)
        T.set_index(T.columns, inplace=True)
        plt.figure(figsize=(10,8))
        sns.heatmap(T.transpose())

        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.title("Transition Matrix Heatmap")
        plt.xlabel("Percentage Change State")
        plt.ylabel("Percentage Change State")

        plt.show()

    def predictions_over_time(self, ticker):
        """predictions versus actual over time"""
        actual = [[float(y) for y in x.split(",")] for x in self.db.lrange(f"{ticker}:test_vector", 0, -1)]
        actual_close = [price[0] for price in actual]

        plt.plot(actual_close, label="Actual Close Prices")
        plt.plot([float(price) for price in self.db.lrange(f"{ticker}:predicted_close", 0, -1)],
                 label="Predicted Close Prices")
        plt.legend()
        plt.title(f"Actual Versus Predicted Stock Prices of {ticker} Over Time")
        plt.ylabel(f"{ticker} Price")
        plt.show()

        pass

    def port_value(self, *tickers):
        """depicts a graph that shows what our algorithm would end at versus the stocks given we start by investing
        in one share of each stock

        *tickers (str) - tickers in stock market
        loss_tol (float) - between 0 and 1 depicts % more we need to be projected to lose to sell"""

        # sum all of the closes -- this represents actual market
        total_close = None

        # for each ticker retrieve and add their actual close prices together
        for ticker in tickers:
            actual = [[float(y) for y in x.split(",")] for x in self.db.lrange(f"{ticker}:test_vector", 0, -1)]
            actual_close = [price[0] for price in actual]

            # create total_close if not made
            if total_close is None:
                total_close = [0] * len(actual_close)

            total_close = [total_close[i] + actual_close[i] for i in range(len(actual_close))]

        # "sell" the stock if we project to go down and buy if we believe it will go up -- only can buys as much as
        # we have in purchasing power

        expected = None
        for ticker in tickers:

            # retrieve open data this is what we make our projection on
            test = [[float(y) for y in x.split(",")] for x in self.db.lrange(f"{ticker}:test_vector", 0, -1)]
            test_opens = [price[1] for price in test]
            test_close = [price[0] for price in test]

            # retrieve predictions for stock
            predicted_closes = [float(close) for close in self.db.lrange(f"{ticker}:predicted_close", 0, -1)]

            # create variables for if we own a share and what fraction of the share we own
            owned = 1
            frac_share = 1
            purchasing_power = 0
            ticker_total = []

            for i in range(len(predicted_closes)):

                # we buy the stock for the amount of our purchasing power
                if predicted_closes[i] > test_opens[i] and owned == 0:
                    frac_share = purchasing_power / test_opens[i]
                    owned = 1

                    # add close that we own to amount
                    ticker_total.append(frac_share * test_close[i])

                # sell the stock if we think it will drop
                elif predicted_closes[i] < test_opens[i] and owned == 1:

                    purchasing_power = test_opens[i]
                    owned = 0

                    # add the total amount we have invested
                    ticker_total.append(purchasing_power)

                # dont buy if we think stock will keep declining
                elif predicted_closes[i] < test_opens[i] and owned == 0:

                    ticker_total.append(purchasing_power)

                # hold if we think the stock will increase
                elif predicted_closes[i] > test_opens[i] and owned == 1:

                    # add close that we own to amount
                    ticker_total.append(frac_share * test_close[i])

            # sum for all stocks
            if expected is None:
                expected = [0] * len(ticker_total)
            expected = [ticker_total[i] + expected[i] for i in range(len(expected))]

        # plot
        plt.plot(total_close, label="(no buy/sell)")
        plt.plot(expected, label="(buy/sell)")
        plt.legend()
        plt.title(f"Value of Portfolio over time of {', '.join(list(tickers))} using Prediction model")
        plt.ylabel(f"Total Value of Portfolio")
        plt.show()

    def cluster_transmat(self, ticker):
        """ find the transition matrix from the hmm model and plot it for a specific ticker"""
        model = hmm.GaussianHMM(int(self.db.get(f"{ticker}:components")))

        # find cluster transition matrix
        model = model.fit([((closes - opens) / opens, (highs - opens) / opens, (opens - lows) / opens)
                           for (closes, opens, highs, lows) in
                           [(float(y) for y in x.split(",")) for x in self.db.lrange(f"{ticker}:obs_vector", 0, -1)]])

        T = model.transmat_

        # plot transition matrix as a heatmap
        sns.heatmap(T)
        plt.title(f"Transition Matrix of {ticker} with {self.db.get(f'{ticker}:components')} Components")
        plt.xlabel("Cluster")
        plt.ylabel("Cluster")

        plt.show()