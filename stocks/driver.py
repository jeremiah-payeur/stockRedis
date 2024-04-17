"""Jeremiah Payeur and Madelyn Redick ForecastMark simple example"""

from stockRedis import StockRedis


def main():
    r = StockRedis()

    # clear database and add new data
    r.clear_db()
    r.add_ticker("META", period="10y", timing=True, train_per=.7, components=5)
    r.add_ticker("AAPL", period="10y", train_per=.7, components=6)
    r.add_ticker("MSFT", period="10y", train_per=.7, components=5)
    r.add_ticker("SPY", period="10y", train_per=.7, components=2)
    r.add_ticker("KO", period="10y", train_per=.7, components=3)
    r.add_ticker("CMG", period="10y", train_per=.7, components=8)
    r.add_ticker("AMZN", period="10y", components=8, train_per=.7)
    r.add_ticker("NKE", period="10y", components=6, train_per=.7)
    r.add_ticker("TSLA", period="10y", components=7, train_per=.7)

    # find cluster for spy and over time and transition matrix for meta and plot portfolio
    r.cluster_transmat("SPY")
    r.predictions_over_time("META")
    r.transition("META")
    r.per_change_transmat("META")
    r.port_value("META", "AAPL", "MSFT", "SPY", "KO", "CMG", "AMZN", "NKE", "TSLA")
    for ticker in r.find_tickers():
        r.port_value(ticker)

    r.close()


main()
