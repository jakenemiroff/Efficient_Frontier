
# import needed modules
import yfinance as yf
from pandas_datareader import data as pdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

yf.pdr_override()
selected = ['FB', 'AMZN', 'NFLX', 'GOOG', 'AAPL', 'MSFT', 'SHOP']

data = pdr.get_data_yahoo(selected, start="2000-01-01", end="2020-03-18")

adj_close_table = data['Adj Close']

# calculate daily and annual returns of the stocks
daily_returns = adj_close_table.pct_change()
yearly_returns = daily_returns.mean() * 252
selected_stocks = yearly_returns.index.get_level_values(0).values

# get daily and covariance of returns of the stock
daily_covariance = daily_returns.cov()
annual_covariance = daily_covariance * 252

# empty lists to store returns, volatility and weights of imiginary portfolios
portfolio_returns = []
portfolio_volatility = []
sharpe_ratio = []
stock_weights = []

# set the number of combinations for imaginary portfolios
quantity_of_stocks = len(selected)
number_of_generated_portfolios = 50000

#set random seed for reproduction's sake
np.random.seed(101)

# populate the empty lists with each portfolios returns,risk and weights
for portfolio in range(number_of_generated_portfolios):
    weights = np.random.random(quantity_of_stocks)
    weights /= np.sum(weights)
    returns = np.dot(weights, yearly_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(annual_covariance, weights)))
    rf_rate = 0.02
    sharpe = (returns - rf_rate) / volatility
    sharpe_ratio.append(sharpe)
    portfolio_returns.append(returns)
    portfolio_volatility.append(volatility)
    stock_weights.append(weights)

# a dictionary for Returns and Risk values of each portfolio
portfolio = {'Returns': portfolio_returns,
             'Volatility': portfolio_volatility,
             'Sharpe Ratio': sharpe_ratio}

# extend original dictionary to accomodate each ticker and weight in the portfolio
for counter, ticker in enumerate(selected):
    print(counter)
    portfolio[ticker + ' Weight'] = [Weight[counter] for Weight in stock_weights]

# make a nice dataframe of the extended dictionary
df = pd.DataFrame(portfolio)

# get better labels for desired arrangement of columns
columns = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock + ' Weight' for stock in selected]

# reorder dataframe columns
df = df[columns]

# find min Volatility & max sharpe values in the dataframe (df)
max_sharpe = df['Sharpe Ratio'].max()

# use the min, max values to locate and create the max sharpe portfolio
sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]

# plot frontier, max sharpe & min Volatility values with a scatterplot
df.plot.scatter(x='Volatility', y='Returns', figsize=(10, 8), grid=True)
plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='red', marker='D', s=100)
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.show()

# print the details of the max sharpe portfolio
print(sharpe_portfolio.T)

