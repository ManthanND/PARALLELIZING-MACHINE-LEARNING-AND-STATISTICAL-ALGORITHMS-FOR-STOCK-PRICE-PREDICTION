# PARALLELIZING-MACHINE-LEARNING-AND-STATISTICAL-ALGORITHMS-FOR-STOCK-PRICE-PREDICTION

Course: High-Performance Scientific Computing (ME 766)
Guide: Prof. Shivasubramanian Gopalakrishnan

# Team members:
Chinmay Gandhareshwar (193109018)
Manthan Dhisale (193109014)
Supriyo Roy (193109013)
Abhishek Chikane (193109004)

# INTRODUCTION

Stock market analysis:
Stock market analysis is divided into 2 parts:
Fundamental analysis:
Fundamental analysis involves analyzing the company’s future profitability on the basis of its current business environment and financial performance.
Technical Analysis:
Technical analysis involves reading the data charts and using statistical figures to identify the trends in the stock market.

This project is mainly focused on technical analysis of the stock market where future stock prices are predicted by analyzing the past data.

Why machine learning, deep learning algorithms, and parallelization:
    As stock price data is a time-series data, where future prices of stock depend on its past prices and also on many other factors. To cater to these dependencies, we need to perform analysis using various regression models starting from averaging to linear regression and then move on to advanced techniques like Auto ARIMA and LSTM. Since, these models involve a lot of operations like calculating errors, weights, gradient descent, etc which can be parallelized to save time.

# LITERATURE REVIEW
    The stock prices depend on many complex factors but the available datasets have very few features like highest stock price in a day, lowest stock prices, opening stock price, closing stock, etc. These features are not enough to predict futures stock prices accurately. So, many times statistical models fail to give accurate results. This is why researchers started working on alternative methods for stock prediction like machine learning and deep learning. Then it was found that machine learning and deep learning algorithms work far better than traditional statistical algorithms. But the only disadvantage is that these models are very time-consuming.
    So, in this work, we have given an attempt to parallelize these machine learning and deep learning algorithms to save time and predict future stock prices faster. In this work, we have parallelized the following algorithms:
Statistical algorithms - Moving average, Auto Regression, AutoRegression Integrated Moving Average
Machine learning algorithms - Linear regression, KNN
Deep learning - Artificial Neural Networks
LIBRARIES, LANGUAGE, COMPILER

# Programming language - Python
Libraries - Numpy, Scipy, Pandas, Matplotlib, Statsmodel
Compiler - Numba
OpenMP threads
DATASET AND FEATURES
For our project, we have taken a time series stock market data of Tata Global. Time series data is a series of data points indexed in time order. Most commonly, it is a sequence taken at successive equally spaced points in time.
No of features used for the prediction is five. Features are date, open, highest, lowest and last value of stock. Date is the most important parameter used in prediction and our predicting label is the closed value of the stock market.
