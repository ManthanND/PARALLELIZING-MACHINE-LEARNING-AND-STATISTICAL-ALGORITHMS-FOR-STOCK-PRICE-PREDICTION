# importing libraries
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import time
from sklearn.metrics import mean_squared_error

# Data preprocessing
def EDA(df):
	# setting the index as date
	df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
	df.index = df['Date']

	#creating dataframe with date and the target variable
	data = df.sort_index(ascending=True, axis=0)
	# Generate new dataframe
	new_data = pd.DataFrame()
	new_data["Date"] = data["Date"]
	new_data["Close"] = data["Close"]
	return new_data

# To find out order of AR
def plotting_pacf(df, sequence_length):
	plot_pacf(df.Close, lags = sequence_length)
	plt.show()

# Generate required dataset
def required_dataset(df,train_size):
	# After finding pacf value, only first ordered AR is considered
	df["Shifted_by_1"] = df["Close"].shift()
	df.dropna(inplace=True)
	# Define dependent and independent variable
	y = df["Close"].values
	X = df["Shifted_by_1"].values
	# Generate train and test dataset
	X_train, X_test = X[0:train_size], X[train_size:len(X)]
	y_train, y_test = y[0:train_size], y[train_size:len(X)]
	X_train = X_train.reshape(-1,1)
	X_test = X_test.reshape(-1,1)
	return X_train, y_train, X_test, y_test

# To find out coefficient and intercept of the AR model
def fit_model(X,y,train_size):
	sum_x, sum_x2, sum_y, sum_xy, n = 0.0,0.0,0.0,0.0, train_size
	for i in range(len(X)):
		sum_x += X[i][0]
		sum_x2 += X[i][0] **2
		sum_y += y[i]
		sum_xy += X[i][0]*y[i]
	# Calculate coefficient and intercept
	Coefficient = (n*sum_xy-sum_x*sum_y)/(n*sum_x2 - sum_x**2)
	Intercept = (sum_y*sum_x2 - sum_x*sum_xy)/(n*sum_x2 - sum_x**2)
	return Coefficient, Intercept

def predictions(Coefficient, Intercept, X):
	y_pred = Coefficient * X + Intercept
	return y_pred

def rmse(y_test, y_pred):
	RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
	return RMSE

def final_plot(y_test, y_pred):
	# Plot
	plt.plot(y_test, label="Actual Values")
	plt.plot(y_pred, label="Predicted Values")
	plt.legend()
	plt.show()

time_taken = []
for _ in range(5):
	# Starting timer
	start = time.time()
	df = pd.read_csv("Dataset.csv")
	new_data = EDA(df)
	# To check pacf plot
	# plotting_pacf(new_data, 30)
	# declare training data size
	train_size = int(len(new_data) * 0.80)
	X_train, y_train, X_test, y_test = required_dataset(new_data, train_size)
	Coefficient, Intercept = fit_model(X_train, y_train, train_size)
	y_pred = predictions(Coefficient, Intercept, X_test)
	RMSE = rmse(y_test, y_pred)
	# End timer
	end = time.time()
	time_taken.append(end-start)
print("The RMSE is :", RMSE)
print("Average time taken by serial code = ",np.mean(time_taken), " seconds")
# To check final plot after predictions
# final_plot(y_test,y_pred)