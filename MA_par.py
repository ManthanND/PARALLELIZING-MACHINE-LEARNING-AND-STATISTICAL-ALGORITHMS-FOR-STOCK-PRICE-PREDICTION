# importing libraries
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import time
from numba import config, njit, set_num_threads, prange

# Setting the OPENMP threading layer
config.THREADING_LAYER = 'omp'
no_of_threads = 8
set_num_threads(no_of_threads)

# reading the data
df = pd.read_csv('Dataset.csv')

# setting the index as date
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

#creating dataframe with date and the target variable
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame()
new_data["Date"] = data["Date"]
new_data["Close"] = data["Close"]

# NOTE: While splitting the data into train and validation set, we cannot use random splitting since that will destroy the time component.
# So here we have set the last year’s data into validation and the 4 years’ data before that into train set.
# There are 248 readings in one year
# Training data length = 987
# Validation data length = 248

# splitting into train and validation
train = new_data[:987]
valid = new_data[987:]

# In the next step, we will create predictions for the validation set and check the RMSE using the actual values.
# making predictions
preds = np.array([0 for i in prange(248)],dtype = np.float64)
@njit(parallel=True)
def compute_pred(time_step,target,preds):
	# Keeping the main for loop in series to maintain the time time component
	# Other inner loops are parallelized
	for i in range(len(preds)):
		a = 0
		if i<time_step:
			for j in prange(len(target)-time_step+i,len(target),1):
				a += target[j]
		if i:
			for k in prange(i):
				a += preds[k]
		b = a/time_step
		preds[i] = b
	return preds

def compute_numba(time_step, df, preds):
    results = compute_pred(time_step,df["Close"].to_numpy(), preds)
    return results

# Compiling function
compute_numba(1,train,preds)

time_taken = []
for _ in range(5):
	# Starting timer
	start = time.time()
	results = compute_numba(155,train,preds)
	# print(results)
	# # checking the results (RMSE value)
	rms=np.sqrt(np.mean(np.power((np.array(valid['Close'])-results),2)))
	# Stopping timer
	end = time.time()
	time_taken.append(end-start)

print('RMSE value on validation set = ',rms)
print(f"Average time taken by parallel code with {no_of_threads} threads = {np.mean(time_taken)*1000} milliseconds")

# #plot
# valid["Predictions"] = results
# plt.plot(data['Close'])
# plt.plot(valid["Predictions"])
# plt.show()