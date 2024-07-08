from pandas import DataFrame
from pandas import concat
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten



def time_series_lagger(data, n_in=1, n_out=1, dropnan=True):
	"""
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		pandas df: dimensionality LEN_DATA x (N_IN + N_OUT) initial:(90x(5+1))
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	
	# input sequence (t-n, ... t-1, t0)
	for i in range(n_in-1, -1, -1):
		cols.append(df.shift(i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
		
	# forecast sequence (t, t+1, ... t+n)
	for i in range(1, n_out+1):
		cols.append(df.shift(-i))
		names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	
	agg = concat(cols, axis=1)
	agg.columns = names
	
	if dropnan:
		agg.dropna(inplace=True)
		
	return agg

def train_test_split(data, test_set_len):
	return data[:-test_set_len, :], data[-test_set_len:, :]



def random_forest_initial(train):
	# transform list into array if necessary
	train_ = np.asarray(train)
	# split into input and output columns
	trainX, trainy = train_[:, :-1], train_[:, -1]
	print("trainX: \n{}".format(trainX[0:10]))
	print("trainy:\n{}".format(trainy))
	print(80*"-")
	
	current_model = RandomForestRegressor(n_estimators=100, random_state=42)
	current_model.fit(trainX, trainy)
	return current_model

def xgboost_initial(train):
	train_ = np.asarray(train)
	# split into input and output columns
	trainX, trainy = train_[:, :-1], train_[:, -1]
	print("trainX: \n{}".format(trainX[0:10]))
	print("trainy:\n{}".format(trainy))
	print(80*"-")
	current_model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
	current_model.fit(trainX, trainy)
	return current_model

def cnn_initial(train):
    train_ = np.asarray(train)
    X_train, y_train = train_[:, :-1], train_[:, -1]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    window_size = X_train.shape[1]
    current_model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(window_size, 1)),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])

    current_model.compile(optimizer='adam', loss='mse')
    current_model.fit(X_train, y_train, epochs=200, verbose=0)
    return current_model



def walk_forward_validation_historic(data, test_set_len, model_selection, input_laggs=6):
	"""
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		test_set_len: Number of test set observations
		model_selection: String from ["RF","XGB","CNN"]
		input_laggs: Number of input laggs from data
	Returns:
		current_model:
		error: 
		test_label: returns the test labels (real values) for the test sequence via 
		prediction_array: returns all prediction via python list
	"""
	prediction_list = list()
	train_data, test_data = train_test_split(data.values, test_set_len) #receives np.array (df.values) 90x5 -> returns df 60x5, 30x5
	print("-.-\n-.-.-\ndata_shape:     "+ str(data.shape))
	print("train_data_shape: " + str(train_data.shape))
	print("test_data_shape: "+ str(test_data.shape) + "\n-.-.-\n-.-")
	if model_selection=="RF":
		current_model = random_forest_initial(train_data)
	elif model_selection=="XGB":
		current_model = xgboost_initial(train_data)
	elif model_selection=="CNN":
		current_model = cnn_initial(train_data)
		print("SUCCESS CNN")
	
	for i in range(len(test_data)):
		testX, testy = test_data[i, :-1], test_data[i, -1]      #--> 1x5; 1x1 from 30x5, 30x1
		if model_selection=="CNN":
			testX = testX.reshape((1, input_laggs, 1))
			y_pred=current_model.predict(testX)
		else:
			y_pred= current_model.predict([testX])
		prediction_list.append(y_pred[0])	
		print('>expected=%.1f, predicted=%.1f' % (testy, y_pred[0]))

	error = mean_absolute_error(test_data[:, -1], prediction_list)
	return current_model, error, test_data[:, -1], prediction_list



#-----#
#-----#
#-------------------------------------------------------------------------FOLLOWING FUNCTIONS IN DEVELOPMENT-------------------------#
#-----#
#-----#
def random_forest_forecast(train, testX):
	# transform list into array if necessary
	train_ = np.asarray(train)
	# split into input and output columns
	trainX, trainy = train_[:, :-1], train_[:, -1]
	print("trainX: \n{}".format(trainX[0:5]))
	print("trainy:\n{}".format(trainy))
	print(80*"-")
	
	model = RandomForestRegressor(n_estimators=300)
	model.fit(trainX, trainy)
	# make a one-step prediction
	#print("real value: {}".format(train))
	y_pred = model.predict([testX])
	return y_pred[0]


def walk_forward_validation_online(data, test_set_len):

	prediction_list = list()
	train_data, test_data = train_test_split(data, test_set_len)
	history = [x for x in train_data]     #--> initial: 60x6 
	for i in range(len(test_data)):		  #--> initial: 30x6	
		testX, testy = test_data[i, :-1], test_data[i, -1]     #--> 1x5; 1x1 from 30x5, 30x1
		y_pred = random_forest_forecast(history, testX)		   #--> fit rf on history, evaluate on test instance		
		prediction_list.append(y_pred)						   #--> build y_pred array	
		history.append(test_data[i])						   #--> append to history	
		print('>expected=%.1f, predicted=%.1f' % (testy, y_pred))
		# estimate prediction error
	print(test_data[:, -1])
	error = mean_absolute_error(test_data[:, -1], prediction_list)
	return error, test_data[:, -1], prediction_list
#-------------------------------------------------------------------------FUNCTIONS ABOVE IN DEVELOPMENT-------------------------#