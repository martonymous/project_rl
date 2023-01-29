# Import necessary libraries
from keras.models import Sequential
from keras.layers import LSTM, Dense
from scipy.optimize import curve_fit
from scipy.misc import derivative
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle


#Normalizes df column by Z score method
def normalize_z_score(df, column_name):
    mean = df[column_name].mean()
    std = df[column_name].std()
    df[column_name] = (df[column_name] - mean) / std
    return df


def prepare_data(df, num_time_steps=10, num_predict_steps=6, num_features=1):
    #Normalize Prices
    df = normalize_z_score(df,'prices')
    # Create empty lists to store X and y
    X, Y = [], []
    # Iterate over the dataframe
    for i in range(len(df) - (num_time_steps+num_predict_steps)):
        X.append(df.iloc[i:i+num_time_steps]['prices'].values)

    for i in range(num_time_steps, len(df) -num_predict_steps):
        Y.append(df.iloc[i:i+num_predict_steps]['prices'].values)

    X = np.array(X)
    Y = np.array(Y)
    #print("X.shape: ", X.shape,"    Y.shape: ",Y.shape)

    #Reshape X to fit the LSTM format     
    X = X.reshape((X.shape[0], num_time_steps, num_features))

    return X, Y

#Cretes a Simple LSTM model
#Fits the prices data
#Show the results
#Returns the trained model and the training data set predictions
def LSTM_Train (df, num_time_steps=10, num_predict_steps=6, units=50, epochs=20, batch_size=32, num_features=1):

    X, Y = prepare_data(df, num_time_steps=num_time_steps, num_predict_steps=num_predict_steps, num_features=num_features)
    
    # Create the LSTM model
    model = Sequential()
    #try units=200, activation='relu'
    
    model.add(LSTM(units,return_sequences=False))
    model.add(Dense(num_predict_steps))

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(X, Y, epochs=epochs, batch_size=batch_size)

    Y_pred = model.predict(X)

    return model, Y_pred

def LSTM_Eval(LSTM, df_test, num_time_steps=10, num_predict_steps=6, num_features=1):
    #Test Trained Model on Val data 
    X, Y = prepare_data(df_test, num_time_steps=num_time_steps, num_predict_steps=num_predict_steps)
    LSTM.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity'])
    score = LSTM.evaluate(X, Y , verbose=0)
    for i in range(5):
        print("%s: %.2f%%" % (LSTM.metrics_names[i], score[i]*100))
    
    Y_pred = LSTM.predict(X)

    #print("Y_pred shape",Y_pred.shape,"Y shape",Y.shape)

    #Y_true = []
    #for i in range(num_time_steps, len(df_test) - num_predict_steps):
    #    Y_true.append(df_test.iloc[i+num_predict_steps]['prices'])
    #Y_true = np.array(Y_true)

    #avg_Y_pred=np.mean(Y_pred, axis=1)
    #avg_Y=np.mean(Y, axis=1)
    #results(Y_true,avg_Y_pred,avg_Y)
    #find_extreme_predictions(Y_pred, X)
    #crucial_time_points=find_null_derivatives_points(Y_pred)
    #positive_change, negative_change = find_derivatives_signal_changes(Y_pred, X)
    #plt.vlines(x = positive_change, ymin = -2, ymax = 8, colors = 'red')
    #plt.vlines(x = negative_change, ymin = -2, ymax = 8, colors = 'green')
    #plt.plot(Y_true, label='Real values')
    #plt.plot(avg_Y_pred, label='Predicted Average')
    #plt.legend()
    #plt.ylim(-2, 8)
    #plt.show()
    actionlist=assign_action_points(Y_pred, X)
    #print(actionlist[:100])
    return LSTM, Y_pred

#Given a set of points Y, fit the data with a non-linear function
#And return the derivatives of every time point (X=[0,...,len(Y)])
def fit_and_derivative(Y):
    x = list(range(len(Y)))
    # Define the non-linear function to fit the data
    def func(x, a, b, c):
        return a * x**2 + b * x + c
    # Fit the function to the data
    popt, _ = curve_fit(func, x, Y)
    # Define a function for the derivative of the fitted function
    def derivative_func(x):
        return derivative(lambda x: func(x, *popt), x)
    # Calculate the derivative at every x point
    derivatives = [derivative_func(x_i) for x_i in x]
    derivatives = np.array(derivatives)
    return derivatives, Y

def has_extreme(values_list, mean, std):
    boolean=False
    threshold = 1 * std
    lower, upper = mean - threshold, mean + threshold
    for value in values_list:
        if value < lower or value > upper:
            boolean = True
    return boolean

#assuming there will only be one change of derivative signal on the provided value list
def signal_change(values_list):
    change=''
    derivatives , ys = fit_and_derivative(values_list)
    for i, derivative in enumerate(derivatives):
        if (derivatives[i-1] < 0 and derivatives[i] > 0):
            change = 'positive'
        elif(derivatives[i-1] > 0 and derivatives[i] < 0):
            change = 'negative'
    return change

def assign_action_points(Y_pred, X):
    inital_hold = Y_pred.shape[1]+X.shape[1]
    action_list = list(np.zeros(inital_hold))
    for i in range(len(X)):
        if has_extreme(Y_pred[i], np.mean(X[i]), np.std(X[i])):
            if signal_change(Y_pred[i])=='positive':
                action_list.append(1)
            elif signal_change(Y_pred[i])=='negative':
                action_list.append(-1)
            else:
                action_list.append(0)
        else:
            action_list.append(0)
    action_list = np.roll(action_list, -2)
    action_list = change_array(action_list)
    return action_list


def change_array(arr):
    result = arr.copy()
    for i in range(1, len(arr) - 1):
        if arr[i] == 1:
            result[i - 1] = 1
            result[i + 1] = 1
        elif arr[i] == -1:
            result[i - 1] = -1
            result[i + 1] = -1
    return result

# Plot the real values against the predictions
def results(Y_true,avg_Y_pred,avg_Y):
    plt.figure(figsize=(13, 4))
    plt.plot(Y_true, label='Real values')
    plt.plot(avg_Y, label='Real Average')
    plt.plot(avg_Y_pred, label='Predicted Average')

    plt.legend()
    plt.ylim(-2, 8)
    plt.show()


if __name__ == "__main__":
    #Hyperparameters
    num_time_steps = 25
    #assuming the periodicity is 24,25
    num_predict_steps=5
    units=150
    epochs=10
    batch_size=32
    num_features=1

    df = pd.read_csv('data/train_processed.csv')
    print("df.shape", df.shape)
    model, y_pred = LSTM_Train(df, num_time_steps=num_time_steps,num_predict_steps=num_predict_steps, units=units, epochs=epochs, batch_size=batch_size)
    
    #To play with the hyper parameters, we are always training the model again
    '''with open(f'./LSTMmodel.pickle', 'wb') as f:
       pickle.dump(model, f)
    with open(f'./LSTMmodel.pickle','rb') as f:
        model = pickle.load(f)'''

    df_test = pd.read_csv('data/val.csv')
    LSTM, Y_pred = LSTM_Eval(model, df_test,num_time_steps=num_time_steps,num_predict_steps=num_predict_steps)

    #Maybe instead of looking for points with derivative close 0.
    #we should map where the slopes change from positive to negative.
    #Because if the change on the behaviour of the price is 
    #more like a spike instead of a Curve, maybe we can't find a derivative 0 there,
    #but in fact the previous point had slope -1 and the next one +1

    #So looking for changes in derivative signal rather then looking for zeros, seems better

    #Also just looking for changes is not that complete
    #since small changes can happen, with small variations in the signal

    #So we should target, changes in signal  where the price values are far away
    #from the mean of a last segment of points.

    #Not just changes on the signal behaviour, but meaningful changes.
    # 
    #Check the period. Normally the period is around 25 time steps.
    # Normal distance between 2 low points 

'''#Given a List of lists of price predictions
#returns the indexs where those price/time points have a derivative x
#Where -0.005 > x < 0.005
def find_null_derivatives_points(Y_pred):
    derivative_prediction=[]

    for idx1, y_predictions in enumerate(Y_pred):
        derivatives = fit_and_derivative(y_predictions)
        for idx2, derivative in enumerate(derivatives):
            if (derivatives[idx2] < 0.005 and derivatives[idx2] > -0.005):
            #if derivative ==0 :
                derivative_prediction.append(idx1+idx2)

    derivative_prediction,i=np.unique(derivative_prediction, return_index=True)
    print(len(derivative_prediction))
    print(derivative_prediction[:20])
    return derivative_prediction

#Given a List of lists of price predictions
#returns the indexs of crucial points,
#A Crucial point i, has i-1  with negative derivative 
#And i+1 with positive derivative 
def find_derivatives_signal_changes(Y_pred, X):
    positive_change=[]
    negative_change=[]
    for idx1, y_predictions in enumerate(Y_pred):
        for previous in X:
            mean=np.mean(previous, axis=1)
            #print('mean',mean)
        derivatives , y_predictions = fit_and_derivative(y_predictions)
        for idx2, derivative in enumerate(derivatives):
            if (derivatives[idx2-1] < 0 and derivatives[idx2] > 0):
                positive_change.append(idx1+(idx2-1))
            if (derivatives[idx2-1] > 0 and derivatives[idx2] < 0):
                negative_change.append(idx1+(idx2-1))

    positive_change,i=np.unique(positive_change, return_index=True)
    negative_change,i=np.unique(negative_change, return_index=True)        
    print("Low_points",len(positive_change))
    print(positive_change[:5])
    print("High_points",len(negative_change))
    print(negative_change[:5])

    return positive_change, negative_change'''  