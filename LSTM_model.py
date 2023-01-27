# Import necessary libraries
from keras.models import Sequential
from keras.layers import LSTM, Dense
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

# Normalizes the df prices column, then
# Divides the df, in lists of size "num_time_steps" with prices values, and saves it on X list
# Y list has for each index X sublist, the correspondig  "num_time_steps+1"th price.
# i.e. returns X - sequences of previous values, y - the correspodent future value.
def prepare_data(df, num_time_steps=10, num_features=1):
    #Normalize Prices
    df = normalize_z_score(df,'prices')
    # Create empty lists to store X and y
    X, y = [], []
    # Iterate over the dataframe
    for i in range(len(df) - num_time_steps):
        X.append(df.iloc[i:i+num_time_steps]['prices'].values)
        y.append(df.iloc[i+num_time_steps]['prices'])
    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)
    #Reshape X to fit the LSTM format     
    X = X.reshape((X.shape[0], num_time_steps, num_features))
    return X, y

#Cretes a Simple LSTM model
#Fits the prices data
#Show the results
#Returns the trained model and the training data set predictions
def LSTM_Train (df, num_time_steps=10, units=50, epochs=20, batch_size=32, num_features=1):

    X, y = prepare_data(df, num_time_steps=num_time_steps, num_features=num_features)
    
    # Create the LSTM model
    model = Sequential()
    model.add(LSTM(units,return_sequences=False))
    model.add(Dense(1))

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(X, y, epochs=epochs, batch_size=batch_size)

    # Make predictions on the test set
    y_pred = model.predict(X)

    #check train results
    results(y,y_pred)
    return y_pred, model

def LSTM_Eval(LSTM, df_test, num_time_steps=10,  num_features=1):
    #Test Trained Model on Val data 
    X, y = prepare_data(df_test,num_time_steps=num_time_steps, num_features=num_features)
    LSTM.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity'])
    
    score = LSTM.evaluate(X, y , verbose=0)
    for i in range(5):
        print("%s: %.2f%%" % (LSTM.metrics_names[i], score[i]*100))

    y_pred = LSTM.predict(X)
    results(y,y_pred)
    return y_pred, LSTM
    

# Plot the real values against the predictions
def results(real_values, predictions):
    plt.figure(figsize=(13, 4))
    plt.plot(real_values, label='Real values')
    plt.plot(predictions, label='Predicted values')
    plt.legend()
    plt.ylim(-2, 8)
    plt.show()



if __name__ == "__main__":

    #Hyperparameters
    num_time_steps=30
    units=50
    epochs=10
    batch_size=32
    num_features=1

    df = pd.read_csv('data/train_processed.csv')
    #Creating and Training the LSTM Model
    y_pred, model = LSTM_Train(df, num_time_steps=num_time_steps, units=units, epochs=epochs, batch_size=batch_size)
    
    #To play with the hyper parameters, we are always training the model again
    '''with open(f'./LSTMmodel.pickle', 'wb') as f:
       pickle.dump(model, f)
    with open(f'./LSTMmodel.pickle','rb') as f:
        model = pickle.load(f)'''
    df_test = pd.read_csv('data/val.csv')
    y_pred, final_model = LSTM_Eval(model, df_test, num_time_steps=num_time_steps)
    

 
  
    
