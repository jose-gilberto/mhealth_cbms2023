# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 19:31:32 2023

@author: lucas
"""
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_squared_error
import numpy as np


def build_model(num_layers, cells_per_layer,windowing = False):
    tam = 900
    if windowing:
        tam= 300
    regressor = Sequential()
    
    for i in range(num_layers):
        regressor.add(LSTM(units = cells_per_layer[i],
                           return_sequences = not(i == num_layers - 1), input_shape = (tam,3)))
        regressor.add(Dropout(0.4))
        
    #regressor.add(Dense(units = 4, activation = 'sigmoid'))
    #regressor.add(Dropout(0.4))
    regressor.add(Dense(units = 1, activation = 'sigmoid'))
    
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mean_absolute_error'])
    regressor.save_weights('ini_model.h5')
    
    return regressor

        
def train(model, X_train, y_train,epochs):
    model.load_weights('ini_model.h5')
    rlr = ReduceLROnPlateau(monitor = 'val_loss', min_delta=0.0001,patience = 5,verbose = 1)
    es = EarlyStopping(monitor = 'val_loss',patience = 10,verbose = 1,min_delta = 0)
    model.fit(X_train,y_train,epochs = epochs,batch_size = 32,validation_split =0.2, callbacks = [rlr,es])
    
    return model


def test(model, X_test, y_test):
    inference = model.predict(X_test)    
    return inference


def K_fold_cv(time_series,parameters,k,normalized,normalizer = None,augmented = False, original_indices = None,windowing = False):
    size = len(time_series)
    errors = []
    predictions= np.array([])
    
    original_time_series = time_series
    
    if augmented:
        original_time_series = time_series[original_indices]
        
    kf = KFold(n_splits=k, shuffle=True, random_state=1)
    
    for train_index,test_index in kf.split(original_time_series):
        
        model = build_model(1,[3],windowing = windowing)
        
        X_train = time_series[np.setdiff1d(range(0,len(time_series)), test_index)]
        y_train = parameters[np.setdiff1d(range(0,len(time_series)), test_index)]
        
        X_test,y_test = time_series[test_index],parameters[test_index]
        
        model = train(model,X_train,y_train,epochs = 200)
        
        prediction = test(model,X_test,y_test)
        if normalized:
            prediction = normalizer.inverse_transform(prediction)
            y_test = normalizer.inverse_transform(y_test)
        
        print("REAL: ")
        print(y_test)
        
        print("\nPREDICTIONS:" )
        print(prediction)
        predictions = np.append(predictions, prediction)
        
        print("\nMSE: " + str(mean_squared_error(y_test, prediction)))
        errors.append(mean_squared_error(y_test, prediction))
        
    return np.median(errors),errors,predictions
