#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 09:55:43 2024

@author: pg496
"""

from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, Reshape
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

def make_n_layer_lstm(num_lstm_layers=2, lstm_units=64, optimizer='adam', loss='categorical_crossentropy', input_shape=(None, None)):
    model = Sequential()
    model.add(Input(shape=input_shape))

    for _ in range(num_lstm_layers - 1):
        model.add(LSTM(units=lstm_units, return_sequences=True))

    model.add(LSTM(units=lstm_units))

    model.add(Dense(units=input_shape[0] * input_shape[1]))
    model.add(Reshape(target_shape=(input_shape[0], input_shape[1])))

    model.compile(optimizer=optimizer, loss=loss)

    return model

if __name__ == "__main__":
    # Example usage
    lstm_model = make_n_layer_lstm(num_lstm_layers=3, lstm_units=128, optimizer=Adam(lr=0.001), loss='mean_squared_error', input_shape=(10, 5))
    print(lstm_model.summary())