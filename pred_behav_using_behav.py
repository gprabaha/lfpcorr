#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 09:57:11 2024

@author: pg496
"""

import random
import os
import numpy as np
import multiprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf

from util import *
from nn_models import *
from plotting import *

n_epochs = 15
batch_size = 64
# Define the folder path
out_folder_path = "data/behav2behav_nn"
os.makedirs(out_folder_path, exist_ok=True)  # Create the folder if it doesn't exist
# Define the prefix for model filenames
file_prefix = "behav2behav_lstm"
run_parallel = False  # Set to False to train in serial

parent_data_dir = "/gpfs/milgram/project/chang/pg496/data_dir/social_gaze"
pos_folder = os.path.join(parent_data_dir, 'social_gaze_eyetracking/aligned_raw_samples/position')
all_files = os.listdir(pos_folder)
mat_files_with_path = [os.path.join(pos_folder, file) for file in all_files if file.endswith('.mat')]

random.seed(9)
ex_file = load_mat(random.choice(mat_files_with_path))
key = list(ex_file.keys())[-1]

m1_positions = ex_file[key]['m1'][0][0]
m2_positions = ex_file[key]['m2'][0][0]
pruned_m1_positions, pruned_m2_positions = remove_nans_in_both(m1_positions, m2_positions)

scaler = StandardScaler()
m1_pos_scaled = scaler.fit_transform(pruned_m1_positions)
m2_pos_scaled = scaler.fit_transform(pruned_m2_positions)


# Split data for m1 and m2 pos
X_train_m1, X_test_m1, y_train_m1, y_test_m1 = train_test_split(m1_pos_scaled[:-1], m1_pos_scaled[1:], test_size=0.2, shuffle=False)
X_train_m2, X_test_m2, y_train_m2, y_test_m2 = train_test_split(m2_pos_scaled[:-1], m2_pos_scaled[1:], test_size=0.2, shuffle=False)

# Create sequences to fit LSTM
seq_length = 200
X_train_seq_m1, y_train_seq_m1, X_test_seq_m1, y_test_seq_m1 = create_train_test_seq_for_lstm(X_train_m1, X_test_m1, y_train_m1, y_test_m1, seq_length)
X_train_seq_m2, y_train_seq_m2, X_test_seq_m2, y_test_seq_m2 = create_train_test_seq_for_lstm(X_train_m2, X_test_m2, y_train_m2, y_test_m2, seq_length)

# Generated model instances for each combination
lstm_model_m1_m1 = make_n_layer_lstm(num_lstm_layers=2, lstm_units=64, optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', input_shape=(X_train_seq_m1.shape[1], X_train_seq_m1.shape[2]))
lstm_model_m2_m2 = make_n_layer_lstm(num_lstm_layers=2, lstm_units=64, optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', input_shape=(X_train_seq_m1.shape[1], X_train_seq_m1.shape[2]))
lstm_model_m1_m2 = make_n_layer_lstm(num_lstm_layers=2, lstm_units=64, optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', input_shape=(X_train_seq_m1.shape[1], X_train_seq_m1.shape[2]))
lstm_model_m2_m1 = make_n_layer_lstm(num_lstm_layers=2, lstm_units=64, optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', input_shape=(X_train_seq_m1.shape[1], X_train_seq_m1.shape[2]))

# Define combinations
combinations = [("m1", "m1"), ("m2", "m2"), ("m1", "m2"), ("m2", "m1")]

file_name = ex_file[key]['unified_filename'][0][0][0]
file_name = os.path.splitext(os.path.basename(file_name))[0]  # Extract filename without extension

# Models and corresponding sequences
models = [lstm_model_m1_m1, lstm_model_m2_m2, lstm_model_m1_m2, lstm_model_m2_m1]
X_train_seqs = [X_train_seq_m1, X_train_seq_m2, X_train_seq_m1, X_train_seq_m2]
y_train_seqs = [y_train_seq_m1, y_train_seq_m2, y_train_seq_m2, y_train_seq_m1]
X_test_seqs = [X_test_seq_m1, X_test_seq_m2, X_test_seq_m2, X_test_seq_m1]
y_test_seqs = [y_test_seq_m1, y_test_seq_m2, y_test_seq_m1, y_test_seq_m2]

# Function to train a model and save it
def train_and_save_model(model, X_train, y_train, X_test, y_test, file_prefix, combination, folder_path):
    print(f"Fitting model for combination: {combination}")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=n_epochs, batch_size=batch_size, verbose=1)
    model.save(os.path.join(folder_path, f"{file_prefix}_{combination[0]}_{combination[1]}.keras"))

# Training and saving either in parallel or serially
if run_parallel:
    processes = []
    for model, X_train_seq, y_train_seq, X_test_seq, y_test_seq, combination in zip(models, X_train_seqs, y_train_seqs, X_test_seqs, y_test_seqs, combinations):
        process = multiprocessing.Process(target=train_and_save_model, args=(model, X_train_seq, y_train_seq, X_test_seq, y_test_seq, file_prefix, combination, out_folder_path))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
else:
    for model, X_train_seq, y_train_seq, X_test_seq, y_test_seq, combination in zip(models, X_train_seqs, y_train_seqs, X_test_seqs, y_test_seqs, combinations):
        train_and_save_model(model, X_train_seq, y_train_seq, X_test_seq, y_test_seq, file_prefix, combination, out_folder_path)

# Plotting training and validation losses
plt.figure(figsize=(10, 8))
for i, model, combination in zip(range(len(models)), models, combinations):
    plt.subplot(2, 2, i+1)
    plt.plot(model.history.history['loss'], label='Training Loss')
    plt.plot(model.history.history['val_loss'], label='Validation Loss')
    plt.title(f"Model {combination[0]} to {combination[1]}: {file_name}")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
plt.tight_layout()
plt.show()
