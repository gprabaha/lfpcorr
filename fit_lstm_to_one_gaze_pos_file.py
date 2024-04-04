#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:37:10 2024

@author: pg496
"""

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

# Create sequences to fit LSTM
seq_length = 200
n_epochs = 15
batch_size = 64
file_prefix = "behav2behav_lstm"
run_parallel = False  # Set to False to train in serial

top_out_dir = "/gpfs/milgram/project/chang/pg496/data_dir/social_gaze/gaze_pred_gaze_lstm"
if len(sys.argv) < 2:
    print("Usage: python fit_lstm_to_one_gaze_file.py <file_path>")
    sys.exit(1)
file_path = sys.argv[1]
file_name = os.path.basename(file_path)
f_name_wo_ext = file_name.split('.')[0]
# Split filename into parts based on underscores
session, detail, run = file_name.split('_')
# Extract the run number from the run part of the filename
run_number = int(run.split('.')[0])
out_dir = os.path.join(top_out_dir, session)
os.makedirs(out_dir, exist_ok=True)

pos_file = load_mat(file_path)
key = list(pos_file.keys())[-1]

m1_positions = pos_file[key]['m1'][0][0]
m2_positions = pos_file[key]['m2'][0][0]
pruned_m1_positions, pruned_m2_positions = remove_nans_in_both(m1_positions, m2_positions)

scaler = StandardScaler()
scaler.fit(pruned_m1_positions)
m1_pos_scaled = scaler.fit_transform(pruned_m1_positions)
m2_pos_scaled = scaler.fit_transform(pruned_m2_positions)

# Split data for m1 and m2 pos
X_train_m1, X_test_m1, y_train_m1, y_test_m1 = train_test_split(m1_pos_scaled[:-1], m1_pos_scaled[1:], test_size=0.2, shuffle=False)
X_train_m2, X_test_m2, y_train_m2, y_test_m2 = train_test_split(m2_pos_scaled[:-1], m2_pos_scaled[1:], test_size=0.2, shuffle=False)

X_train_seq_m1, y_train_seq_m1, X_test_seq_m1, y_test_seq_m1 = create_train_test_seq_for_lstm(X_train_m1, X_test_m1, y_train_m1, y_test_m1, seq_length)
X_train_seq_m2, y_train_seq_m2, X_test_seq_m2, y_test_seq_m2 = create_train_test_seq_for_lstm(X_train_m2, X_test_m2, y_train_m2, y_test_m2, seq_length)

# Generated model instances for each combination
lstm_model_m1_m1 = make_n_layer_lstm(num_lstm_layers=2, lstm_units=64, optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', input_shape=(X_train_seq_m1.shape[1], X_train_seq_m1.shape[2]))
lstm_model_m2_m2 = make_n_layer_lstm(num_lstm_layers=2, lstm_units=64, optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', input_shape=(X_train_seq_m1.shape[1], X_train_seq_m1.shape[2]))
lstm_model_m1_m2 = make_n_layer_lstm(num_lstm_layers=2, lstm_units=64, optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', input_shape=(X_train_seq_m1.shape[1], X_train_seq_m1.shape[2]))
lstm_model_m2_m1 = make_n_layer_lstm(num_lstm_layers=2, lstm_units=64, optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', input_shape=(X_train_seq_m1.shape[1], X_train_seq_m1.shape[2]))

# Define combinations
combinations = [("m1", "m1"), ("m2", "m2"), ("m1", "m2"), ("m2", "m1")]

# Models and corresponding sequences
models = [lstm_model_m1_m1, lstm_model_m2_m2, lstm_model_m1_m2, lstm_model_m2_m1]
X_train_seqs = [X_train_seq_m1, X_train_seq_m2, X_train_seq_m1, X_train_seq_m2]
y_train_seqs = [y_train_seq_m1, y_train_seq_m2, y_train_seq_m2, y_train_seq_m1]
X_test_seqs = [X_test_seq_m1, X_test_seq_m2, X_test_seq_m2, X_test_seq_m1]
y_test_seqs = [y_test_seq_m1, y_test_seq_m2, y_test_seq_m1, y_test_seq_m2]

# Function to train a model and save it
def train_and_save_model(model, X_train, y_train, X_test, y_test, file_name, file_prefix, combination, out_dir):
    print(f"Fitting model for combination: {combination}")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=n_epochs, batch_size=batch_size, verbose=1)
    model.save(os.path.join(out_dir, f"{file_prefix}_{file_name}_{combination[0]}_{combination[1]}.keras"))

# Training and saving either in parallel or serially
if run_parallel:
    processes = []
    for model, X_train_seq, y_train_seq, X_test_seq, y_test_seq, combination in zip(models, X_train_seqs, y_train_seqs, X_test_seqs, y_test_seqs, combinations):
        process = multiprocessing.Process(target=train_and_save_model, args=(model, X_train_seq, y_train_seq, X_test_seq, y_test_seq, file_name, file_prefix, combination, out_dir))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
else:
    for model, X_train_seq, y_train_seq, X_test_seq, y_test_seq, combination in zip(models, X_train_seqs, y_train_seqs, X_test_seqs, y_test_seqs, combinations):
        train_and_save_model(model, X_train_seq, y_train_seq, X_test_seq, y_test_seq, file_name, file_prefix, combination, out_dir)

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
plt.savefig( os.path.join(out_dir, f"{file_prefix}_{file_name}.png") )
