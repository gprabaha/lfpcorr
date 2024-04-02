# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import LSTM, Dense, Reshape
from sklearn.preprocessing import StandardScaler

from util import *


# Function to create sequences
def create_sequences(data, seq_length):
    num_sequences = len(data) - seq_length
    sequences = np.empty((num_sequences, seq_length, data.shape[1]))
    for i in range(num_sequences):
        sequences[i] = data[i:i + seq_length]
    return sequences


parent_data_dir = "/gpfs/milgram/project/chang/pg496/data_dir/social_gaze"

pos_folder = os.path.join(parent_data_dir, 'social_gaze_eyetracking/aligned_raw_samples/position')

all_files = os.listdir(pos_folder)
mat_files_with_path = [os.path.join(pos_folder, file) for file in all_files if file.endswith('.mat')]

ex_file = load_mat(mat_files_with_path[0])

m1_positions = ex_file['var']['m1'][0][0]
m2_positions = ex_file['var']['m2'][0][0]
m1_positions = m1_positions.reshape((m1_positions.shape[1], 2))
m2_positions = m2_positions.reshape((m2_positions.shape[1], 2))


non_nan_indices = np.where(~np.isnan(m2_positions[:, 0]))[0]
# Extract corresponding columns from m1_positions
m1_positions = m1_positions[non_nan_indices]
m2_positions = m2_positions[non_nan_indices]

# Assuming 'X' is your feature matrix
scaler = StandardScaler()
m1_pos_scaled = scaler.fit_transform(m1_positions)
m2_pos_scaled = scaler.fit_transform(m2_positions)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(m1_pos_scaled[:-1], m1_pos_scaled[1:], test_size=0.2, shuffle=False)

# Define sequence length
seq_len = 200

# Create sequences for training
X_train_seq = create_sequences(X_train, seq_len)
y_train_seq = create_sequences(y_train, seq_len)

# Create sequences for testing
X_test_seq = create_sequences(X_test, seq_len)
y_test_seq = create_sequences(y_test, seq_len)

# Print the shapes of the sequences
print("Training sequences shape:", X_train_seq.shape)
print("Testing sequences shape:", X_test_seq.shape)

# Create the model
model = Sequential([
    LSTM(units=64, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]), return_sequences=True),
    LSTM(units=64),
    Dense(units=y_train_seq.shape[1]*y_train_seq.shape[2]),  # Output layer
    Reshape(target_shape=(y_train_seq.shape[1], y_train_seq.shape[2]))
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Print model summary
model.summary()

n_epochs = 1000
# Fit the model
history = model.fit(X_train_seq, y_train_seq, epochs=n_epochs, batch_size=64, validation_split=0.2)

# Evaluate the model on the test data
test_loss = model.evaluate(X_test_seq, y_test_seq)

print("Test Loss:", test_loss)

# Plot training and validation loss
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
