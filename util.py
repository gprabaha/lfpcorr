#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:48:26 2024

@author: prabaha
"""

import h5py
import scipy
import os
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal import get_window
from scipy.fft import rfft, rfftfreq
from tqdm import tqdm
from joblib import Parallel, delayed
import tensorflow as tf
from keras.models import clone_model


def load_mat(file_path):
    """
    Load MAT file.

    Parameters:
        file_path (str): Path to the MAT file.

    Returns:
        h5py.File or dict: Loaded MAT file object.
    """
    print("Loading mat file...")
    try:
        f = h5py.File(file_path, 'r')
    except:
        f = scipy.io.loadmat(file_path)
    return f


# Function to transpose data if necessary
def maybe_transpose(ts):
    """
    Transpose data if necessary to ensure consistent shape.

    Parameters:
        ts (numpy.ndarray): Input data.

    Returns:
        numpy.ndarray: Transposed or original data.
    """
    if ts.shape[0] > ts.shape[1]:
        return ts.T
    else:
        return ts


# Convert MAT data to numpy array
def mat_to_timeseries(f):
    """
    Convert MAT data to numpy array.

    Parameters:
        f (h5py.File or dict): Loaded MAT file object.

    Returns:
        numpy.ndarray: Numpy array containing timeseries data.
    """
    print("Converting MAT data to numpy array...")
    timeseries = np.array(f['mat'])
    timeseries = timeseries.astype(float)
    return maybe_transpose(timeseries)


# Apply filter on individual channel
def apply_filter_on_channel(b, a, data):
    """
    Apply filter on individual channel.

    Parameters:
        b (numpy.ndarray): Numerator coefficients of the filter.
        a (numpy.ndarray): Denominator coefficients of the filter.
        data (numpy.ndarray): Input data.

    Returns:
        numpy.ndarray: Filtered data.
    """
    return filtfilt(b, a, data)


# Butterworth lowpass filter
def butter_lowpass_filter(data, cutoff_freq, sampling_rate, order=5):
    """
    Applies a Butterworth lowpass filter to the input data.

    Parameters:
        data (numpy.ndarray): Input data to be filtered. Shape (num_channels, num_samples).
        cutoff_freq (float): Cutoff frequency for the lowpass filter (in Hz).
        sampling_rate (float): Sampling rate of the input data (in Hz).
        order (int): Order of the Butterworth filter. Default is 5.

    Returns:
        numpy.ndarray: Filtered data.
    """
    print("Applying lowpass filter...")
    nyquist_freq = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist_freq
    # Design Butterworth lowpass filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    num_channels = data.shape[0]
    # Parallel computation of filtered data
    results = Parallel(n_jobs=-1)(
        delayed(apply_filter_on_channel)(b, a, data[i,])
        for i in range(num_channels)
    )
    
    return np.array(results)


# Process a single channel of data by performing FFT
def process_channel(channel_data, window, window_length):
    """
    Process a single channel of data by performing FFT and computing power and phase spectra.

    Parameters:
        channel_data (numpy.ndarray): Data of a single channel for FFT computation.
        window (numpy.ndarray): Tapered window for FFT.
        window_length (int): Length of the window.

    Returns:
        tuple: Tuple containing power spectrum and phase spectrum of the channel.
    """
    # Perform FFT on the channel data
    fft_result = rfft(channel_data * window)
    # Compute power spectrum
    power_spectrum = np.abs(fft_result) ** 2
    # Compute phase spectrum
    phase_spectrum = np.angle(fft_result)
    return power_spectrum, phase_spectrum


# Perform windowed FFT on filtered data in parallel
def windowed_fft_parallel(filtered_data, sampling_rate, window_duration=0.4, window_type='kaiser', beta=14, stride=0.1, num_workers=-1):
    """
    Perform windowed FFT on the filtered data in parallel across channels.

    Parameters:
        filtered_data (numpy.ndarray): Filtered time series data. Shape (num_channels, num_samples).
        sampling_rate (float): Sampling rate of the data (in Hz).
        window_duration (float): Duration of the tapered window in seconds. Default is 0.4 seconds.
        window_type (str): Type of window to use. Default is 'kaiser'.
        beta (float): Shape parameter for the Kaiser window. Default is 14.
        stride (float): Stride length within the window moving, in seconds. Default is 0.1.
        num_workers (int): Number of parallel workers. Default is -1, which uses all available CPU cores.

    Returns:
        tuple: Tuple containing frequencies, power spectra, phase spectra, window parameters, and timestamps.
    """
    num_channels, num_samples = filtered_data.shape
    window_length = int(window_duration * sampling_rate)
    freqs = rfftfreq(window_length, 1 / sampling_rate)
    window = get_window((window_type, beta), window_length)
    
    num_windows = len(range(0, num_samples - window_length, int(sampling_rate * stride)))
    num_frequency_bins = window_length // 2 + 1
    
    # Initialize arrays to store power and phase spectra
    power_spectra = np.full((num_windows, num_channels, num_frequency_bins), np.nan)
    phase_spectra = np.full((num_windows, num_channels, num_frequency_bins), np.nan)
    timestamps = np.full((num_windows, 2), np.nan)
    
    # Iterate over windows
    for idx, j in enumerate(tqdm(range(0, num_samples - window_length, int(sampling_rate * stride)), desc="rFFT Window")):
        start_time = j / sampling_rate
        end_time = (j + window_length) / sampling_rate
        timestamps[idx] = (start_time, end_time)
    
        # Parallelize FFT computation across channels
        results = Parallel(n_jobs=num_workers)(
            delayed(process_channel)(filtered_data[i, j:j + window_length], window, window_length)
            for i in range(num_channels)
        )
        # Extract power and phase spectra from results
        for i, result in enumerate(results):
            power_spectrum = result[0]
            phase_spectrum = result[1]
            
            # Normalize spectra
            power_spectrum /= window_length
            phase_spectrum /= window_length
    
            # Assign to pre-allocated arrays
            power_spectra[idx, i] = power_spectrum
            phase_spectra[idx, i] = phase_spectrum

    # Store window parameters
    window_params = {'window_type': window_type, 'duration': window_duration, 'beta': beta}

    return freqs, power_spectra, phase_spectra, window_params, timestamps


def create_sequences(data, seq_length):
    """
    Creates sequences of a specified length from the input data.

    Args:
        data (numpy.ndarray): The input data.
        seq_length (int): The length of each sequence.

    Returns:
        numpy.ndarray: Sequences of the specified length.
    """
    num_sequences = len(data) - seq_length
    sequences = np.empty((num_sequences, seq_length, data.shape[1]))
    for i in range(num_sequences):
        sequences[i] = data[i:i + seq_length]
    return sequences


def remove_nans_in_both(array1, array2):
    """
    Removes rows with NaNs from both input arrays using the indices of NaNs in both arrays.

    Args:
        array1 (numpy.ndarray): First input array.
        array2 (numpy.ndarray): Second input array.

    Returns:
        tuple: Tuple containing the pruned versions of both input arrays.
    """
    # Convert arrays to float
    array1 = array1.astype(float)
    array2 = array2.astype(float)
    
    # Reshape arrays
    array1 = array1.reshape((array1.shape[1], 2))
    array2 = array2.reshape((array2.shape[1], 2))
    
    # Find rows with NaNs in array2
    nan_indices_array2 = np.isnan(array2).any(axis=1)
    
    # Remove rows with NaNs from both arrays
    array1 = array1[~nan_indices_array2]
    array2 = array2[~nan_indices_array2]
    
    # Find rows with NaNs in array1
    nan_indices_array1 = np.isnan(array1).any(axis=1)
    
    # Remove rows with NaNs from both arrays
    array1 = array1[~nan_indices_array1]
    array2 = array2[~nan_indices_array1]
    
    return array1, array2


def create_train_test_seq_for_lstm(X_train, X_test, y_train, y_test, seq_length):
    """
    Creates sequences from input arrays for both training and testing data, suitable for LSTM models.

    Args:
        X_train (numpy.ndarray): Input features for training data.
        X_test (numpy.ndarray): Input features for testing data.
        y_train (numpy.ndarray): Target values for training data.
        y_test (numpy.ndarray): Target values for testing data.
        seq_length (int): Length of each sequence.

    Returns:
        tuple: Tuple containing sequences for training and testing data: (X_train_seq, y_train_seq, X_test_seq, y_test_seq)
    """
    # Create sequences for training
    X_train_seq = create_sequences(X_train, seq_length)
    y_train_seq = create_sequences(y_train, seq_length)

    # Create sequences for testing
    X_test_seq = create_sequences(X_test, seq_length)
    y_test_seq = create_sequences(y_test, seq_length)

    return X_train_seq, y_train_seq, X_test_seq, y_test_seq


# Define a function to fit the model
def fit_model(model, X_train_seq, y_train_seq, n_epochs, batch_size, validation_split):
    """
    Fit the provided model to the training data.

    Parameters:
        model: Keras model object
            The model to be trained.
        X_train_seq: numpy.ndarray
            Input sequences for training.
        y_train_seq: numpy.ndarray
            Target sequences for training.
        n_epochs: int
            Number of epochs for training.
        batch_size: int
            Batch size for training.
        validation_split: float
            Fraction of the training data to be used as validation data.

    Returns:
        History object: Object containing training metrics.
    """
    return model.fit(X_train_seq, y_train_seq, epochs=n_epochs, batch_size=batch_size, validation_split=validation_split)

# Define a function to distribute the training across multiple GPUs
def train_model_parallel(strategy, model, X_train_seq, y_train_seq, n_epochs, batch_size, validation_split):
    """
    Train the provided model in parallel across multiple GPUs.

    Parameters:
        strategy: tf.distribute.Strategy object
            The distribution strategy to use for training across multiple GPUs.
        model: Keras model object
            The model to be trained.
        X_train_seq: numpy.ndarray
            Input sequences for training.
        y_train_seq: numpy.ndarray
            Target sequences for training.
        n_epochs: int
            Number of epochs for training.
        batch_size: int
            Batch size for training.
        validation_split: float
            Fraction of the training data to be used as validation data.

    Returns:
        Tuple: A tuple containing the distributed model and training history object.
    """
    with strategy.scope():
        distributed_model = clone_model(model)
        distributed_model.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics)
    history = fit_model(distributed_model, X_train_seq, y_train_seq, n_epochs, batch_size, validation_split)
    return distributed_model, history


def file_sorting_key(filepath):
    """
    Generate sorting key for file paths based on session and run number.

    Parameters:
        filepath (str): The file path to extract session and run number from.

    Returns:
        tuple: A tuple containing session and run number, used for sorting.
               Returns (None, None) if the file detail is not 'position'.
    """
    # Extract filename from the filepath
    filename = os.path.basename(filepath)

    # Split filename into parts based on underscores
    session, detail, run = filename.split('_')

    # Discard files if detail is not 'position'
    if detail != 'position':
        return None, None  # Return (None, None) to indicate this file should not be considered for sorting

    # Extract the run number from the run part of the filename
    run_number = int(run.split('.')[0])  # Assumes run part ends with '.mat' extension

    # Return a tuple containing session and run number
    return session, run_number


