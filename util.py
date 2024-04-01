#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:48:26 2024

@author: prabaha
"""

import h5py
import scipy
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal import get_window
from scipy.fft import rfft, rfftfreq
from tqdm import tqdm
from joblib import Parallel, delayed


# Function to load MAT file
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