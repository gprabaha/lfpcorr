#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:48:22 2024

@author: prabaha
"""

# from kymatio.numpy import Scattering1D
import os
import h5py
import scipy
from scipy.signal import butter, filtfilt
from scipy.signal import get_window
from scipy.fft import rfft, rfftfreq
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


def load_mat(file_path):
    print("Loading mat...")
    try:
        f = h5py.File(file_path, 'r')
    except:
        f = scipy.io.loadmat(file_path)
    return f



def maybe_transpose(ts):
    if ts.shape[0] > ts.shape[1]:
        return ts.T
    else:
        return ts



def mat_to_timeseries(f):
    print("Converting ts to np array...")
    timeseries = np.array(f['mat'])
    timeseries = timeseries.astype(float)
    return maybe_transpose(timeseries)

def apply_filter_on_channel(b, a, data):
    return filtfilt(b, a, data)

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
    results = Parallel(n_jobs=-1)(
        delayed(apply_filter_on_channel)(b, a, data[i,])
        for i in range(num_channels)
    )
    
    return np.array(results)



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
    
    # Initialize power_spectra, phase_spectra, and timestamps with NaNs
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


# Define paths and filenames
data_path = "data"
fname = "Kuro_Hitch_ACC_BLA_dmPFC_10012018-acc.mat"
output_fname = "windowed_fft_results.npz"

# Check if timeseries variable is already defined
if 'timeseries' not in locals():
    # Load data
    file_path = os.path.join(data_path, fname)
    timeseries = mat_to_timeseries(load_mat(file_path))

# Filter data
cutoff_freq = 1000  # Hz
sampling_rate = 4e4
timeseries_lowpass = butter_lowpass_filter(timeseries, cutoff_freq, sampling_rate)

# Perform windowed FFT in parallel with progress bar
freqs, power_spectra, phase_spectra, window_params, timestamps = windowed_fft_parallel(timeseries_lowpass, sampling_rate)

# Save results to NPZ file
output_path = os.path.join(data_path, output_fname)
np.savez(output_path,
         freqs=freqs,
         power_spectra=power_spectra,
         phase_spectra=phase_spectra,
         window_params=window_params,
         timestamps=timestamps)

print("Done")

# t_end = 2**20
# x = timeseries[0,:t_end]

# J = 8
# T = t_end
# Q = 16

# scattering = Scattering1D(J, T, Q)

# meta = scattering.meta()
# order0 = np.where(meta['order'] == 0)
# order1 = np.where(meta['order'] == 1)
# order2 = np.where(meta['order'] == 2)

# Sx = scattering(x)

# plt.figure(figsize=(8, 8))
# plt.subplot(3, 1, 1)
# plt.plot(Sx[order0][0])
# plt.title('Zeroth-order scattering')
# plt.subplot(3, 1, 2)
# plt.imshow(Sx[order1], aspect='auto')
# plt.title('First-order scattering')
# plt.subplot(3, 1, 3)
# plt.imshow(Sx[order2], aspect='auto')
# plt.title('Second-order scattering')
# plt.tight_layout()
# plt.show()
