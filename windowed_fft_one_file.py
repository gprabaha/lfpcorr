#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:48:22 2024

@author: prabaha
"""

# Importing necessary libraries
import os
import sys
import numpy as np

from util import *


# Main function
def main():
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python windowed_fft_one_file.py input_file_path")
        return
    # Set cutoff frequency and sampling rate
    cutoff_freq = 1000  # Hz
    sampling_rate = 4e4 # Hz
    # Access the value of input_path (the first command-line argument)
    input_file = sys.argv[1]
    # Load MAT file and convert to numpy array
    timeseries = mat_to_timeseries(load_mat(input_file))
    # Apply Butterworth lowpass filter
    timeseries_lowpass = butter_lowpass_filter(timeseries, cutoff_freq, sampling_rate)
    
    # Perform windowed FFT
    freqs, power_spectra, phase_spectra, window_params, timestamps = windowed_fft_parallel(timeseries_lowpass, sampling_rate)
    
    # Define output directory
    out_dir = "data/windowed_fft_kaiser"
    if not os.path.exists(out_dir):
        # If the directory doesn't exist, create it
        os.makedirs(out_dir)
    filename_without_extension = os.path.splitext(os.path.basename(input_file))[0]
    output_fname = filename_without_extension + '_windowed_fft_kaiser.npz'
    
    # Save results to NPZ file
    output_path = os.path.join(out_dir, output_fname)
    np.savez(output_path,
             freqs=freqs,
             power_spectra=power_spectra,
             phase_spectra=phase_spectra,
             window_params=window_params,
             timestamps=timestamps)
    print("Done")



if __name__ == "__main__":
    main()
