#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:46:07 2024

@author: prabaha
"""

import sys
import os
import numpy as np

from util import *


# Main function
def main():
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python extract_lfp_one_file.py input_file_path")
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
    
    # Define output directory
    parent_dir = os.path.dirname(os.path.dirname(input_file))
    out_dir_name = "social_gaze_lfp"
    out_dir = os.path.join(parent_dir, out_dir_name)
    if not os.path.exists(out_dir):
        # If the directory doesn't exist, create it
        os.makedirs(out_dir)
    filename_without_extension = os.path.splitext(os.path.basename(input_file))[0]
    output_fname = filename_without_extension + '_lfp.npz'
    
    # Save results to NPZ file
    output_path = os.path.join(out_dir, output_fname)
    np.savez(output_path, lfp=timeseries_lowpass)
    print("Done")



if __name__ == "__main__":
    main()