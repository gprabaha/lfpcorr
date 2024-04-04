#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:03:03 2024

@author: pg496
"""

import os
import subprocess
from util import *


parent_data_dir = "/gpfs/milgram/project/chang/pg496/data_dir/social_gaze"
pos_folder = os.path.join(parent_data_dir, 'social_gaze_eyetracking/aligned_raw_samples/position')
all_files = os.listdir(pos_folder)
mat_files_with_path = [os.path.join(pos_folder, file) for file in all_files if file.endswith('.mat')]

top_out_dir = "gaze_pred_gaze_lstm"

sorted_file_paths = sorted((f for f in mat_files_with_path if file_sorting_key(f) != (None, None)), key=file_sorting_key)

with open('lstm_dsq_job_list.txt', 'w') as file:
    for path in sorted_file_paths:
        line_entry = "module load miniconda CUDA/11.1.1-GCC-10.2.0; conda activate nn_gpu; python fit_lstm_to_one_gaze_pos_file.py {}\n"
        file.write(line_entry.format(path))
