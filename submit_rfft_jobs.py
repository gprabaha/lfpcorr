#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:08:13 2024

@author: prabaha
"""

import os
import subprocess

data_path = "/gpfs/milgram/project/chang/pg496/social_gaze_raw_mat"

all_files = os.listdir(data_path)
mat_files_with_path = [os.path.join(data_path, file) for file in all_files if file.endswith('.mat')]

job_script_dir = "job_scripts"
template_file = "template_rfft.txt"

for path in mat_files_with_path:
    f_name = os.path.splitext(os.path.basename(path))[0]
    s_name = ['submit_' + f_name + '.sh']
    with open(template_file, 'r') as f:
        template_script = f.read()
    modified_script = template_script.format(f_name, f_name, f_name, path)
    output_file = os.path.join(job_script_dir, s_name)
    with open(output_file, 'w') as f:
        f.write(modified_script)
    command = ['sbatch ' + output_file]
    subprocess.run(command)
