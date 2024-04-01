#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:02:25 2024

@author: prabaha
"""


import os
import subprocess
import shlex

data_path = "/gpfs/milgram/project/chang/pg496/data_dir/social_gaze/social_gaze_raw_mat"

all_files = os.listdir(data_path)
mat_files_with_path = [os.path.join(data_path, file) for file in all_files if file.endswith('.mat')]

job_script_dir = "job_scripts"
template_file = "template_lfp.txt"

job_line_template = "module load miniconda; conda activate lfp_cluster; python extract_lfp_one_file.py {}"

output_fname = "dsq_job_list.txt"
output_file = os.path.join(job_script_dir, output_fname)
with open(output_file, 'w') as f:
    for path in mat_files_with_path:
        f_name = os.path.splitext(os.path.basename(path))[0]
        job_line = job_line_template.format(path)
        f.write(job_line)

command = "module load dSD; dsq --job-file " + output_file + " --partition psych_day --cpus-per-task 6 --mem 200G -t 2:00:00"
command = shlex.split(command)
print(command)
subprocess.run(command.split())