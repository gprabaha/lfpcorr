#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 17:44:17 2024

@author: pg496
"""

import numpy
import h5py
import os
from scipy.io import loadmat

data_path = "/gpfs/milgram/project/chang/pg496/social_gaze_raw_mat"

fname = "Lynch_Cronenberg_ACCg_BLA_ACCg_01142019-acc.mat"


with h5py.File()
file_path = os.path.join(data_path, fname)

data = loadmat(file_path)