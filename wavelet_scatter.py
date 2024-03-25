#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:48:22 2024

@author: prabaha
"""

from kymatio.numpy import Scattering1D
import os
import h5py
import scipy
import numpy as np


def load_mat(file_path):
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
    timeseries = np.array(f['mat'])
    timeseries = timeseries.astype(float)
    return maybe_transpose(timeseries)



data_path = "/gpfs/milgram/project/chang/pg496/social_gaze_raw_mat"
data_path = "data"

fname = "Kuro_Hitch_ACC_BLA_dmPFC_10012018-acc.mat"

file_path = os.path.join(data_path, fname)

timeseries = mat_to_timeseries( load_mat( file_path ) )
t_end = 2**20
x = timeseries[0,:t_end]

J = 8
T = t_end
Q = 16
sampling_rate = 4e4

scattering = Scattering1D(J, T, Q)

meta = scattering.meta()
order0 = np.where(meta['order'] == 0)
order1 = np.where(meta['order'] == 1)
order2 = np.where(meta['order'] == 2)

Sx = scattering(x)

plt.figure(figsize=(8, 8))
plt.subplot(3, 1, 1)
plt.plot(Sx[order0][0])
plt.title('Zeroth-order scattering')
plt.subplot(3, 1, 2)
plt.imshow(Sx[order1], aspect='auto')
plt.title('First-order scattering')
plt.subplot(3, 1, 3)
plt.imshow(Sx[order2], aspect='auto')
plt.title('Second-order scattering')
plt.tight_layout()
plt.show()
