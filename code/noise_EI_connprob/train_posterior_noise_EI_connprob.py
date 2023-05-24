import sys
sys.path.append('../')
import torch
import os
import numpy as np
from functools import partial

from sbi import inference as sbi_inference
from utils import train_posterior
import pickle
import dill

device = 'cpu'

ntrain_sims = 10_000

data_path = '../../data/noise_EI_connprob'

# Window specifying portion of time series for inference
x_noise_amp = 0.0
theta_noise_amp = 0.0
extra_dict = {'aperiodic_fname': f'{data_path}/sbi_sims/aperiodic_sbi.npy',
              'window_samples': [0, -1]}
train_posterior(data_path, ntrain_sims, x_noise_amp, theta_noise_amp, extra_dict=extra_dict)

   
#os.system('scancel -u ntolley')
