import sys
sys.path.append('../')
import os
import numpy as np
from functools import partial

from fooof import FOOOF
from dask_jobqueue import SLURMCluster
import dask
from distributed import Client
from utils import get_aperiodic
import dill
num_cores = 128


data_path = '../../data/noise_EI_connprob'
x_sbi = np.load(f'{data_path}/sbi_sims/dpl_sbi.npy')
x_grid = np.load(f'{data_path}/sbi_sims/dpl_grid.npy')

with open(f'{data_path}/sbi_sims/sim_metadata.pkl', 'rb') as output_file:
    sim_metadata = dill.load(output_file)
    
dt = sim_metadata['dt'] # Sampling interval used for simulation
fs = (1/dt) * 1e3

 # Set up cluster and reserve resources
cluster = SLURMCluster(
    cores=32, processes=32, queue='shared', memory="256GB", walltime="00:30:00",
    job_extra_directives=['-A csd403', '--nodes=1'], log_directory=os.getcwd() + '/slurm_out')

client = Client(cluster)
client.upload_file('../utils.py')

client.cluster.scale(num_cores)
print(client.dashboard_link)

for x, fname in zip([x_sbi, x_grid], ['aperiodic_sbi.npy', 'aperiodic_grid.npy']):
    res_list = []
    for idx in range(x.shape[0]):
        data = x[idx,:]
        res = dask.delayed(get_aperiodic)(data, fs=fs)
        res_list.append(res)

    # Run tasks
    final_res = dask.compute(*res_list)
    final_res_array = np.array(final_res)
    np.save(f'{data_path}/sbi_sims/{fname}', final_res_array)

#os.system('scancel -u ntolley')
