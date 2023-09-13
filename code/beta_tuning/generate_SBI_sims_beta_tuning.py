import sys
sys.path.append('../')
import os
import dill
from utils import (linear_scale_forward, log_scale_forward, start_cluster,
                   run_hnn_sim, beta_tuning_param_function, UniformPrior)
from hnn_core import calcium_model, pick_connection
import numpy as np

nsbi_sims = 10_000
tstop = 1000
dt = 0.05

net = calcium_model()

# Extract all E-I connection types
E_gids = np.concatenate([net.gid_ranges['L2_pyramidal'], net.gid_ranges['L5_pyramidal']]).tolist()
I_gids = np.concatenate([net.gid_ranges['L2_basket'], net.gid_ranges['L5_basket']]).tolist()

EI_connections = pick_connection(net, src_gids=E_gids, target_gids=I_gids)
EE_connections = pick_connection(net, src_gids=E_gids, target_gids=E_gids)
II_connections = pick_connection(net, src_gids=I_gids, target_gids=I_gids)
IE_connections = pick_connection(net, src_gids=I_gids, target_gids=E_gids)

# Store in dictionary to be added to theta_dict
theta_extra = {'EI_connections': EI_connections, 'EE_connections': EE_connections, 
               'II_connections': II_connections, 'IE_connections': IE_connections,
               'lamtha': 4.0}
   
save_path = '/expanse/lustre/scratch/ntolley/temp_project/beta_tuning'
# save_path = '../../data/beta_tuning'
save_suffix = 'sbi'

prior_dict = {'EI_gscale': {'bounds': (-2, 2), 'rescale_function': log_scale_forward},
              'EE_gscale': {'bounds': (-2, 2), 'rescale_function': log_scale_forward},
              'II_gscale': {'bounds': (-2, 2), 'rescale_function': log_scale_forward},
              'IE_gscale': {'bounds': (-2, 2), 'rescale_function': log_scale_forward},
              'EI_prob': {'bounds': (0, 1), 'rescale_function': linear_scale_forward},
              'EE_prob': {'bounds': (0, 1), 'rescale_function': linear_scale_forward},
              'II_prob': {'bounds': (0, 1), 'rescale_function': linear_scale_forward},
              'IE_prob': {'bounds': (0, 1), 'rescale_function': linear_scale_forward},
              'L2e_distal': {'bounds': (-4, 0), 'rescale_function': log_scale_forward},
              'L2i_distal': {'bounds': (-4, 0), 'rescale_function': log_scale_forward},
              'L5e_distal': {'bounds': (-4, 0), 'rescale_function': log_scale_forward},
              'L5i_distal': {'bounds': (-4, 0), 'rescale_function': log_scale_forward},
              'L2e_proximal': {'bounds': (-4, 0), 'rescale_function': log_scale_forward},
              'L2i_proximal': {'bounds': (-4, 0), 'rescale_function': log_scale_forward},
              'L5e_proximal': {'bounds': (-4, 0), 'rescale_function': log_scale_forward},
              'L5i_proximal': {'bounds': (-4, 0), 'rescale_function': log_scale_forward},
              }

with open(f'{save_path}/sbi_sims/prior_dict.pkl', 'wb') as f:
    dill.dump(prior_dict, f)

sim_metadata = {'nsbi_sims': nsbi_sims, 'tstop': tstop, 'dt': dt, 'gid_ranges': net.gid_ranges, 'theta_extra': theta_extra}
with open(f'{save_path}/sbi_sims/sim_metadata.pkl', 'wb') as f:
    dill.dump(sim_metadata, f)

prior = UniformPrior(parameters=list(prior_dict.keys()))
theta_samples = prior.sample((nsbi_sims,))

start_cluster() # reserve resources for HNN simulations

run_hnn_sim(net=net, param_function=beta_tuning_param_function, prior_dict=prior_dict,
            theta_samples=theta_samples, tstop=tstop, save_path=save_path, save_suffix=save_suffix,
            theta_extra=theta_extra)

#os.system('scancel -u ntolley')