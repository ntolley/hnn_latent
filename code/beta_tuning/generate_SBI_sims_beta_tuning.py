import sys
sys.path.append('../')
import os
import dill
from utils import (linear_scale_forward, log_scale_forward, start_cluster,
                   run_hnn_sim, beta_tuning_param_function, UniformPrior,
                   PriorBetaFiltered)
from hnn_core import calcium_model, pick_connection
from neurodsp.spectral import compute_spectrum, trim_spectrum
from torch import optim
from tqdm import tqdm
import joblib
import numpy as np

nsbi_sims = 10_000
tstop = 1000
num_prior_fits = 4
relative_beta_threshold = 0.3
dt = 0.05
fs = 1000 / dt

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
              'L2e_proximal': {'bounds': (-4, 0), 'rescale_function': log_scale_forward},
              'L2i_proximal': {'bounds': (-4, 0), 'rescale_function': log_scale_forward},
              'L5e_proximal': {'bounds': (-4, 0), 'rescale_function': log_scale_forward},
              'L5i_proximal': {'bounds': (-4, 0), 'rescale_function': log_scale_forward},
              }

with open(f'{save_path}/sbi_sims/prior_dict.pkl', 'wb') as f:
    dill.dump(prior_dict, f)

sim_metadata = {'nsbi_sims': nsbi_sims, 'tstop': tstop, 'dt': dt, 'gid_ranges': net.gid_ranges,
                'num_prior_fits': num_prior_fits, 'relative_beta_threshold': relative_beta_threshold,
                'theta_extra': theta_extra}
with open(f'{save_path}/sbi_sims/sim_metadata.pkl', 'wb') as f:
    dill.dump(sim_metadata, f)

prior = UniformPrior(parameters=list(prior_dict.keys()))
theta_samples = prior.sample((nsbi_sims,))

start_cluster() # reserve resources for HNN simulations

for flow_idx in range(num_prior_fits):
    if flow_idx == 0:
        prior = UniformPrior(parameters=list(prior_dict.keys()))
    else:
        prior = prior_filtered
        
    theta_samples = prior.sample((nsbi_sims,))
    
    save_suffix = f'sbi_{flow_idx}'
    
    run_hnn_sim(net=net, param_function=beta_tuning_param_function, prior_dict=prior_dict,
                theta_samples=theta_samples, tstop=tstop, save_path=save_path, save_suffix=save_suffix,
                theta_extra=theta_extra)
    
    dpl_filter = np.load(f'{save_path}/sbi_sims/dpl_sbi_{flow_idx}.npy')
    theta_filter = np.load(f'{save_path}/sbi_sims/theta_sbi_{flow_idx}.npy')
    
    times = np.linspace(0, tstop / 1000, dpl_filter.shape[1])
    times_mask = times > 0.2 # burn in time
    
    freqs, powers = compute_spectrum(dpl_filter[:, times_mask], fs, method='welch', avg_type='median')
    freqs, powers = trim_spectrum(freqs, powers, [0, 100])
    
    beta_mask = (freqs > 13) & (freqs < 30)
    beta_power = powers[:, beta_mask].sum(axis=1)
    relative_beta = beta_power / powers.sum(axis=1)
    
    relative_beta_mask = relative_beta > relative_beta_threshold
    
    prior_filtered = PriorBetaFiltered(parameters=list(prior_dict.keys()))
    optimizer = optim.Adam(prior_filtered.flow.parameters())

    num_iter = 5000
    for i in tqdm(range(num_iter)):
        optimizer.zero_grad()
        loss = -prior_filtered.flow.log_prob(inputs=theta_filter).mean()
        loss.backward()
        optimizer.step()
    state_dict = prior_filtered.flow.state_dict()
    joblib.dump(state_dict, f'{save_path}/flows/prior_filtered_flow_{flow_idx}.pkl')


# os.system('scancel -u ntolley')