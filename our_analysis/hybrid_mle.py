# Copyright (C) 2016, 2017, 2018, 2023 Carolina Feher da Silva

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Fit the hybrid reinforcement learning model using maximum-likelihood estimation"""

import sys
import os
from os.path import join, exists
from os import mkdir
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

# Parameter names
PARAM_NAMES = ('alpha1', 'alpha2', 'lmbd', 'beta1', 'beta2', 'p', 'w')
NOPTIM = 1000

def optimize_model(stan_model, model_dat, noptim=NOPTIM):
    """Optimize the model using maximum-likelihood estimation."""
    log_lik = -np.inf
    params = None
    for _ in range(noptim):
        while True:
            try:
                op = stan_model.optimize(data=model_dat)
            except RuntimeError as rterror:
                sys.stderr.write(f"Error: {str(rterror)}\n")
            else:
                if op.converged:
                    break
        if op.optimized_params_dict["lp__"] > log_lik:
            log_lik = op.optimized_params_dict["lp__"]
            params = op.optimized_params_dict
    return params

def fit_hybrid_mixed_model(data_df, stan_file="hybrid_mixed.stan", output_file=None, noptim=NOPTIM):
    
    if output_file and not exists(os.path.dirname(output_file)) and os.path.dirname(output_file):
        mkdir(os.path.dirname(output_file))
    
    ntrials = data_df.trial.max() + 1
    
    model_dat = {
        "N": 0,
        "num_trials": [],
        "action1": [],
        "action2": [],
        "s2": [],
        "reward": [],
    }
    model_dat["maxtrials"] = ntrials
    
    participants = []
    conditions = []
    
    for _, part_data in data_df.groupby("participant"):
        model_dat["N"] += 1
        participants.append(part_data.iloc[0].participant)
        conditions.append(part_data.iloc[0].condition)
        
        action1 = list(part_data.choice1)
        action2 = list(part_data.choice2)
        s2 = list(part_data.final_state)
        reward = list(part_data.reward)
        
        for lst in (action1, action2, s2, reward):
            lst += [1] * (ntrials - len(part_data))
            assert len(lst) == ntrials
            
        model_dat["num_trials"].append(len(part_data))
        model_dat["action1"].append(action1)
        model_dat["action2"].append(action2)
        model_dat["s2"].append(s2)
        model_dat["reward"].append(reward)
    
    # Fit the mixed-effects model
    stan_model = CmdStanModel(stan_file=stan_file)
    
    # Optimize the model
    params = optimize_model(stan_model, model_dat, noptim)
    
    results = []
    for part_num in range(len(participants)):
        results.append({
            'participant': participants[part_num],
            'condition': conditions[part_num],
            'alpha1': params['alpha1'],
            'alpha2': params['alpha2'],
            'lmbd': params['lmbd'],
            'beta1': params['beta1'],
            'beta2': params['beta2'],
            'p': params['p'],
            'w': params[f"w[{part_num + 1}]"],
        })
    
    results_df = pd.DataFrame(results)
    
    if output_file:
        results_df.to_csv(output_file, index=False)
    
    return results_df, params


def fit_hybrid_mixed_dynamic_model(data_df, stan_file="dynamic_hybrid_mixed.stan", output_file=None, noptim=NOPTIM, fixed_params=None, return_logli=False):
    
    if output_file and not exists(os.path.dirname(output_file)) and os.path.dirname(output_file):
        mkdir(os.path.dirname(output_file))
    
    ntrials = data_df.trial.max() + 1
    
    model_dat = {
        "N": 0,
        "num_trials": [],
        "action1": [],
        "action2": [],
        "s2": [],
        "reward": [],
    }
    model_dat["maxtrials"] = ntrials

    #Here we're going to figure out which fixed params were passed and add them. We'll use a flag for this. 

    #These are placeholder values because the stan code will break if we don't feed it initilization values. These will be ignored if fix flag is 0
    model_dat["alpha1"] = 0.5  
    model_dat["alpha2"] = 0.5
    model_dat["lmbd"] = 0.5
    model_dat["beta1"] = 1.0
    model_dat["beta2"] = 1.0
    model_dat["p"] = 0.5
    model_dat["w"] = 0.5

    if fixed_params:
        for param, value in fixed_params.items():
            if param in PARAM_NAMES: #this is only going to work for valid params in our model
                model_dat[f"fix_{param}"] = 1 # flag for the stan file
                model_dat[param] = value
            else:
                None

    for param in PARAM_NAMES:
        if fixed_params is None or param not in fixed_params:
            model_dat[f"fix_{param}"] = 0

    
    participants = []
    conditions = []
    
    for _, part_data in data_df.groupby("participant"):
        model_dat["N"] += 1
        participants.append(part_data.iloc[0].participant)
        conditions.append(part_data.iloc[0].condition)
        
        action1 = list(part_data.choice1)
        action2 = list(part_data.choice2)
        s2 = list(part_data.final_state)
        reward = list(part_data.reward)
        
        for lst in (action1, action2, s2, reward):
            lst += [1] * (ntrials - len(part_data))
            assert len(lst) == ntrials
            
        model_dat["num_trials"].append(len(part_data))
        model_dat["action1"].append(action1)
        model_dat["action2"].append(action2)
        model_dat["s2"].append(s2)
        model_dat["reward"].append(reward)
    
    # Fit the mixed-effects model
    stan_model = CmdStanModel(stan_file=stan_file)
    
    # Optimize the model
    params = optimize_model(stan_model, model_dat, noptim)
    print("Optimized parameters:")
    print(params)

    param_mapping = {
    'alpha1_local': 'alpha1',
    'alpha2_local': 'alpha2',
    'lmbd_local': 'lmbd',
    'beta1_local': 'beta1',
    'beta2_local': 'beta2',
    'p_local': 'p'
}

    # edit the param names becuase something is off in the stan
    transformed_params = {}
    for stan_name, our_name in param_mapping.items():
        if stan_name in params:
            transformed_params[our_name] = params[stan_name]

    ## try using these new
    params = {**params, **transformed_params}
    
    logli = params['lp__']


    results = []
    for part_num in range(len(participants)):
        participant_result = {
            'participant': participants[part_num],
            'condition': conditions[part_num],
        }
        
        for param in PARAM_NAMES:
            if param =='w':
                participant_result[param] = params[f"w_local[{part_num+1}]"]
            else:
                if fixed_params and param in fixed_params:
                    participant_result[param] = fixed_params[param]
                else:
                    participant_result[param] = params.get(param, float('nan'))
        results.append(participant_result)

    
    
    results_df = pd.DataFrame(results)
    
    if output_file:
        results_df.to_csv(output_file, index=False)
    
    if return_logli:
        return results_df, params, logli
    else:
        return results_df, params