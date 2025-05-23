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

"""Analysis using the hybrid reinforcement learning model and Stan"""

import sys
from os.path import join, exists
from os import mkdir
import pandas as pd
from cmdstanpy import CmdStanModel
from config import (
    ANALYSIS_RESULTS_DIR,
    CONDITIONS,
)

model = "hybrid_hier"


def main():
    stan_model = CmdStanModel(stan_file=f"{model}.stan")
    if not exists(ANALYSIS_RESULTS_DIR):
        mkdir(ANALYSIS_RESULTS_DIR)
    model_dat = {
        "N": 0,
        "num_trials": [],
        "action1": [],
        "action2": [],
        "s2": [],
        "reward": [],
        "condition": [],
    }
    game_results = pd.read_csv("beh_noslow.csv")
    NTRIALS = game_results.trial.max() + 1
    model_dat["maxtrials"] = NTRIALS
    for part, part_data in game_results.groupby("participant"):
        model_dat["N"] += 1
        action1, action2, s2, reward = [], [], [], []
        for trial in part_data.itertuples():
            action1.append(trial.choice1)
            action2.append(trial.choice2)
            s2.append(trial.final_state)
            reward.append(trial.reward)
        for lst in (action1, action2, s2, reward):
            lst += [1] * (NTRIALS - len(part_data))
            assert len(lst) == NTRIALS
        model_dat["num_trials"].append(len(part_data))
        model_dat["action1"].append(action1)
        model_dat["action2"].append(action2)
        model_dat["s2"].append(s2)
        model_dat["reward"].append(reward)
        model_dat["condition"].append(int(part_data.iloc[0].condition == CONDITIONS[1]))
    fit = stan_model.sample(
        data=model_dat,
        iter_sampling=20_000,
        chains=6,
        iter_warmup=80_000,
        output_dir=join(ANALYSIS_RESULTS_DIR, f"fit_{model}_output"),
        refresh=1,
        max_treedepth=20,
        show_progress=True,
        adapt_delta=0.95,
    )
    print(fit.diagnose())
    fit.summary().to_csv(join(ANALYSIS_RESULTS_DIR, f"{model}_fit.csv"))


def run_bayesian_model(data_file="beh_noslow.csv", stan_file="hybrid_hier.stan", 
                       output_dir=ANALYSIS_RESULTS_DIR, model_name="hybrid_hier",
                       iter_sampling=20_000, chains=6, iter_warmup=80_000,
                       max_treedepth=20, adapt_delta=0.95):
    """
    Run Bayesian fitting with Stan for the hybrid reinforcement learning model.
    
    Parameters:
    -----------
    data_file : str
        Path to the CSV file containing behavioral data
    stan_file : str
        Path to the Stan model file
    output_dir : str
        Directory to save results
    model_name : str
        Name of the model for output files
    iter_sampling : int
        Number of sampling iterations
    chains : int
        Number of MCMC chains
    iter_warmup : int
        Number of warmup iterations
    max_treedepth : int
        Maximum tree depth for NUTS sampler
    adapt_delta : float
        Target acceptance rate
        
    Returns:
    --------
    fit : CmdStanMCMC
        The fitted Stan model
    """
    # Create output directory if it doesn't exist
    if not exists(output_dir):
        mkdir(output_dir)
    
    # Initialize Stan model
    stan_model = CmdStanModel(stan_file=stan_file)
    
    # Read data
    game_results = pd.read_csv(data_file)
    NTRIALS = game_results.trial.max() + 1
    
    # Prepare data for Stan
    model_dat = {
        "N": 0,
        "num_trials": [],
        "action1": [],
        "action2": [],
        "s2": [],
        "reward": [],
        "condition": [],
    }
    model_dat["maxtrials"] = NTRIALS
    
    # Process participant data
    for part, part_data in game_results.groupby("participant"):
        model_dat["N"] += 1
        action1, action2, s2, reward = [], [], [], []
        
        for trial in part_data.itertuples():
            action1.append(trial.choice1)
            action2.append(trial.choice2)
            s2.append(trial.final_state)
            reward.append(trial.reward)
            
        # Pad data to match maximum number of trials
        for lst in (action1, action2, s2, reward):
            lst += [1] * (NTRIALS - len(part_data))
            assert len(lst) == NTRIALS
            
        model_dat["num_trials"].append(len(part_data))
        model_dat["action1"].append(action1)
        model_dat["action2"].append(action2)
        model_dat["s2"].append(s2)
        model_dat["reward"].append(reward)
        
        # Convert condition to binary format
        model_dat["condition"].append(int(part_data.iloc[0].condition == CONDITIONS[1]))
    
    # Run MCMC sampling
    print(f"Starting Bayesian sampling with {chains} chains, {iter_sampling} samples per chain")
    fit = stan_model.sample(
        data=model_dat,
        iter_sampling=iter_sampling,
        chains=chains,
        iter_warmup=iter_warmup,
        output_dir=join(output_dir, f"fit_{model_name}_output"),
        refresh=1,
        max_treedepth=max_treedepth,
        show_progress=True,
        adapt_delta=adapt_delta,
    )
    
    # Check diagnostics
    print("Checking MCMC diagnostics...")
    diagnostics = fit.diagnose()
    print(diagnostics)
    
    # Save summary to CSV
    summary_file = join(output_dir, f"{model_name}_fit.csv")
    fit.summary().to_csv(summary_file)
    print(f"Summary statistics saved to {summary_file}")
    
    return fit


# if __name__ == "__main__":
#     main()
