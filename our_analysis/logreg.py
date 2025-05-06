#This code is modifying the orignal code from:
# Copyright (C) 2016, 2017, 2018, 2023 Carolina Feher da Silva


#My main mods are:
#Changing the code so it executes as a function from another script
#Removing config settings and manually setting this
#Adding the ability to adjust the csv input dynamically from another script


"""Logistic regression analysis (one trial back)"""

import os
from os.path import join
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel



# Conditions and directory setup
CONDITIONS = ('abstract', 'story')
ANALYSIS_DIR = os.path.dirname(os.path.realpath(__file__))
ANALYSIS_RESULTS_DIR = os.path.join(ANALYSIS_DIR, 'analysis_results')


def topm1(v):
    return 2 * v - 1

def get_regression_row(reward, common):
    return np.array([1, reward, common, reward * common])

def get_part_data(csv_path):
    x, y, condition = [], [], []
    df = pd.read_csv(csv_path)
    for _, part_data in df.groupby("participant"):
        part_condition = part_data.iloc[0].condition
        assert part_condition in CONDITIONS
        condition.append(int(part_condition == CONDITIONS[-1]))
        part_x, part_y = [], []
        for prev_row, next_row in zip(part_data[:-1].itertuples(), part_data[1:].itertuples()):
            reward = topm1(prev_row.reward)
            common = topm1(prev_row.common)
            x_row = get_regression_row(reward, common)
            part_x.append(x_row)
            part_y.append(int(prev_row.choice1 == next_row.choice1))
        x.append(part_x)
        y.append(part_y)

    num_trials = max(len(yy) for yy in y)
    xdummy = [0] * len(x[0][0])
    for xx, yy in zip(x, y):
        while len(yy) < num_trials:
            yy.append(0)
            xx.append(xdummy)
    return x, y, condition

def get_logreg_model_dat(csv_path):
    x, y, condition = get_part_data(csv_path)
    return {
        'M': len(y),
        'N': len(y[0]),
        'K': len(x[0][0]),
        'y': y,
        'x': x,
        'condition': condition,
    }

# Sampling settings
SAMPLE_FN = join(ANALYSIS_RESULTS_DIR, 'logreg_samples')
SAMPLESFIT_FN = join(ANALYSIS_RESULTS_DIR, 'logreg_analysis_fit.csv')
ITER = 32000
WARMUP = 16000
CHAINS = 4

def get_exp_fit(csv_path, its=ITER, chains=CHAINS, warmup=WARMUP):
    stan_model = CmdStanModel(stan_file=join(ANALYSIS_DIR, 'logreg_model.stan'))
    model_dat = get_logreg_model_dat(csv_path)
    fit = stan_model.sample(
        data=model_dat,
        iter_sampling=its,
        chains=chains,
        iter_warmup=warmup,
        output_dir=SAMPLE_FN,
        show_progress=True,
    )
    return fit

# def run_logistic_regression(csv_path):
#     if not os.path.exists(ANALYSIS_RESULTS_DIR):
#         os.mkdir(ANALYSIS_RESULTS_DIR)
#     fit = get_exp_fit(csv_path)
#     fit.summary().to_csv(SAMPLESFIT_FN)
#     return fit


def run_logistic_regression(df, condition_name, output_dir=None, model_file=None,
                           iter_sampling=32000, warmup=16000, chains=4):
   
    # Setup directories
    analysis_dir = os.path.dirname(os.path.realpath(__file__))
    if output_dir is None:
        output_dir = os.path.join(analysis_dir, 'analysis_results')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if model_file is None:
        model_file = os.path.join(analysis_dir, 'logreg_model.stan')
    
    if condition_name not in ('story', 'abstract'):
        raise ValueError("condition_name must be either 'story' or 'abstract'")
    
    # Load Stan model
    stan_model = CmdStanModel(stan_file=model_file)
    model_data = prepare_condition_data(df, condition_name)
    
    # Set up output files
    sample_fn = join(output_dir, f'logreg_samples_{condition_name}')
    fit_fn = join(output_dir, f'logreg_analysis_fit_{condition_name}.csv')
    
    # Run the model
    fit = stan_model.sample(
        data=model_data,
        iter_sampling=iter_sampling,
        chains=chains,
        iter_warmup=warmup,
        output_dir=sample_fn,
        show_progress=True,
    )
    
    fit.summary().to_csv(fit_fn)
    return fit
    
def prepare_condition_data(df, condition_name):

    conditions = ('abstract', 'story')
    x, y, condition = [], [], []
    
    # Group by participant and process each participant's data
    for _, part_data in df.groupby("participant"):
        part_x, part_y = [], []
        condition.append(int(condition_name == conditions[-1]))  # 0 for abstract, 1 for story
        
        for prev_row, next_row in zip(part_data[:-1].itertuples(), part_data[1:].itertuples()):
            reward = topm1(prev_row.reward)
            common = topm1(prev_row.common)
            x_row = get_regression_row(reward, common)
            part_x.append(x_row)
            part_y.append(int(prev_row.choice1 == next_row.choice1))
        
        x.append(part_x)
        y.append(part_y)
    
    num_trials = max(len(yy) for yy in y) if y else 0
    xdummy = [0] * 4  # 4 is the length of the x_row
    
    for xx, yy in zip(x, y):
        while len(yy) < num_trials:
            yy.append(0)
            xx.append(xdummy)
    
    # Prepare the data for Stan
    return {
        'M': len(y),
        'N': len(y[0]) if y else 0,
        'K': len(x[0][0]) if x and x[0] else 4,
        'y': y,
        'x': x,
        'condition': condition,
    }