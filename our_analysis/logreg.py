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

def run_logistic_regression(csv_path):
    if not os.path.exists(ANALYSIS_RESULTS_DIR):
        os.mkdir(ANALYSIS_RESULTS_DIR)
    fit = get_exp_fit(csv_path)
    fit.summary().to_csv(SAMPLESFIT_FN)
    return fit