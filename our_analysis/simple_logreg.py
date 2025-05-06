"""Logistic regression analysis (one trial back) using OLS"""

import os
from os.path import join
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from scipy import stats
import pickle

# Directory setup
ANALYSIS_DIR = os.path.dirname(os.path.realpath(__file__))
ANALYSIS_RESULTS_DIR = os.path.join(ANALYSIS_DIR, 'analysis_results')

def topm1(v):
    """Convert 0/1 to -1/1"""
    return 2 * v - 1

def get_regression_row(reward, common):
    """Create a row for the design matrix"""
    return np.array([1, reward, common, reward * common])

def prepare_participant_data(df):
    """Prepare data for regression at the participant level
    
    This function organizes the data by participant, creating for each trial:
    - x: design matrix with intercept, reward (-1/1), common (-1/1), and reward*common interaction
    - y: whether the choice in the next trial is the same as in the current trial (0/1)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the experiment data
        
    Returns:
    --------
    tuple
        (x_all, y_all, participant_ids) where:
        - x_all is a list of numpy arrays, each array containing the design matrix for a participant
        - y_all is a list of numpy arrays, each array containing the dependent variable for a participant
        - participant_ids is a list of participant identifiers
    """
    x_all, y_all, participant_ids = [], [], []
    
    for participant, part_data in df.groupby("participant"):
        part_x, part_y = [], []
        
        for prev_row, next_row in zip(part_data[:-1].itertuples(), part_data[1:].itertuples()):
            reward = topm1(prev_row.reward)
            common = topm1(prev_row.common)
            x_row = get_regression_row(reward, common)
            part_x.append(x_row)
            part_y.append(int(prev_row.choice1 == next_row.choice1))
        
        if part_x:  # Only include participants with data
            x_all.append(np.array(part_x))
            y_all.append(np.array(part_y))
            participant_ids.append(participant)
    
    return x_all, y_all, participant_ids

def run_participant_level_logit(x, y, max_value=20):
    """Run logistic regression for a single participant with safeguards against numerical issues
    
    Parameters:
    -----------
    x : numpy.ndarray
        Design matrix with shape (n_trials, n_predictors)
    y : numpy.ndarray
        Binary outcome variable with shape (n_trials,)
    max_value : float
        Maximum absolute value allowed for coefficients (to prevent numerical overflow)
        
    Returns:
    --------
    statsmodels.discrete.discrete_model.BinaryResults or None
        Result of the logistic regression, or None if fitting failed
    """
    try:
        # Check for basic data issues
        if len(np.unique(y)) < 2:
            print("Warning: All responses are the same, cannot fit logistic regression")
            return None
            
        # Check for perfect separation
        for col in range(x.shape[1]):
            if col > 0:  # Skip intercept column
                col_values = np.unique(x[:, col])
                for val in col_values:
                    y_for_val = y[x[:, col] == val]
                    if len(np.unique(y_for_val)) < 2 and len(y_for_val) > 1:
                        print(f"Warning: Potential separation in predictor column {col}")
        
        # Create logit model
        model = Logit(y, x)
        
        # Try different optimization methods with regularization
        optimization_methods = ['newton', 'bfgs', 'lbfgs', 'powell', 'cg', 'ncg']
        alphas = [0, 0.01, 0.1, 0.5]  # Start with no regularization, then increase
        
        result = None
        best_result = None
        min_extreme_value = float('inf')
        
        # First try standard methods with different regularization strengths
        for method in optimization_methods:
            for alpha in alphas:
                try:
                    if alpha > 0:
                        # L2 regularization (Ridge)
                        this_result = model.fit_regularized(
                            method=method, 
                            alpha=alpha, 
                            maxiter=200, 
                            disp=0, 
                            warn_convergence=False
                        )
                    else:
                        # No regularization
                        this_result = model.fit(
                            method=method, 
                            maxiter=200, 
                            disp=0, 
                            warn_convergence=False
                        )
                    
                    # Check for extreme coefficient values
                    max_abs_coef = np.max(np.abs(this_result.params))
                    
                    # Keep track of the result with the least extreme coefficients
                    if max_abs_coef < min_extreme_value:
                        min_extreme_value = max_abs_coef
                        best_result = this_result
                    
                    # If we found a good result, we can stop here
                    if max_abs_coef < max_value:
                        result = this_result
                        break
                        
                except Exception as e:
                    continue
            
            if result is not None:
                break
        
        # If all methods failed or produced extreme values, use the least extreme one
        if result is None:
            if best_result is not None:
                print(f"Warning: Using result with extreme coefficient values ({min_extreme_value:.2e})")
                result = best_result
            else:
                print("All fitting methods failed for this participant")
                return None
        
        return result
    except Exception as e:
        print(f"Error in logistic regression: {e}")
        return None

def run_logistic_regression(df, condition_name, output_dir=None):
    """Run logistic regression for participants in a single condition dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data for a single condition
    condition_name : str
        Name of the condition, used for naming output files
    output_dir : str, optional
        Directory where to save the results
        
    Returns:
    --------
    tuple
        (model_results, summary_df, coefficients_df)
    """
    # Setup directories
    if output_dir is None:
        output_dir = os.path.join(ANALYSIS_DIR, 'analysis_results', condition_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Prepare data for all participants
    x_all, y_all, participant_ids = prepare_participant_data(df)
    
    if not participant_ids:
        print(f"No valid participants found in the dataset for condition: {condition_name}")
        return None, None, None
    
    # Run regression for each participant
    results = {}
    coefficients = []
    
    for i, (x, y, participant_id) in enumerate(zip(x_all, y_all, participant_ids)):
        result = run_participant_level_logit(x, y)
        
        if result is not None:
            results[participant_id] = result
            coefficients.append(result.params)
    
    # Save results
    results_file = join(output_dir, f'logit_results_{condition_name}.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Create a DataFrame with coefficients for all participants
    coef_df = pd.DataFrame(coefficients, index=list(results.keys()))
    coef_df.columns = ['Intercept', 'Reward', 'Common', 'Reward*Common']
    
    # Calculate mean and standard error
    mean_coef = coef_df.mean()
    se_coef = coef_df.sem()
    
    # Using scipy.stats.norm.cdf for p-value calculation instead of statsmodels
    from scipy import stats
    
    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'Mean': mean_coef,
        'SE': se_coef,
        't': mean_coef / se_coef,
        'p': 2 * (1 - stats.norm.cdf(abs(mean_coef / se_coef)))
    })
    
    # Save summary
    summary_file = join(output_dir, f'logit_summary_{condition_name}.csv')
    summary_df.to_csv(summary_file)
    
    # Save all coefficients
    coef_file = join(output_dir, f'logit_coefficients_{condition_name}.csv')
    coef_df.to_csv(coef_file)
    
    # Print summary
    print(f"\nSummary for {condition_name} condition:")
    print(summary_df)
    print("\n")
    
    return results, summary_df, coef_df

def compare_conditions(abstract_results, story_results, output_dir=None):
    """Compare coefficients between abstract and story conditions
    
    Parameters:
    -----------
    abstract_results : tuple
        Results from run_logistic_regression for abstract condition (model_results, summary_df, coef_df)
    story_results : tuple
        Results from run_logistic_regression for story condition (model_results, summary_df, coef_df)
    output_dir : str, optional
        Directory where to save the comparison results
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the comparison statistics
    """
    if output_dir is None:
        output_dir = os.path.join(ANALYSIS_DIR, 'analysis_results', 'comparison')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract coefficient DataFrames
    _, _, abstract_coef = abstract_results
    _, _, story_coef = story_results
    
    if abstract_coef is None or story_coef is None:
        print("Cannot compare conditions: One or both results are missing")
        return None
    
    # Perform t-test for each coefficient
    from scipy import stats
    comparison = {}
    
    for col in abstract_coef.columns:
        t_stat, p_val = stats.ttest_ind(abstract_coef[col].dropna(), story_coef[col].dropna(), equal_var=False)
        comparison[col] = {
            'Abstract_mean': abstract_coef[col].mean(),
            'Story_mean': story_coef[col].mean(),
            'Difference': story_coef[col].mean() - abstract_coef[col].mean(),
            't_statistic': t_stat,
            'p_value': p_val
        }
    
    comparison_df = pd.DataFrame(comparison).T
    
    # Save comparison
    comparison_file = join(output_dir, 'condition_comparison.csv')
    comparison_df.to_csv(comparison_file)
    
    # Print comparison
    print("\nCondition Comparison Results:")
    print(comparison_df)
    print("\n")
    
    return comparison_df

# Example usage:
'''
# For story condition
story_results = run_logistic_regression(
    df=story_trials,
    condition_name='story',
    output_dir='story_logreg_results'
)

# For abstract condition
abstract_results = run_logistic_regression(
    df=abstract_trials,
    condition_name='abstract',
    output_dir='abstract_logreg_results'
)

# Compare conditions
comparison = compare_conditions(abstract_results, story_results, output_dir='comparison_results')
'''