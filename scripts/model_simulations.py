# script for running model simulations 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.insert(0,'/Users/Kathy/Desktop/UW/seaflow/decomposition_project/scripts/')
from diel_tools import *
from tsd_functions import *
from rate_functions import *
from simulation_tools import *

## helper function to convert results to a dataframe
def to_df(dist,cols, error):
    df=pd.DataFrame(dist)
    df.columns=cols
    return(df.melt(var_name='removed', value_name=error))

## helper function to plot boxplot of errors
def error_boxplot(df, error, model):
    fig,axs=plt.subplots(figsize=(10,6))
    sns.boxplot(data=df, x='removed',y=error, palette=sns.color_palette("viridis"))
    axs.set_xlabel('Proportion of Data removed (%)')
    axs.set_ylabel(error)
    axs.set_title(f'%Data Removed vs. Error: {model}')
    plt.show()
    
## helper function to plot grouped boxplot of errors
def grouped_error_boxplot(df, error, model):
    fig,axs=plt.subplots(figsize=(10,6))
    sns.boxplot(x="remove", y=error,
        hue="block",
        data=df)
    if error.lower().startswith('r'):
        axs.set_ylim(0.001,0.0028)
    else:
        axs.set_ylim(1.0,2.5)
    axs.set_xlabel('Proportion of Data removed (%)')
    axs.set_ylabel(error)
    axs.set_title(f'%Data Removed vs. Error: {model}')
    plt.show()
    
## helper function to run model simulations
# input: pro_data=dataframe of original zinser dataset, data_len=integer specifying number of days to replicate,
# model = string representing model type (must be 'STL', 'baseline', or 'rolling'),
# model_runs = int for simulations to run, save_data=boolean to save data in files 
def run_model_simulations(pro_data, data_len, model, show_plots=True, model_runs=100, save_data=True):
    # list of data percentages to remove
    to_remove=[0, 0.1, 0.25, 0.4, 0.5, 0.6]
    # save error metrics distribution
    rmse_dist=[]
    mase_dist=[]
    # run model 100 times to collect error distributions
    for i in range(0,model_runs):
        # save rmses in each proportion
        print(i)
        rmses=[]
        smapes=[]
        mases=[]
        # run model for several iterations
        for prop in to_remove:
            # run model
            rmse, smape, mase=run_full_model(pro_data, data_len, prop, 
                                             model=model, runs=model_runs, show_plots=show_plots)
            print(f'data removed: {prop}, RMSE: {rmse}, MASE: {mase}')
            # save errors
            rmses.append(rmse)
            smapes.append(smape)
            mases.append(mase)
        # save list of rmses
        rmse_dist.append(rmses)
        mase_dist.append(mases)
    if save_data:
        ## Save files
        print('saving dfs...')
        to_df(rmse_dist, to_remove, 'RMSE').to_pickle(f'simulation_data/{model}_rmse_df')
        to_df(mase_dist, to_remove, 'MASE').to_pickle(f'simulation_data/{model}_mase_df')
    # return error distributions as dataframes
    return(to_df(rmse_dist, to_remove, 'rmse'), to_df(mase_dist, to_remove, 'mase'))



## heler function to run block missing data
# input: remove=float for proportion of data to remove
def block_run(data, days, remove, model):
    # list of data blokcs to remove 
    block_lens=[3,5,12,24,48,72]
    row_list=[]
    for block in block_lens:
        # check if chunk proportions exceeds data length
        if (len(data)*days)*remove < block:
            continue
        # else run model
        rmse, smape, mase=run_full_model(data, days, remove, 
                                         blocks=True, block_len=block, model=model, runs=100)
        # save rows
        row = {'model':model,'remove':remove, 'block':block, 'rmse':rmse, 'mase':mase}
        print(row)
        row_list.append(row)
    # return rows as df
    return(pd.DataFrame(row_list))


## helper function to run model simulations with block removed data
# input: pro_data=dataframe of original zinser dataset, data_len=integer specifying number of days to replicate,
# model = string representing model type (must be 'STL', 'baseline', or 'rolling'),
# model_runs = int for simulations to run, save_data=boolean to save data in files 
def run_block_model_simulations(pro_data, data_len, model, show_plots=True, model_runs=100, save_data=False):
    # list of data percentages to remove
    to_remove=[0, 0.1, 0.25, 0.4, 0.5, 0.6]
    # save error metrics distribution
    error_dfs=[]
    # run model 100 times to collect error distributions
    for i in range(0,model_runs):
        # save rmses in each proportion
        print(i)
        runs=pd.DataFrame(columns=['model','remove','block','rmse','mase'])
        # run model for several iterations
        for prop in to_remove:
            # run blocks model
            block_df=block_run(pro_data, data_len, prop, model)
            # list of dicts to df
            runs=pd.concat([runs,block_df])
        # save df to list
        error_dfs.append(runs.reset_index(drop = True))
    if save_data:
        ## Save files
        print('saving dfs...')
        pd.concat(error_dfs).to_pickle(f'simulation_data/block_{model}_error')
    # return list of error dfs
    return(error_dfs)


## helper function to run model with noise
# input: data=df to run model on days=int for number of days to simulate,
# remove=float for proportion of data to remove, model=str for model name
def noise_run(data, days,remove,model, add_trend=False, trend_df=None, trend=None):
    # list of data blokcs to remove 
    noise_percents=[0.05, 0.1, 0.25, 0.5, 0.75, 1]
    row_list=[]
    for noise in noise_percents:
        # run model
        bagged, rmse, mase, resid_percent=run_full_model(data, 
                                         days=days, 
                                         remove=remove,
                                         add_trend=add_trend,
                                         trend_df=trend_df,
                                         noise=noise,
                                         model=model)
        # save rows
        if add_trend:
            row = {'model':model,'remove':remove, 'trend': trend, 'noise':noise, 'rmse':rmse, 'mase':mase}
        else:
            row = {'model':model,'remove':remove, 'noise':noise, 'rmse':rmse, 'mase':mase}
        print(row)
        row_list.append(row)
    # return rows as df
    return(pd.DataFrame(row_list), bagged)

## function to run model simulations with noise
# input: pro_data=df to run model on, data_len=int for number of days to simulate,
# model=str for model name, show_plots (deprecated), model_runs=# simulations to run, 
# save_data= boolean flag to save data to file
def run_noise_model_simulations(pro_data, data_len, model, add_trend=False, trend_df=None, trend=None,
                                show_plots=True, model_runs=100, save_data=False):
    # save data in df
    if add_trend:
        runs=pd.DataFrame(columns=['model','remove','trend','noise','rmse','mase'])
    else:
        runs=pd.DataFrame(columns=['model','remove','noise','rmse','mase'])
    for i in range(0,model_runs):
        # save rmses in each proportion
        print(i)
        # run model for several iterations
        run,bagged=noise_run(pro_data, data_len, 0, model, add_trend=add_trend, trend_df=trend_df, trend=trend)
        # list of dicts to df
        runs=pd.concat([runs,run])
        # save df to list
    if save_data:
        runs.to_pickle(f'simulation_data/noise_{model}_error')
    return(runs.reset_index(drop = True),bagged)