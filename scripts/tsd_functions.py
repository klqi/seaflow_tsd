#! Users/Kathy/anaconda3/envs/seaflow/bin/python3
## script that contains helper functions for running and managing TSD results

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from diel_tools import find_night
import matplotlib.dates as mdates
from datetime import timedelta
from diel_tools import plotseasonal, calc_daily_vars

# helper function to resample dataframe by time so we can see where we have missing data
# inputs: df = pd.DataFrame with latitude, time, and pop columns; pop = string for population to resample by
def resample_hourly(df, pop):
    df=df.loc[df['lat'].notnull()]
    df['time'] = pd.to_datetime(df['time'])
    # set the index as time
    df_res = df.set_index('time')
    df_res.index = pd.to_datetime(df_res.index)
    # split by population 
    df_sub = df_res[df_res['pop']==pop]
    df_sub=df_sub.resample('1H').asfreq()#.mean()
    # set cols column so none are missing
    df_sub['pop']=pop
    return(df_sub)


# helper function to interpolate data based on time
# inputs: df = pd.DataFrame, pop = string for column in df, method = interpolation type (default linear)
def interp_by_time(df, pop, method='linear', order=1):
    interp_test = df
    # drop null location values
    interp_test = interp_test[interp_test['lat'].notnull()]
    # reset data type
    interp_test['time'] = pd.to_datetime(interp_test['time'])
    # set the index as time
    interp_test = interp_test.set_index('time')
    interp_test.index = pd.to_datetime(interp_test.index)
    # interpolate by population 
    sub = interp_test[interp_test['pop']==pop]
    # resample by every hour to fill missing with NAs and run interpolation
    if method=='linear':
        sub_res = sub.resample('1H').mean().interpolate(method=method)
    else:
        sub_res = sub.resample('1H').mean().interpolate(method=method, order=order)
    sub_res['pop'] = pop
    # return fully interpolated dataframe
    return(sub_res)

# helper function to append decomp results together
# input: interp_list = list of pd.DataFrames to concat and append results
def concat_tsd_results(interp_list):
    # reset indices
    interp_both = pd.concat(interp_list).reset_index()
    # get night/day 
    interp_both = find_night(interp_both)
    # add resid values into df 
    interp_both.trend = np.nan
    interp_both.loc[interp_both['pop']=='prochloro', 'trend'] = result_pro.trend.values
    interp_both.loc[interp_both['pop']=='synecho', 'trend'] = result_syn.trend.values
    # add seasonal value into df
    interp_both.seasonal = np.nan
    interp_both.loc[interp_both['pop']=='prochloro', 'seasonal'] = result_pro.seasonal.values
    interp_both.loc[interp_both['pop']=='synecho', 'seasonal'] = result_syn.seasonal.values
    # add resid values into df 
    interp_both.resid = np.nan
    interp_both.loc[interp_both['pop']=='prochloro', 'resid'] = result_pro.resid.values
    interp_both.loc[interp_both['pop']=='synecho', 'resid'] = result_syn.resid.values
    # return the concantenated resulting df
    return(interp_both)

from statsmodels.tsa.seasonal import seasonal_decompose
# function to run tsd model n # times based on a specified rolling time (hourly resolution)
# input: df = dataframe for either pro/syn, period = frequency length (24 hours default), rolling = how often to 
# rerun the model (1 hour) default, window = number of days for each model run (default = 3 days)
def rolling_tsd(df, col, period=24, rolling=1, window=3, type='multiplicative'):
    # set start and stop times 
    start = 0
    end = period * window
    # hours possible to run
    cycle = len(df)-end
    # store seasonal components in list
    seasonal_list = []
    trend_list = []
    resid_list = []
    # run on specified rolling basis 
    for n in np.arange(0, cycle+rolling, rolling):
        # slice each part of df for specified period and window length
        n_slice = df[start:end]
        # run decomposition
        if (type=='multiplicative'):
            result = seasonal_decompose(n_slice[col], model=type, period=period)
            # save seasonal components as dataframes in list
            result_df = pd.DataFrame(result.seasonal).reset_index()
            trend_df = pd.DataFrame(result.trend).reset_index()
            resid_df = pd.DataFrame(result.resid).reset_index()
        else: 
            result = seasonal_decompose(np.log(n_slice[col]), model=type, period=period)
            # save seasonal components as dataframes in list
            result_df = pd.DataFrame(np.exp(result.seasonal)).reset_index()
            trend_df = pd.DataFrame(np.exp(result.trend)).reset_index()
            resid_df = pd.DataFrame(np.exp(result.resid)).reset_index()

        # add model number
        result_df['model'] = f'Hour {int(n)}'
        trend_df['model'] = f'Hour {int(n)}'
        resid_df['model'] = f'Hour {int(n)}'
        # append to list to save 
        seasonal_list.append(result_df)
        trend_list.append(trend_df)
        resid_list.append(resid_df)
        # shift start and end indices
        start += rolling
        end += rolling
    # return saved seasonal component list as output 
    return(pd.concat(seasonal_list), pd.concat(trend_list), pd.concat(resid_list))

from statsmodels.tsa.filters.filtertools import convolution_filter
# helper function to extract biomass trend from each cruise
## inputs: interp_df = interpolated dataframe with biomass column, 
## pop = string specifiying population (ie 'prochloro')
def get_trend_only(interp_df, pop, col):
    x=interp_df.loc[interp_df['pop'] == pop, col]
    # 24 hours in one day
    period=24
    # apply convolution filter with centered moving average 
    filt = np.array([0.5] + [1] * (period - 1) + [0.5]) / period
    nsides = int(True) + 1
    biomass_trend = convolution_filter(x, filt, nsides)
    # returns pd.Series with time as index and biomass trend data
    return(biomass_trend)

# helper function to normalize a list/series of values to a desired range
## ipnuts: values = list/series of values, actual_bounds = tuple of min and max of original values, desired_bounds = 
## tuple of min/max of desired range
def normalize(values, actual_bounds, desired_bounds):
    return [desired_bounds[0] + (x - actual_bounds[0]) * 
            (desired_bounds[1] - desired_bounds[0]) / 
            (actual_bounds[1] - actual_bounds[0]) for x in values]


# helper function to summarize rolling tsd outputs by calculating mean and std (error)
## Inputs: df = dataframe with combined model components, name = string for component type (ie seasonal, trend, resid)
def summarize_by_pop_time(df, name):
    df_mean = df.groupby(['pop','time']).agg({
        name:['mean','std']
    }).reset_index()
    df_mean.columns = [' '.join(col).strip() for col in df_mean.columns.values]
    # output: returns summarized mean/std dataframe by hour
    return(df_mean)

# function to plot results of tsd model with other environmental parameters
## inputs: seasonal_df = dataframe with seasonal components, trend_df = dataframe from trend components
## par_df = dataframe with daily summmed par data, diel_df = dataframe with night/day, syn_diel = dataframe for plotting
## syn biomass, cruise_name = string of cruise name
def plot_diel_cycle(seasonal_df, trend_df, diel_df, cruise_name):
    # make figure object
    fig, axs = plt.subplots(ncols=1, nrows=3, sharex=True, sharey=False, figsize=(16,12))
    plt.rcParams.update({'font.size': 15})
    # make secondary axes
    ax0 = axs[0].twinx()
    ax1 = axs[1].twinx()
    
    # plot 1 subplot
    pro_hourly = seasonal_df.loc[seasonal_df['pop']=='prochloro']
    pro_trend = trend_df.loc[trend_df['pop']=='prochloro']
    # set variables
    x = pro_hourly['time']
    x1 = pro_trend['time']
    y1 = pro_trend['trend mean'] 
    y = pro_hourly['seasonal mean']
    error = pro_hourly['seasonal std']
    error1 = pro_trend['trend std']

    # plot seasonal
    l1=axs[0].plot(x, y, label='Diel')
    # plot uncertainity (std)
    axs[0].fill_between(x, y-error, y+error, alpha=0.2)
    # plot trend in secondary axis (std = 0 bc trend calculated by CMA)
    l2=ax0.plot(x1, y1, linestyle='--', label='Trend')
    ax0.fill_between(x1, y1-error1, y1+error1, alpha=0.2)

    # add legend
    lns = l1+l2
    labs = [l.get_label() for l in lns]
    axs[0].legend(lns, labs, loc=0)

    # plot 2 subplot
    syn_hourly = seasonal_df.loc[seasonal_df['pop']=='synecho']
    syn_trend = trend_df.loc[trend_df['pop']=='synecho']
    x = syn_hourly['time']
    x1 = syn_trend['time']
    y1 = syn_trend['trend mean']
    y = syn_hourly['seasonal mean'] 
    error = syn_hourly['seasonal std']
    error1 = syn_trend['trend std']
    l1 = axs[1].plot(x, y, color='green', label='Diel')
    axs[1].fill_between(x, y-error, y+error, color = 'green', alpha=0.2)
    # plot trend in secondary axis axis
    l2=ax1.plot(x1, y1, linestyle='--', color='green', label='Trend')
    ax1.fill_between(x1, y1-error1, y1+error1, alpha=0.2)
    # add legend
    lns = l1+l2
    labs = [l.get_label() for l in lns]
    axs[1].legend(lns, labs, loc=0)

    # add night bars
    # plot par edges as lines for both subplots
    axs[0].fill_between(diel_df['time'], 0, 1, where=diel_df['night'] != 'day',
                    color='gray', alpha=0.3, transform=axs[0].get_xaxis_transform())
    axs[1].fill_between(diel_df['time'], 0, 1, where=diel_df['night'] != 'day',
                    color='gray', alpha=0.3, transform=axs[1].get_xaxis_transform())
    
    # plot 3 subplot (lat/par)
    # plot latitude
    ln1=axs[2].plot(diel_df['time'], diel_df['lat'], c='red', alpha=0.5, label='Latitude')
    # calculate par from diel df
    # par_df, day_inds = calc_daily_vars(diel_df, col='par', func='sum')
    # plot PAR on secodnary axis
    ax2 = axs[2].twinx()
    # ln2=ax2.plot(par_df['time'], par_df['par_sum'], c='orange', alpha=0.5, marker='.', label='Total Daily Par')

    ###### move this into panels to the next column from data in the future (deprecated for now) #########
    # pro_bio = np.log(get_trend_only(diel_df, 
    #               pop='prochloro',
    #               col='biomass'))
    # syn_bio = np.log(get_trend_only(syn_diel, 
    #               pop='synecho',
    #               col='biomass'))
    # ln3=ax2.plot(diel_df['time'],pro_bio.values, alpha=0.5, label='Pro', linestyle=':')
    # ln4=ax2.plot(syn_diel['time'],syn_bio.values, c='g', alpha=0.5, label='Syn', linestyle=':')
    # add legend for panel
    lns = ln1#+ln2#+ln3+ln4
    labs = [l.get_label() for l in lns]
    axs[2].legend(lns, labs)#, loc='center left', bbox_to_anchor=(1.05, 0.5))
    
    # add subplot titles
    axs[0].set_title('Prochlorococcus', color='blue')
    axs[1].set_title('Synechococcus', color='green')

    # add x and y labels
    axs[0].set_ylabel('% Change from Trend', c='blue')
    ax0.set_ylabel('Trend (pgC $\mathregular{cell^{-1}}$)', c='blue')
    axs[1].set_ylabel('% Change from Trend', c='green')
    ax1.set_ylabel('Trend (pgC $\mathregular{cell^{-1}}$)', c='green')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Latitude', c='red')
    # ax2.set_ylabel('PAR', c='orange')
    plt.suptitle(f'{cruise_name} Hourly Averaged Diel Cycle')
    # # fix y range- diel
    # axs[0].set_ylim([0.85, 1.25])
    # axs[1].set_ylim([0.85, 1.25])
    # fix for diam- pro
    #ax0.set_ylim([0, 0.1])
    # syn
    #ax1.set_ylim([0, 0.25])
    # aesthetics
    plt.tight_layout()
    # return figure
    return(fig)
    
# function to run the TSD model and output results
# input: df = input dataframe for cruise, show_figs = show plots or not (default False), type=type of decomposition to run
# (can be log transformed additive or multiplicative, depends on the desired output)
def run_TSD(df, cruise_name, show_tsd_figs=False,type='multiplicative'):
    # drop values with no lat data first to prevent weird interpolation artifacts
    df = df[df['lat'].notnull()].reset_index()
    # run interpolation helper function (linear)
    pro_res = interp_by_time(df, 'prochloro')
    syn_res = interp_by_time(df, 'synecho')
    # run find night to get day/night and sunrise/sunset hours
    diel_df = find_night(pd.concat([pro_res, syn_res]).reset_index().drop_duplicates(
        subset=['time']).sort_values(by='time').reset_index())
    diel_df = diel_df.reset_index().drop(columns=['index'])
    # run naive multiplicative seasonal decompose on interpolated df
    if (type=='multiplicative'):
        result_pro = seasonal_decompose(pro_res['diam_med'], model=type, period=24)
        result_syn = seasonal_decompose(syn_res['diam_med'], model=type, period=24)
    # run log additive
    else:
        result_pro = seasonal_decompose(np.log(pro_res['diam_med']), model=type, period=24)
        result_syn = seasonal_decompose(np.log(syn_res['diam_med']), model=type, period=24)

    # show results of initial TSD model if True
    if show_tsd_figs:
        fig, axes = plt.subplots(ncols=2, nrows=4, sharex=True, figsize=(12,5))
        plotseasonal(result_pro, axes[:,0], pop=f'{cruise_name} Pro', lat=pro_res['lat'])
        plotseasonal(result_syn, axes[:,1], pop=f'{cruise_name} Syn', lat=syn_res['lat'])
        plt.tight_layout()
        plt.show()
    # run model using rolling hourly data on 3 day windows
    pro_seasonal, pro_trend, pro_resid = rolling_tsd(pro_res, 'diam_med', type=type)
    pro_seasonal['pop'] = 'prochloro'
    pro_trend['pop'] = 'prochloro'
    pro_resid['pop'] = 'prochloro'
    syn_seasonal, syn_trend, syn_resid = rolling_tsd(syn_res, 'diam_med', type=type)
    syn_seasonal['pop'] = 'synecho'
    syn_trend['pop'] = 'synecho'
    syn_resid['pop'] = 'synecho'
    # concat into 1 combined df
    seas_all = pd.concat([pro_seasonal, syn_seasonal])
    trend_all = pd.concat([pro_trend, syn_trend])
    resid_all = pd.concat([pro_resid, syn_resid])
    # summarized dfs 
    seas_hourly = summarize_by_pop_time(seas_all, 'seasonal')
    trend_hourly = summarize_by_pop_time(trend_all, 'trend')
    resid_hourly = summarize_by_pop_time(resid_all, 'resid')
    # generate plot
    final = plot_diel_cycle(seas_hourly, trend_hourly, diel_df, cruise_name)
    return(seas_all, trend_all, resid_all, diel_df, final)


## helper function to display the trend and seasonal component strengths, along with the residuals
## inputs: resid_df = dataframe that contains every model run of the residual components, s_df =  dataframe that contains every model run of the seasonal components,
## t_df =  dataframe that contains every model run of the trend components, cruise_name = string for the name of the cruise ran
def component_strength(resid_df, s_df, t_df, cruise_name):
    test_resid = resid_df.copy()
    # add trend and resid together (denominator)
    x = np.log(s_df['seasonal'])
    r = np.log(test_resid['resid'])
    t = np.log(t_df['trend'])
    # add seasonal and residual while ignorning nan
    test_resid['Fs_denom']=np.nansum(np.stack((x,r)), axis=0)
    # add trend and residual while ignorning nan
    test_resid['Ft_denom']=np.nansum(np.stack((t,r)), axis=0)
    # log residual
    test_resid['log resid'] = np.log(test_resid['resid'])
    # group by model run
    group_resid = test_resid.groupby(['model', 'pop']).agg({
        'log resid':'var',
        'Fs_denom': 'var',
        'Ft_denom': 'var',
        'time': 'first'
    }).reset_index()
    # calculate strength of components (var(res)/(var(res+components)))
    group_resid['Fs']=1 - (group_resid['log resid']/group_resid['Fs_denom'])
    group_resid['Ft']=1 - (group_resid['log resid']/group_resid['Ft_denom'])
    # adjust time
    group_resid['time'] = group_resid['time'] + timedelta(hours=72/2)
    # get summarized residual plot
    resid_hourly = summarize_by_pop_time(resid_df, 'resid')
    # plot
    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(16,10), sharex=True)
    x = group_resid.loc[group_resid['pop']=='prochloro', 'time']
    y = group_resid.loc[group_resid['pop']=='prochloro', 'Fs']
    y1 = group_resid.loc[group_resid['pop']=='prochloro', 'Ft']
    # plot pro seasonal strength
    ln1 = axs[0].plot(x,y, marker='.', linestyle='', label='Seasonal')
    # add intervals for each bar
    intervals = np.tile([72],len(y))
    moves = [pd.Timedelta(hours=h/2) for h in intervals]
    axs[0].errorbar(x,y, xerr=moves, fmt="o", alpha=0.1, color='blue')
    axs[0].errorbar(x,y1, xerr=moves, fmt="o", alpha=0.1, color='orange')
    # pro trend strength
    ln2 = axs[0].plot(x,y1, marker='.', linestyle='', label='Trend')
    axs[0].set_ylabel('Pro Component Strengths')
    # plot pro residuals
    x = resid_hourly.loc[resid_hourly['pop']=='prochloro', 'time']
    y = resid_hourly.loc[resid_hourly['pop']=='prochloro', 'resid mean']
    error = resid_hourly.loc[resid_hourly['pop']=='prochloro', 'resid std']
    #axs[0].set_ylim((0,1.1))
    # plot on secondary axis
    ax0 = axs[0].twinx()
    ln3 = ax0.plot(x,y,c='g',label='Residual+Uncertainty', alpha=0.7)
    # add error 
    ax0.fill_between(x, y-error, y+error, alpha=0.2, color='green')
    ax0.set_ylabel('Pro Residuals', c='green')
    ax0.set_ylim((0.93,1.06))
    # add legend
    lns = ln1+ln2+ln3
    labs = [l.get_label() for l in lns]
    axs[0].legend(lns, labs, loc=0)
    x = group_resid.loc[group_resid['pop']=='synecho', 'time']
    y = group_resid.loc[group_resid['pop']=='synecho', 'Fs']
    y1 = group_resid.loc[group_resid['pop']=='synecho', 'Ft']
    axs[1].plot(x,y, marker='.', linestyle='', label='Seasonal')
    axs[1].plot(x,y1, marker='.', linestyle='', label='Trend')
    axs[1].set_ylabel('Syn Component Strengths')
    # plot error bars
    intervals = np.tile([72],len(y))
    moves = [pd.Timedelta(hours=h/2) for h in intervals]
    axs[1].errorbar(x,y, xerr=moves, fmt="o", alpha=0.1, color='blue')
    axs[1].errorbar(x,y1, xerr=moves, fmt="o", alpha=0.1, color='orange')
    
    # plot syn residuals
    x = resid_hourly.loc[resid_hourly['pop']=='synecho', 'time']
    y = resid_hourly.loc[resid_hourly['pop']=='synecho', 'resid mean']
    error = resid_hourly.loc[resid_hourly['pop']=='synecho', 'resid std']
    #axs[1].set_ylim((0,1.1))
    # plot on secondary axis
    ax1 = axs[1].twinx()
    ln3 = ax1.plot(x,y,c='g',label='Residual+Uncertainty', alpha=0.7)
    # add error 
    ax1.fill_between(x, y-error, y+error, alpha=0.2, color='green')
    ax1.set_ylim((0.93,1.06))
    ax1.set_ylabel('Syn Residuals', c='green')

    # xticks
    axs[1].xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.gcf().autofmt_xdate()
    # hide every other label
    for label in axs[1].xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    # set aesthetics
    plt.tight_layout()
    axs[0].set_title(f'{cruise_name} Residual and Strength')
    plt.rcParams.update({'font.size':15})
    # returns the dataframe with calculated strength components (Fs, Ft) and resulting figure
    return(group_resid, fig)
