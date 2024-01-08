#! Users/Kathy/anaconda3/envs/seaflow/bin/python3
## script that contains helper functions for calculating growth and productivity rates

import pandas as pd
import numpy as np
from tsd_functions import summarize_by_pop_time, interp_by_time
from diel_tools import days_by_sunrise, find_night
from statsmodels.tsa.seasonal import STL

### helper function to run TSD model (STL only for now) to get cleaned dataframe for estimating growth/productivity
## input: df=cruise dataframe
## output: days_only=dataframe with saved diel components for only day time values, days_growth = hourly data per day
def get_tsd_outputs(df):
    # interpolate for pro and syn separately
    raw_cruise = df[df['lat'].notnull()].reset_index()
    # run linear interpolation helper function to fill in missing data
    pro_res = interp_by_time(raw_cruise, 'prochloro')
    syn_res = interp_by_time(raw_cruise, 'synecho')
    # run diel cycle on 
    diel=find_night(pro_res.reset_index())
    # get days by sunrise
    sr_days = days_by_sunrise(diel).drop(columns=['pop','diam_med','n_per_uL','c_per_uL'])
    # merge with output of sunrise days to get cruise days defined by sunrise
    cruise_daily = pd.concat([pro_res, 
                              syn_res]).reset_index().merge(sr_days[['time','time_day','lat',
                                                                     'lon','night','cruise_day']])
    # Run STL
    fit_pro = STL(np.log(pro_res['diam_med']), period=24)
    fit_syn = STL(np.log(syn_res['diam_med']), period=24)
    stl_pro = fit_pro.fit()
    stl_syn = fit_syn.fit()
    # save seasonal (diel) components
    pro_seas=pd.DataFrame(stl_pro.seasonal)
    pro_seas['pop']='prochloro'
    syn_seas=pd.DataFrame(stl_syn.seasonal)
    syn_seas['pop']='synecho'
    both_seas=pd.concat([pro_seas,syn_seas]).reset_index()
    # merge with cruise_daily
    cruise_days=both_seas.merge(cruise_daily)
    # only get day values
    days_only=cruise_days.loc[cruise_days['night']!='night'].sort_values(by='time')
    return days_only


## helper function to do a centered difference to calculate growth rate, edges calculated by first order diff
# input: df=dataframe with diel component, col=specififying diel component of particular model
def centered_diff(df,col):
    return(pd.Series(np.gradient(df[col]), name='hourly_growth'))

## helper function to calculate exponential growth rate, meant to calculate for the day time values
# input: df=dataframe with diel component, col=specififying diel component of particular model, spacing=time between measurements
def exp_growth(df,col,spacing):
    return(np.log(df[col]/df[col].shift())/spacing)

## function to calculate daily average hourly growth
# input: df=dataframe with cruise data
# output: returns df with daily averaged hourly growth + std
def calc_daily_hourly_growth(cruise):
    # just grab cruise name from last row
    name=cruise.iloc[-1].cruise
    days_only=get_tsd_outputs(cruise)
    # calculate average hourly cellular growth for each day after night time has been removed
    hourly_all=[]
    for day in pd.unique(days_only['cruise_day']):
        # subset by day
        sub_df=days_only.loc[days_only['cruise_day']==day]
        # calculate hourly cellular growth by centered diff
        # check how pandas will organize the df
        try:
            test=sub_df.sort_values(by='time').groupby(['pop']).apply(lambda v: 
                                             centered_diff(v,'season')).unstack().reset_index()
        except:
            continue
        if len(test)!=len(sub_df):
            hourly=sub_df.sort_values(by='time').groupby(['pop']).apply(lambda v: 
                                         centered_diff(v,'season')).reset_index()
        else:
            hourly=test
        # add cols
        hourly['cruise_day']=day
        hourly['time_day']=sub_df['time_day'].values
        # clean up
        hourly=hourly.drop(columns=['hourly_growth']).rename(columns={0:'hourly_growth'})
        hourly['time']=sub_df.sort_values(by='time').reset_index()['time']
        hourly_all.append(hourly)
    # get all hours
    hourly_all=pd.concat(hourly_all)
    days_growth=days_only[['time','lat','lon','temp','pop']].merge(hourly_all)
    # group by each day
    daily_avg=days_growth.groupby(['cruise_day','pop']).agg({
        'lat':'mean',
        'lon':'mean',
        'temp':'mean',
        'hourly_growth':['mean','std'],
        'time_day':'first'
    })
    # clean up index names
    daily_avg.columns=['_'.join(col).strip() for col in daily_avg.columns.values]
    cruise_hourly_avg=daily_avg.reset_index()
    cruise_hourly_avg['cruise']=name
    return days_growth, cruise_hourly_avg


    ### helper functions used in Zinser simulations experiment
    