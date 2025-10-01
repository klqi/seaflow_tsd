import pandas as pd
from astral import Observer
import sys
sys.path.insert(0,'/Users/Kathy/Desktop/UW/seaflow/decomposition_project/scripts/')
from diel_tools_clean import sunrise_sunset, label_daytime

## helper function to fill in sunrise/sunset times by using imputed values
def impute_daytime(df):
    # let's attempt to test imputing day/night time
    df_time=df.set_index('time')
    df_resamp=df_time[['lat','lon','ALOHA']].resample('1H').mean().interpolate(method='linear').reset_index()
    df_interp=pd.merge(df_resamp, df_time.reset_index(), how='left')
    # recalculate sunrise/sunset times from missing values
    df_missing=df_interp.loc[df_interp['Qc_hour'].isnull()]
    # if none are missing, then return
    if df_missing.empty:
            return df_interp
    df_missing['obs']=df_missing.apply(lambda x: Observer(x.lat, x.lon, 0), axis=1)
    res=df_missing.apply(lambda x: sunrise_sunset(x.time, x.obs),axis=1)
    df_missing[['sunrise', 'sunset']]=pd.DataFrame(res.tolist(), index=df_missing.index)
    # set day/night times in interpolated df
    df_interp.loc[df_interp['Qc_hour'].isnull(), 'night']=df_missing.apply(lambda x: 
                                                                           label_daytime(x.time, 
                                                                                         x.sunrise, x.sunset), 
                                                                           axis=1)
    return df_interp

import numpy as np
def days_by_sunrise(diel):
    # create copy of diel dataframe
    diel_df=diel.reset_index().copy()
    dd=diel_df.loc[diel_df['night']=='sunrise']
    
    # only get sunrise indices
    sunrise_inds=dd.index.tolist()
    # set a check for incomplete days (add a 0)
    if sunrise_inds[0]!=0:
        sunrise_inds.insert(0,0)
    # keep track of days
    count=0
    # loop through each index on sunrise days
    for i in sunrise_inds:
        ## cruise day 0 for first day
        if (count==0):
            # check if cruise starts at sunrise (complete first day)
            if (i==0):
                # set inds to be from index 0-1 of sunrise_inds
                inds=np.arange(i, sunrise_inds[count+1])
            # first day not complete
            else:
                # set inds to be from index 0 to next sunrise
                inds=np.arange(0,sunrise_inds[count])
        ## check if we're on the last sunrise index
        elif (i == sunrise_inds[-1]):
            # set inds to be from last sunrise to the end of the cruise
            inds=np.arange(i, len(diel_df))
        else:
            inds=np.arange(i, sunrise_inds[count+1])
        # set indices
        diel_df.loc[inds, 'cruise_day']=count
        # increase count 
        count+=1
    return(diel_df)

# function to run STL model
## input: cruise df
## output: pro_res = dataframe with cleaned tsd components for pro, syn_res=same but with syn
from statsmodels.tsa.seasonal import STL
def run_STL(df, col, period=12, robust=True):
    ## subset initial data
    data=df[['time','hour','cruise_day','time_of_day','lat','lon','par',
             'abundance_dawn','qc0','daylength','cruise','pop',col]]

    # get data to run in model
    train=data[col]
    # Run multiplicative STL model
    stl_model = STL(np.log(train), period=period, robust=robust)#, seasonal=15)
    # fit to data
    stl_fit = stl_model.fit()
    
    # add components to datasets
    data['trend']=np.exp(stl_fit.trend.values.reshape(-1,1))
    data['diel']=np.exp(stl_fit.seasonal.values.reshape(-1,1))
    data['resid']=np.exp(stl_fit.resid.values.reshape(-1,1))
    # return both components
    return data


# helper function to iteratively look for missing data (from Züfle et al. 2020 paper)
## edge cases: edges where forward/back pattern is also missing
import math
def iteratively_impute(sub_df, col):
    # linearly interpolate the trend
    trend_df=sub_df[['trend']].interpolate(method='linear')
    sub_df['trend']=trend_df['trend']
    # reset index
    sub_df=sub_df.reset_index()
    # set cutoff for max index
    cutoff=np.max(sub_df.index)
    # iteratively impute missing data 
    for index, row in sub_df.iterrows():
        # check if missing
        if math.isnan(row[col]):
            # index 24 hours before
            back_ind=index-24
            # index 24 hours after
            fwd_ind=index+24
            # get current trend
            trend_curr = sub_df.loc[index,'trend']
            # if neither exist, skip row (fill with l-int. later)
            if (back_ind<0)&(fwd_ind>=cutoff):
                continue
            # if 24 hours before doesn't exist or is missing, only use forward
            elif (back_ind<0):
                diam_smooth=sub_df.loc[fwd_ind, col]
                trend_factor=sub_df.loc[fwd_ind,'trend']
            # if 24 hours after doesn't exist or is missing, only only use fwd
            elif fwd_ind>=len(sub_df):
                diam_smooth=sub_df.loc[back_ind, col]
                trend_factor=sub_df.loc[back_ind,'trend']
            # if both exist, take the average (ignorning nan)
            else:
                diam_smooth=np.nanmean([sub_df.loc[fwd_ind, col],sub_df.loc[back_ind, col]])
                trend_factor=trend_curr

            ### debugging
            diam_pred=(trend_curr/trend_factor)*diam_smooth
            # check if diam_pred is null or negative
            if (diam_pred <= 0) | (np.isnan(diam_pred)):
                continue
            ##### diam_pred returning null
            sub_df.loc[index,col]=diam_pred
            # set filled flag
            sub_df.loc[index,'filled']=1
        # not missing
        else:
            # set filled flag if not missing
            sub_df.loc[index,'filled']=1
    return(sub_df)


## helper function to run imputation function and fill in data
# input: missing_df=dataframe with 'with_missing' column with data removed
# returns: final_impute=dataframe with imputed data in 'with_missing'
from statsmodels.tsa.seasonal import seasonal_decompose
def run_imputation(missing_df,col,missing_col='with_missing', data_type='simulation',
                   period=12, interval=2):
    # create subsetted df excluding nan values 
    missing_cont=missing_df.loc[missing_df[missing_col].notna()]
    # run seasonal decomposition on raw data and drop nan values for now
    train=missing_cont[missing_col]
    try: 
        decompose=seasonal_decompose(train, model='multiplicative', period=period, extrapolate_trend='freq')
    except: 
        print('Not enough data for imputation')
        return
    #get trend and seasonal components
    missing_cont.loc[train.index, 'trend']=decompose.trend
    missing_cont.loc[train.index, 'seasonal']=decompose.seasonal

    # check what kind of data to set up for interpolation
    if data_type=='simulation':
        # set index as time for interpolation for experimental/simulation data
        missing_cont.set_index('hour',inplace=True)
        # grab first and last hours of complete dataframe
        hour_range=missing_df.iloc[[0,-1]]['hour'].values
        # create resamppled list 
        resampled=np.arange(hour_range[0],hour_range[1]+1, interval)
        # resample interpolated list
        missing_resamp = missing_cont.reindex(missing_cont.index.union(
            resampled)).interpolate('values',limit_direction='both').loc[resampled]
        # add missing diam_med data back to interpolated data
        missing_resamp[missing_col]=missing_cont[missing_col]
            # add flag to check if filled
        missing_resamp['filled']=0
        # iteratively impute
        pre_impute = iteratively_impute(missing_resamp, missing_col)
        # fill additional gaps with linear interpolation
        final_impute = pre_impute.reindex(pre_impute.index.union(
            resampled)).interpolate(limit_direction='both',axis=0).loc[resampled].reset_index()
        # replace altered with its original values
        if col.startswith('data'):
            final_impute[col]=missing_df[col].values
        else:
            final_impute['Qc_hour']=missing_df['Qc_hour']
            final_impute['NPP']=missing_df['NPP']
            final_impute['par']=missing_df['par']
    else:
        # set index as time for interpolation for field data
        missing_resamp=missing_cont.reset_index(drop=True).copy()
        missing_cont.set_index('time',inplace=True)
        # resample and get only fill missing_col values
        missing_resamp=missing_cont.resample('1H').agg(pd.Series.sum, 
                                                  min_count=1)
        
        # add flag to check if filled
        missing_resamp['filled']=0
        # iteratively impute
        pre_impute=iteratively_impute(missing_resamp, missing_col).set_index('time')
        # fill additional gaps with linear interpolation
        final_impute = pre_impute.resample('1H').mean().interpolate(method='linear').reset_index()
        # add cruise and population data back, and original data with missing values
        final_impute['cruise']=pd.unique(missing_cont['cruise'])[0]
        final_impute['pop']=pd.unique(missing_cont['pop'])[0]
        final_impute[col]=missing_resamp.reset_index()[col]

    return(final_impute)


# function to run tsd model n # times based on a specified rolling time (hourly resolution)
# input: df = dataframe for either pro/syn, period = frequency length (24 hours default), rolling = how often to 
# rerun the model (1 hour) default, window = number of days for each model run (default = 3 days)
# output: 
def rolling_tsd(df, col, period=24, rolling=1, window=3, type='multiplicative', extrapolate=False):
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
        #### debugging ####
        data=n_slice[col]
        # run decomposition
        if (type=='multiplicative'):
            result = seasonal_decompose(data, model=type, period=period, extrapolate_trend=extrapolate)
            # save seasonal components as dataframes in list
            result_df = pd.DataFrame(result.seasonal).reset_index()
            trend_df = pd.DataFrame(result.trend).reset_index()
            resid_df = pd.DataFrame(result.resid).reset_index()
        else: 
            result = seasonal_decompose(np.log(data), model="additive", period=period, extrapolate_trend=extrapolate)
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


## helper function to sumarize rolling tsd model for bagging
# inputs: seasonal=series for seasonal component output, trend = series for trend component outut, resid=series for resid component
# output: dataframe with hourly grouped means of TSD results
def summarize_rolling(seasonal, trend, resid):
    # join all components
    all_comp=pd.concat([seasonal, trend, resid], axis=1)
    # drop duplicates
    all_comp = all_comp.loc[:,~all_comp.columns.duplicated()]
    # group by on each hour for mean
    comp_mean = all_comp.groupby(['hour']).mean().reset_index()
    return(comp_mean)

## running entire model w/ bootstrapping (no simulation) (VERSION 1!!)
# needs to be run on 1 dataset at a time (ie: 1 population for 1 cruise)
## runs both STL and rolling model, but chooses the results from the model with a lower SE
from diel_tools_clean import calc_daylength
from rate_functions import exp_growth, get_daily_growths
def run_full_model(df,col,missing_col,pop):
    ###### set up data for model ######
    abund_threshold=0.02
    # Criteria 1: filter out those with low abundance
    df=df.loc[df['abundance']>=abund_threshold]
    # split by population
    pop_df=df[df['pop']==pop].reset_index(drop=True).sort_values(by='time')
    # run imputation algorithm and sort by time
    impute_df=run_imputation(pop_df,col=col,missing_col=missing_col,
                             data_type='field',period=24,interval=1)
    # check if impute_df returns a value
    if impute_df is None:
        # try to resolve impute df by splitting data into different segments to run in cruise
        # return if unresolvable
        return None,None, None, None
    # set consecutive hours for df throughout length of cruise
    impute_df['hour']=np.arange(0,len(impute_df))
    ## find night and day, and cruise days -NEW METHOD
    # create observer col
    impute_df['obs']=impute_df.apply(lambda x: Observer(x.lat, x.lon, 0), axis=1)
    # create hourly rounded sunrise/sunset cols for bb 
    res=impute_df.apply(lambda x: sunrise_sunset(x.time, x.obs), axis=1)
    impute_df[['sunrise', 'sunset']]=pd.DataFrame(res.tolist(), index=impute_df.index)
    # label by using night/day col
    impute_df['night']=impute_df.apply(lambda x: label_daytime(x.time, x.sunrise, x.sunset), axis=1)
    impute_df_days=days_by_sunrise(impute_df).drop(columns=['index'])
    impute_df_days.rename(columns={'night':'time_of_day'},inplace=True)

    # save sunrise df subset
    sunrise=impute_df_days.loc[impute_df_days['time_of_day']=='sunrise']
    ## add dawn values to tsd data
    # calculate day length
    uniq_days=sunrise['time'].dt.round('1d')
    for day in uniq_days:
        # grab abundance and qc values at dawn
        abundance_dawn=sunrise.loc[sunrise['time'].dt.round('1d')==day, 'abundance'].values[0]
        qc_dawn=sunrise.loc[sunrise['time'].dt.round('1d')==day, 'data_with_missing'].values[0]
        # set in impute df
        impute_df_days.loc[impute_df_days['time'].dt.round('1d')==day, 'abundance_dawn']=abundance_dawn
        impute_df_days.loc[impute_df_days['time'].dt.round('1d')==day, 'qc0']=qc_dawn
    # calculate day length and add to tsd data
    impute_df_days=calc_daylength(impute_df_days)
    # calculate biomass 
    impute_df_days['biomass']=impute_df_days['Qc_hour']*impute_df_days['abundance']
    # temporarily save
    if pop == 'prochloro':
        impute_df_days.to_pickle('data/indian_ocean/imputed_df_v1_all.pickle')

    # calculate cruise length which model to run
    cruise_len=len(impute_df_days)//24
    
    ## Run both models!!
    ## run STL model and get tsd components
    stl_tsd=run_STL(impute_df_days, col='data_with_missing',period=24)
    stl_tsd['ALOHA']=impute_df_days['ALOHA']
    stl_tsd['biomass']=impute_df_days['biomass']
    # calculate daily averaged hourly growth
    stl_bagged, skip_days=get_daily_growths(stl_tsd)
    # specify model
    stl_tsd['model']='STL'
    stl_bagged['model']='STL'
    
    # make dataframe for skipped days based on short cruise length for STL Model
    skip_STL=pd.DataFrame(columns=['cruise','cruise_day','pop','model', 'lat','lon'])
    skip_STL['cruise_day']=skip_days
    skip_STL['pop']=pop
    skip_STL['model']='STL'
    skip_STL['cruise']=pd.unique(df['cruise'].values)[0]
    # round impute_df_days by cruise day
    mean_days_df=impute_df_days.groupby(['pop','cruise_day']).mean().reset_index()
    skip_STL['lat']=mean_days_df.loc[mean_days_df['cruise_day'].isin(skip_days),'lat'].values
    skip_STL['lon']=mean_days_df.loc[mean_days_df['cruise_day'].isin(skip_days),'lon'].values
    ## run rolling model
    # skip if not long enough
    if cruise_len < 5:
        # return only STL results
        tsd_df=stl_tsd
        bagged=stl_bagged
        return(impute_df_days, tsd_df, bagged, skip_STL)
        
    # get components from rolling model
    pro_seasonal, pro_trend, pro_resid = rolling_tsd(impute_df_days.set_index('hour'), 'data_with_missing', period=24,
                                                    window=3, type='log additive', extrapolate=True)
    pro_all=summarize_rolling(pro_seasonal, pro_trend, pro_resid)
    pro_all.rename(columns={'seasonal':'diel'}, inplace=True)
    # get other necessary columns`a
    rolling_tsd_df=pd.merge(pro_all, impute_df_days[['time','hour','cruise','pop','time_of_day','lat','lon',
                                             'abundance_dawn','qc0','daylength',
                                             'cruise_day','data_with_missing']], on='hour')
    # define station aloha
    rolling_tsd_df['ALOHA']=impute_df_days['ALOHA']
    rolling_tsd_df['biomass']=impute_df_days['biomass']
    # calcaulte daily avg growth and productivity
    rolling_bagged, skip_days=get_daily_growths(rolling_tsd_df)
    # specify model
    rolling_tsd_df['model']='Rolling'
    rolling_bagged['model']='Rolling'

    ## make skip df for rolling
    # make dataframe for skipped days based on short cruise length for STL Model
    skip_rolling=pd.DataFrame(columns=['cruise','cruise_day','pop','model','lat','lon'])
    skip_rolling['cruise_day']=skip_days
    skip_rolling['pop']=pop
    skip_rolling['model']='Rolling'
    skip_rolling['cruise']=pd.unique(df['cruise'].values)[0]
    skip_rolling['lat']=mean_days_df.loc[mean_days_df['cruise_day'].isin(skip_days),'lat'].values
    skip_rolling['lon']=mean_days_df.loc[mean_days_df['cruise_day'].isin(skip_days),'lon'].values
    # return both STL and rolling results
    tsd_df=pd.concat([rolling_tsd_df, stl_tsd])
    bagged=pd.concat([rolling_bagged, stl_bagged])
    return(impute_df_days, tsd_df, bagged, pd.concat([skip_STL, skip_rolling] ))