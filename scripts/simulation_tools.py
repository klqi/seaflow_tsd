#! Users/Kathy/anaconda3/envs/seaflow/bin/python3
## script that contains helper functions for zinser simulations experiment

from statsmodels.tsa.seasonal import seasonal_decompose
from itertools import compress, product
import math
from random import sample
import matplotlib as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
# import functions from custom modules
import sys
sys.path.insert(0,'/Users/Kathy/Desktop/UW/seaflow/decomposition_project/scripts/')
from diel_tools import *
from fig_tools import *
from tsd_functions import *
from rate_functions import *

## helper function to generate simulated dataset from Zinser experiment
# input: df=dataframe with hour column to repeat days with, days=integer to specify number of days to extend
# output: returns a dataframe with new simulated data
def generate_simulated(df, days):
    # generate simulated dataset (original is 2 days long, so multiply by half- currently only works for even days)
    sim_df=pd.concat([df[:-1]]*int(days/2)).reset_index().drop(columns=['index'])
    # reformat hour column to represent new, simulated time with every other hour sampling
    sim_df['hour']=np.arange(0,len(sim_df)*2,2)
    return(sim_df)

## helper function to fill shading for "nights"
# inputs: ax=axis object on figure for to apply day/night shading, ylims=tuple with lower/upper bounds for yaxis,
# df=dataframe with hour and par columns to identify where to plot shaded areas
def plot_night_day(ax, df, ylims):
    # fill in shading
    ax.fill_between(df['hour'], 0, 1, 
                    where=df['par'] <= 20,color='gray', alpha=0.3, transform=ax.get_xaxis_transform())
    # set y limits
    ax.set_ylim(ylims[0],ylims[1])
    # hide y ticks
    ax.tick_params(left=False, labelleft=False, top=False, labeltop=False,
                   right=False, labelright=False, bottom=False, labelbottom=False)

## helper function to plot simulated data to check if successful
# inputs: df=dataframe with hour, Qc_hour, GPP, carbonloss, and NPP columns for plotting
def plot_simulated_rates(df):
    # create figure
    fig,axs=plt.subplots(figsize=(10,8), nrows=2,sharex=True)
    # plot hourly Qc
    ln0=axs[0].plot(df['hour'],df['Qc_hour'],marker='.', label='Qc')
    # set labels and titles
    axs[0].set_ylabel('Qc (pg C/cell)')
    # set up secondary axis
    twinax=axs[0].twinx()
    # plot night and day for secondary axis in top subplot
    plot_night_day(twinax, df, (-0.00075,0.01))

    # plot Gross PP in bottom subplot
    ln1=axs[1].plot(df['hour'],df['GPP'],marker='.', label='Gross C-Fixation', c='g')
    # plot C-loss
    ln2=axs[1].plot(df['hour'],df['carbonloss'],marker='.', label='Carbon Loss', c='red',alpha=0.8)
    # plot net PP
    ln3=axs[1].plot(df['hour'],df['NPP'],
                marker='.', label='Net C-Fixation', c='orange',alpha=0.7)

    # plot day night
    twinax=axs[1].twinx()
    # plot night and day for secondary axis in top subplot
    plot_night_day(twinax, df, (-0.00075,0.01))

    # create combined legend for all components 
    lns=ln0+ln1+ln2+ln3
    labs = [l.get_label() for l in lns]
    axs[0].legend(lns,labs,loc='upper right')
    # axis labels
    axs[1].set_xlabel('Time (Hours)')
    axs[1].set_ylabel('Rate (pg C/cell*hour)')

    plt.rcParams.update({'font.size':15})
    plt.tight_layout()

# helper function to iteratively look for missing data (from Züfle et al. 2020 paper)
## edge cases: edges where forward/back pattern is also missing
def iteratively_impute(sub_df, col):
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
            # if 24 hours before doesn't exist or is missing, only use back
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

            diam_pred=(trend_curr/trend_factor)*diam_smooth
            sub_df.loc[index,col]=diam_pred
            # set filled flag
            sub_df.loc[index,'filled']=1
        # not missing
        else:
            # set filled flag if not missing
            sub_df.loc[index,'filled']=1
    return(sub_df)

## helper function to generate missing data from Qc column
# inputs: df=dataframe with "Qc_hourly" column, col = column to generate missing data on, 
# p = float that specifies % of data to remove
# returns: missing_data=data_frame with new column with data removed
def generate_missing_data(df, col, p,missing_col='with_missing'):
    # make copy of dataframe 
    missing_data=df.copy()
    # grab hourly Qc only
    qc_only=missing_data[[col]]
    # calculate number of nans to add to data
    n = int(qc_only.shape[0]*p)
    # randomly sample to get indices to remove data
    ids = sample(list(product(range(qc_only.shape[0]), range(qc_only.shape[1]))), n)
    # grab indices of missing data
    idx, idy = list(zip(*ids))
    # reshape data to numpy
    data=qc_only.to_numpy().astype(float)
    # update numpy view with np.nan
    data[idx,idy]=np.nan
    # store column with missing data in dataframe
    missing_data[missing_col]=data
    # add population column
    missing_data['pop']='prochloro'
    return(missing_data)


## helper function to remove blocks of data
# inputs: df=dataframe with data to remove (needs Qc_hour column), n=int for block length, percent=percent of data
# col=column to generate missing data on
# to remove 
import random
def generate_missing_chunks(df, n, percent, col):
    # create copy of dataframe
    frame=df.copy()
    chunks_to_remove = int(percent*frame.shape[0]/n)
    #split the indices into chunks of length n+2
    chunks = [list(range(i,i+n+2)) for i in range(0, frame.shape[0]-n)]
    drop_indices = list()
    # randomly select chunks to drop
    for i in range(chunks_to_remove):
        # check if empty, break if true
        if len(chunks) == 0:
            break
        indices = random.choice(chunks)
        drop_indices+=indices[1:-1]
        #remove all chunks which contain overlapping values with indices
        chunks = [c for c in chunks if not any(n in indices for n in c)]
    # add new cols and set drop indices to nan
    frame['with_missing']=frame[col]
    frame.loc[drop_indices, 'with_missing']=np.nan
    frame['pop']='prochloro'
    return frame

## helper function to remove randomly sized blocks of data
# inputs: df=dataframe with data to remove, col=column to remove from,p=proportion of data to remove
from random import choices
def generate_random_chunks(df,col,p,missing_col='with_missing'):
    # make a copy of dataframe
    frame=df.copy()
    # set max length of chunk that can be removed
    max_length = int(len(frame)*p)
    # set chunk sizes
    chunk_sizes=np.arange(1,max_length+1)
    # get data indices
    inds=frame.index.to_list()
    # first arbitrarily choose a chunk sizes
    # need to set limit on chunk sizes based on proportion data removed and dataset length
    chunk=choices(chunk_sizes)
    # loop through to generate chunk sizes
    chunks_to_remove=[]
    # save
    chunks_to_remove.append(chunk[0])
    while max_length>0:
        # reset max_length
        max_length=max_length-chunk[0]
        # generate new chunk with new max length
        if max_length>0:
            chunk=choices(np.arange(1,max_length+1))
            chunks_to_remove.append(chunk[0])
    # sort chunks
    sorted_chunks=np.sort(chunks_to_remove)[::-1]
    
    ## split indices into chunks
    # create chunks for each sorted chunk
    all_chunks = []
    for n in sorted_chunks:
        # create chunked dataset
        chunks = [list(range(i,i+n+1)) for i in range(0, frame.shape[0]-n)]
        all_chunks.append(chunks)

    # keep track of previously seen indices
    drop_indices=list()
    drops=[]
    for chunk in all_chunks:
        # check if index is in any of the chunks
        drops=[x for x in chunk if set(x).intersection(drop_indices)]
        # create mask and chunks to select from
        masks=[n not in drops for n in chunk]
        good_chunk=[b for a, b in zip(masks, chunk) if a]
        # randomly select indices
        indices = random.choice(good_chunk)
        # check if drop index already used
        if any(x in indices for x in drop_indices):
            print('repeat!!!!!')
        # stored
        drop_indices+=indices[0:-1]
    # set indices to nan to remove
    frame[missing_col]=frame[col]
    frame.loc[drop_indices, missing_col]=np.nan
    frame['pop']='prochloro'
    return(frame)


## helper function to run imputation function and fill in data
# input: missing_df=dataframe with 'with_missing' column with data removed
# returns: final_impute=dataframe with imputed data in 'with_missing'
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
        pre_impute =iteratively_impute(missing_resamp, missing_col)
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
                                                  min_count=1).reset_index()
        
        # add flag to check if filled
        missing_resamp['filled']=0
        # iteratively impute
        pre_impute=iteratively_impute(missing_resamp, missing_col).set_index('time')
        # fill additional gaps with linear interpolation
        final_impute = pre_impute.resample('1H').mean().interpolate(method='linear').reset_index()
        # add cruise and population data back, and original data with missing values
        final_impute['cruise']=pd.unique(missing_cont['cruise'])[0]
        final_impute['pop']=pd.unique(missing_cont['pop'])[0]
        final_impute[col]=missing_resamp[col]

    return(final_impute)

## helper function to plot imputed data
# input: df=data frame with imputed values stored in with_missing column. Requires columns with hour and Qc_hour
def plot_imputed(df):
    # plot missing filled data with truth data
    fig,axs = plt.subplots(figsize=(10,8), nrows=2)
    # plot missing data first
    axs[0].plot(df['hour'], df['Qc_hour'], marker='.', label='Original Data')
    # plot filled in data
    axs[0].plot(df['hour'], df['with_missing'], marker='.', label='Imputed Data', alpha=0.5)

    # calculate RMSE
    rmse=mean_squared_error(df.Qc_hour.values, df.with_missing.values, squared=False)

    # add legend and labels
    axs[0].legend(loc='upper right')
    axs[0].set_xlabel('Hour')
    axs[0].set_ylabel('Hourly Qc (pg C/cell*hour)')

    # create second subplot of comparision with missing data
    filled_df=df.loc[df['pop'].isnull()]
    # plot comparision of both
    axs[1].plot(filled_df['Qc_hour'], filled_df['with_missing'], marker='o',linestyle='')
    # add 1:1 line
    axs[1].axline((0.04, 0.04), (0.07, 0.07),c='red', label='1:1 line')
    axs[1].set_xlabel('Original Qc')
    axs[1].set_ylabel('Imputed Qc')
    axs[1].legend(title=f'RMSE: {np.round(rmse,6)}')

    plt.tight_layout()


## helper fucntion to show tsd results
def plot_TSD(res, axes, model):
    # plot observed subplot
    res.observed.plot(ax=axes[0])
    axes[0].set_ylabel('Observed')

    # plot trend subplot
    res.trend.plot(ax=axes[1], legend=False)
    axes[1].set_ylabel('Trend')

    # plot seasonal subplot
    res.seasonal.plot(ax=axes[2], legend=False)
    axes[2].set_ylabel('Diel (Seasonal)')

    # plot residual subplot
    res.resid.plot(ax=axes[3], legend=False)
    axes[3].set_ylabel('Residual')

    # set title and x-axis label
    axes[0].set_title(f"{model} TSD Decomposition")
    axes[3].set_xlabel('Hour')
    plt.show()

# function to run STL model to first get diel component
## input: df=dataframe with imputed data, col=string representation of column with imputed data
## output: data = dataframe with cleaned STL components
def run_STL(df, col, period=12, robust=True):
    # subset df
    # check what col is
    if col.startswith('data'):
        data = df[['hour','time_of_day',col]]
    else:
        data = df[['hour','Qc_hour','par','NPP',col]]
    # get data to run in model
    train=data[col]
    # Run multiplicative STL model
    stl_model = STL(np.log(train), period=period, robust=robust)#, seasonal=15)
    # fit to data
    stl_fit = stl_model.fit()
    # make figure
    #fig,axs=plt.subplots(figsize=(10,8),nrows=4,sharex=True)
    #plot_TSD(stl_fit, axs, 'STL')
    
    # add components to datasets
    data['trend']=np.exp(stl_fit.trend.values.reshape(-1,1))
    data['diel']=np.exp(stl_fit.seasonal.values.reshape(-1,1))
    data['resid']=np.exp(stl_fit.resid.values.reshape(-1,1))
    # return both components
    return data


# function to run STL model to first get diel component
## input: df=dataframe with imputed data, col=string representation of column with imputed data
## output: data = dataframe with cleaned naive tsd components
def run_naive(df, col,period=12):
    # subset df 
    if col.startswith('data'):
        data = df[['hour','time_of_day',col]]
    else:
        data = df[['hour','Qc_hour','par','NPP',col]]
    # resolve 0 and negative values,
    data[col]=np.where(data[col]>0, data[col], 
         np.mean(data[col]))
    # get data for model
    train=data[col]
    # Run multiplicative naive model
    model = seasonal_decompose(train, model='multiplicative',period=period, extrapolate_trend=True)
    
    # add components to datasets
    data['trend']=(model.trend.values.reshape(-1,1))
    data['diel']=(model.seasonal.values.reshape(-1,1))
    data['resid']=(model.resid.values.reshape(-1,1))
    # return df
    return data

## helper function to check stationarity of a time series using adfuller and kpss tests
# inputs: df=dataframe with growth column and qc column to calculate productivity, growth_col = string to represent
# column with trend values, qc_col=string to represent column with qc values
from statsmodels.tsa.stattools import adfuller, kpss
def calc_productivity(df, growth_col, qc_col):
    # see if data is stationary
    timeseries=df[qc_col]
    ## get p-values from both tests
    # null hypothesis: data is non-stationary
    adftest=adfuller(timeseries.dropna())[1]
    # null hypothesis: data is stationary
    kpsstest = kpss(timeseries, regression='c', nlags=12)[1]
    # if stationary, calculate average qc across dataset
    if (adftest < 0.05)&(kpsstest > 0.05):
        # get conversion term from mean Qc
        prod_conv=np.mean(df[qc_col])
        df['productivity']=df[growth_col]*prod_conv
    # if one or both rejects null, calculate breakpoints
    else:
        print('Calculating break points ... not really')
        prod_conv=np.mean(df[qc_col])
        df['productivity']=df[growth_col]*prod_conv
        # # use pelt to get break points
        # bkps=pelt_bkps(df['trend'])
        # # set break points as new column in df
        # df['bkps']=make_bkps(bkps)
        # # find mean trend per change point
        # mean_bkps=df.groupby(['pop','bkps']).agg({
        #     'trend':'mean'}).reset_index()
        # # rename col
        # mean_bkps.rename(columns={'trend':'mean_bkp'},inplace=True)
        # # merge with data that has growth
        # # merge with df with growth col
        # prod_df=df.merge(mean_bkps)
        # # calculate productivity 
        # prod_df['productivity']=prod_df[growth_col]*prod_df['mean_bkp']
        # # reset df name
        # df=prod_df

    return(df)

## helper function to plot productivity results, also shows only for day time values
# input: df=data frame with productivity values stored in productivity column. Requires columns with par,
# hour, and NPP
def plot_productivity(df):
    # calculate RMSE for day values only
    days_only=df.loc[df['par']>0]
    # skip missing values at the end due to productivity calculation
    rmse=mean_squared_error(days_only.NPP.values[:-1], days_only.productivity.values[:-1], squared=False)
    
    # plot NPP time series
    fig,axs=plt.subplots(figsize=(10,8),nrows=2)
    # plot original NPP and TSD estimated NPP
    axs[0].plot(df['hour'],df['NPP'],marker='.',label='"Measured" NPP')
    axs[0].plot(df['hour'],df['productivity'],marker='.',label='TSD NPP')
    # plot night and day
    twinax=axs[0].twinx()
    plot_night_day(twinax, df, ylims=(0,0.2))
    axs[0].legend()

    # plot comparison of day productivity values
    axs[1].plot(days_only['NPP'],days_only['productivity'],marker='.',linestyle='',label='Comparison')
    axs[1].plot(days_only['NPP'],days_only['NPP'],label='1:1 line',c='r',marker='')
    # make axis labels
    axs[1].set_xlabel('Measured NPP')
    axs[1].set_ylabel('TSD NPP')
    axs[1].legend(title=f'RMSE: {np.round(rmse,5)}')

    plt.tight_layout()
    return fig,rmse


from arch.bootstrap import MovingBlockBootstrap
from numpy.random import standard_normal
from numpy.random import RandomState

## helper function to run bootstrapping desired number of times to calculate growth and productivity
# inputs: df=dataframe with columns: hour, resid, trend, diel to generate new bootstrapped time series dfs, 
# model: string to determine whihc model to run ('STL', 'rolling'),
# seed=int for random state, period=int for defining block length, runs=int for number of times to run bootstrap
def run_bootstrapping(df, model='STL', period=13, runs=100):
    ## apply mbb to residuals of STL decomposition
    mbb_df=df.set_index('hour')
    data=mbb_df.resid.values
    # separate residuals into overlapping blocks
    mbb=MovingBlockBootstrap(period, data)

    # grab synthetic datasets in each bootstrap
    mbb_data=[]
    count=1

    # run bootstrapping to calculate growth and productivity
    for data in mbb.bootstrap(runs):
        # grab blocked residual values
        new_resid=data[0][0]
        # generate synthetic ts from trend and diel components
        new_ts=new_resid*mbb_df.trend*mbb_df.diel
        # store in mbb_df as a column
        mbb_df['mbb_qc']=new_ts
        ## run decomposition to calculate growth rate
        # get tsd components from chosen model
        if model.lower().startswith('s'):
            mbb_tsd=run_STL(mbb_df.reset_index(), 'mbb_qc')
        elif model.lower().startswith('roll'):
            # hour is already index
            seasonal, trend, resid=rolling_tsd(mbb_df, 'mbb_qc', period=12,
                                                 window=3, type='log additive', extrapolate=True)
            mbb_tsd = summarize_rolling(seasonal, trend, resid)
            mbb_tsd.rename(columns={'seasonal':'diel'}, inplace=True)
            # get other necessary columns
            mbb_tsd=pd.merge(mbb_tsd, mbb_df.reset_index()[['hour','Qc_hour','par','NPP']], on='hour')
        elif model.lower().startswith('n'):
            mbb_tsd=run_naive(mbb_df.reset_index(), 'mbb_qc')
        else:
            print('Please choose a model!')
            return
        # calculate hourly growth by exponential growth and maintain correct order
        mbb_tsd['hourly_growth']=exp_growth(mbb_tsd, 'diel',2).shift(-1)
        # add column to keep track of bootstrap run
        mbb_tsd['bs_run']=count
        # calculate hourly productivity
        rates_df=calc_productivity(mbb_tsd,'hourly_growth','Qc_hour')
        # save into list
        mbb_data.append(rates_df)
        # increase count
        count+=1
    # return list of bootstrapped dfs
    return(mbb_df, mbb_data)


## helper function to plot bootstrapping results
# inputs: mbb_df=pdataframe with hour set as index and columns for NPP, list_dfs=list of bootstrapped dataframes,
# runs=int with number of bootstrapping runs completed
def plot_bootstrapping(mbb_df, list_dfs, runs):
    # make plot with bootstrapped results
    fig,axs=plt.subplots(figsize=(10,6))
    # iteratively plot each bootstrap time series
    for bs_df in list_dfs:
        axs.plot(bs_df.hour,bs_df.productivity, alpha=0.25)
    # plot original measured NPP
    axs.plot(mbb_df.NPP, c='k', label='Original NPP', marker='.')
    # plot day and night shading
    twinax=axs.twinx()
    plot_night_day(twinax, mbb_df.reset_index(), ylims=(0.04, 0.075))
    # set labels, legend, and title
    axs.set_xlabel('Time (Hours)')
    axs.set_ylabel('Hourly C-Fixation (pg C/cell)')
    axs.legend()
    axs.set_title(f'{runs}x Bootstrapped STL Decomposition')
    plt.show()


## helper functions for returning 95% confidence interval by quantiles
def lower_q(x):
    return x.quantile(0.0275)

def upper_q(x):
    return x.quantile(0.975)

## helper function to calculate 95% CI
def calc_95CI(df, groupby_col, agg_col):
    # perform groupby
    stats_df=df.groupby(groupby_col).agg(
        {
            agg_col:['mean', 'count', 'std']
        }
    )
    # filter out rows with 0 or nan
    stats_df=stats_df.loc[(stats_df != 0).all(axis=1), :]
    # 95% confidence = 1.96x SD from the mean
    ci95_hi=[]
    ci95_lo=[]
    # calculate CI
    for i in stats_df.index:
        m, c, s = stats_df.loc[i]
        ci95_hi.append(m + 1.96*s)
        ci95_lo.append(m - 1.96*s)
    # flatten multi index to single
    stats_df.columns=['_'.join(col).strip() for col in stats_df.columns.values]
    # set 95 CI columns
    stats_df[f'{agg_col}_ci95_hi'] = ci95_hi
    stats_df[f'{agg_col}_ci95_lo'] = ci95_lo
    # drop count col
    stats_df=stats_df[stats_df.columns.drop(list(stats_df.filter(regex='count')))]
    # return dataframe
    return(stats_df.reset_index())

## helper function to aggregate bootstrapping results (bagging)
# inputs: list_dfs=list of bootstrapped dataframes, runs=int with number of bootstrapping runs completed
def bagging_results(list_dfs):
    ## get bagged results
    # first concatenate list of dataframes
    bs_all = pd.concat(list_dfs)
    # calculate mean run and uncertainty (95% confidence interval) for residuals and productivity
    # get stats for residuals and productivity
    resid_stats=calc_95CI(bs_all, 'hour','resid')
    prod_stats=calc_95CI(bs_all, 'hour','productivity')
    # merge stats
    merged_stats=resid_stats.merge(prod_stats)
    # get NPP and PAR data
    bs_bag=merged_stats.merge(bs_all.groupby(['hour']).first().reset_index()[['hour','par','NPP']])
    # return grouped dataframe
    return(bs_bag.set_index('hour'))

## Helper function to get error metrics for models
# inputs: pred=list or series of model predicted values, actual=list or series of actual values
# returns: rmse, smape, mase
def error_metrics(pred, actual):
    # calculate rmse
    rmse=mean_squared_error(actual, pred, squared=False)
    
    # calculate smape
    a=abs(actual-pred)
    b=(abs(actual)+abs(pred))/2
    smape=np.mean(a/b)*100
    
    ## calculate mase
    ## first calculate naive seasonal mae
    # reset indices for actual series
    mase_actual=actual.reset_index()
    # get index for start
    start_m=mase_actual.loc[mase_actual['hour']>=24].index[0]
    # get hours to search
    hours = mase_actual.loc[start_m:, 'hour']
    # get column to search
    col = [col for col in mase_actual if col.startswith('NPP')][0]
    naive_ae=[]
    # loop through each hour
    for i in hours:
        # get the value starting 24 hours later
        curr=mase_actual.loc[mase_actual['hour']==i, col].values[0]
        past=mase_actual.loc[mase_actual['hour']==i-24, col].values[0]
        naive_ae.append(abs(curr-past))
    # calculate mase
    mase=np.mean(abs(actual-pred))/np.mean(naive_ae)
    return rmse, smape, mase


## Helper function to get error metrics for models by looping through each bootstrapped time series
# inputs: mbb_dfs=list of dataframes with bootstrapped results (needs productivity and NPP, diel and resid columns)
# returns: rmse and SNR
def loop_error_metrics(mbb_dfs):
    # store rmse and snr
    rmse_list = []
    snr_list = []
    ## calculate SNR from signal (diel component) and noise (residual component)
    for df in mbb_dfs:
        ## calculate rmse
        # get day time values
        days_only = df.loc[df['par']>0]
        # get predicted and actual values
        pred = days_only['productivity'][:-1]
        actual = days_only['NPP'][:-1]
        rmse=mean_squared_error(actual, pred, squared=False)
        # save rmse
        rmse_list.append(rmse)
        ## calculate SNR
        signal=df['diel']
        noise=df['resid']
        # save snr
        snr_list.append(np.var(signal)/np.var(noise))
    return(rmse_list, snr_list)

## helper function to plot bagging results
# inputs: df=dataframe with bagged results with hour as index. Requires columns: par_mean, NPP, 
# productivity_mean, productivity_lower_q, productivity_upper_q, model=string defining model name
def plot_bagging(df,model):
    # get days only
    days_only = df.loc[df['par']>0]
    # calculate error metrics
    pred=days_only.productivity_mean[:-1]
    actual=days_only.NPP[:-1]
    # get rmse
    rmse, smpae, mase=error_metrics(pred, actual)
    
    # make plot with bagged results
    fig,axs=plt.subplots(figsize=(10,8), nrows=2)
    # plot original and bagged results
    axs[0].plot(df.NPP, label='Original NPP', marker='.')
    axs[0].plot(df.productivity_mean, label='Bagged NPP', marker='.')
    # plot 95% confidence range
    axs[0].fill_between(df.index, df.productivity_ci95_lo, 
                     df.productivity_ci95_hi, color = 'green', alpha=0.5, label='95% CI')
    # plot day and night
    twinax=axs[0].twinx()
    plot_night_day(twinax, df.reset_index(), ylims=(0.04, 0.075))
    axs[0].set_xlabel('Time (Hours)')
    axs[0].set_ylabel('Hourly C-Fixation (pg C/cell)')
    axs[0].legend()
    axs[0].set_title(f'100x Bootstrapped {model} Decomposition')

    # plot comparison of day productivity values
    axs[1].plot(days_only['NPP'],days_only['productivity_mean'],marker='.',linestyle='',label='Comparison')
    axs[1].plot(days_only['NPP'],days_only['NPP'],label='1:1 line',c='r',marker='')
    # make axis labels
    axs[1].set_xlabel('Measured NPP')
    axs[1].set_ylabel('TSD Bagged NPP')
    axs[1].legend(title=f'RMSE: {np.round(rmse,5)}')
    # clean up
    plt.tight_layout()
    # show plot
    # return fig and error metrics 
    return fig, rmse


## helper function to sumarize rolling tsd model for bagging
# inputs: seasonal=series for seasonal component output, trend = series for trend component outut, resid=series for resid component
def summarize_rolling(seasonal, trend, resid):
    # join all components
    all_comp=pd.concat([seasonal, trend, resid], axis=1)
    # drop duplicates
    all_comp = all_comp.loc[:,~all_comp.columns.duplicated()]
    # group by on each hour for mean
    comp_mean = all_comp.groupby(['hour']).mean().reset_index()
    return(comp_mean)


## helper function to add gaussian noise by varying std
# inputs: x=data, mu = mean, std=standard deviation
# returns: data with noise added
def gaussian_noise(x,mu,std):
    noise = np.random.normal(mu, std, size = x.shape)
    x_noisy = x + noise
    return x_noisy 


## helper function to find residuals signicantly different from a mean=1
# ipnuts: mbb_data=list of dataframes returned from run_bootstrapping()
from scipy import stats
def find_bad_resids(mbb_data):
    mbb_df = pd.concat(mbb_data)
    # store signficantly different residuals
    resid_hours=[]
    # perform 1 sample t-test to check if residual is significantly different from 0, use 99% accuracy
    threshold=0.01
    for hour in pd.unique(mbb_df.hour):
        sub_resid=mbb_df.loc[mbb_df['hour']==hour, 'resid']
        # run test
        # null hypothesis: mean residual is the same as the population mean (1)
        # reject null hypothesis if p-value < threshold
        stat,p=stats.ttest_1samp(sub_resid.values, popmean=1)
#         print(f'Hour: {hour}, p-value: {np.round(p,4)}')
        # check null hypothesis
        if p < threshold:
            # save the hour
            resid_hours.append(hour)
    # return hours with bad residuals
    return(resid_hours)


## helper function to calculate percentage of significantly different residuals
# inputs: resid_hours=list of hours returned from find_bad_resids(), bagged=dataframe returned from bagging_results()
def bad_resid_percent(resid_hours, bagged):
    # return length of resid hours / length of bagged *100
    return (len(resid_hours)/len(bagged))*100

## helper function to add a trend to data
# inputs: df=dataframe to add trend to, trend = list containing trend values
def add_trend(df, trend):
    df=df.copy()
    # set experiment days
    df['experiment_day']=df['hour'] // 24
    days = int(len(df)/12)
    
    # multiply by trend -- this results in changing amplitude
    # in this dataset, 12 rows represent 24H
    # multiply trend by each day in row
    for day, new_trend in zip(range(0,days), trend):
        # subset by day and multiply diel and resid by new trend
        sub = df.loc[df['experiment_day']==day]
        df.loc[sub.index,'new_ts'] = new_trend * sub.Qc_hour
    return df

## function to run experiments with trend/noise/missing data, TSD model, bootstrapping, and bagging workflow
# returns error metrics 
# inputs: df=dataframe with dataset to generate simulations from, days=int with # days to simulate data for, 
# model=string to choose model (options are 'baseline', 'STL'),
# remove=float for proportion of data to remove, runs=int with times to run bootstrapping,
# noise = float for proportion of noise to add
def run_full_model(df, days, remove, add_trend=False, trend_df=None,
                   noise=0, blocks=False, model='STL', runs=100, show_plots=True):
    # check if running model with trend
    if add_trend:
        # use inputted data for model
        sim_df=trend_df
        # set column to for further testing
        col='new_ts'
    else:
        # create day simulated data for Qc data
        sim_df=generate_simulated(df, days)
        # set column to for further testing
        col='Qc_hour'

    # first generate noise if prompted
    if noise > 0:
        # calculate std to sample from noise
        x=sim_df[col]
        # set mean to 0 for gaussian noise
        mu=0
        std = noise * np.std(x) 
        # add noise to data
        sim_df['with_noise']=gaussian_noise(x,mu,std)
    # don't generate noise    
    else:
        sim_df['with_noise']=sim_df[col]
    
    # generate missing data if prompted
    if remove > 0:
        # generate blocks of missing datam
        if blocks:
            # remove block length and amount of data to remove
            #missing=generate_missing_chunks(sim_df, block_len, remove, 'with_noise')
            missing=generate_random_chunks(sim_df,'with_noise',remove)
        else:
            # generate misisng data at random
            missing=generate_missing_data(sim_df, 'with_noise', remove)
        # calculate imputed values
        impute_df=run_imputation(missing)
        # check if imputation ran
        if impute_df is None:
            print('Imputation Failed')
            return
    else: # run STL model on complete dataset (no missing values)
        # add necessary columns
        impute_df=sim_df.copy()
        # replace with noise column (doesn't matter if noise was added or not)
        impute_df['with_missing']=impute_df['with_noise']
        # check if there are missing values, and replace with noise
        impute_df['with_missing']=np.where(impute_df['with_missing']>0, impute_df['with_missing'], 
         np.mean(impute_df['with_missing']))
    
    if model.lower().startswith('s'):
        # get tsd components
        tsd_df=run_STL(impute_df, 'with_missing')
        
        # calculate hourly growth by exponential growth and maintain correct order
        tsd_df['hourly_growth']=exp_growth(tsd_df, 'diel',2).shift(-1)
        # calculate hourly productivity
        # always calculated on original Qc, even with trend, to compare back with 14C results
        rates_df=calc_productivity(tsd_df,'hourly_growth','Qc_hour')
        
        # run bootstrapping to get list of new dataframes
        mbb_df, mbb_data=run_bootstrapping(rates_df, runs=runs)

        # perform bagging on bootstrapped data
        bagged=bagging_results(mbb_data)
        ## return error metrics of day time values
        days_only = bagged.loc[bagged['par']>0]
       # get error metrics
        rmse_list,snr_list=loop_error_metrics(mbb_data)
        # calculate residual analytics
        resid_hours=find_bad_resids(mbb_data)
        resid_percent=bad_resid_percent(resid_hours, bagged)
        # flag bad residuals in bagged data
        bagged['resid_flag']=0
        # flag with 1 if bad residual
        bagged.loc[bagged.index.isin(resid_hours), 'resid_flag']=1
    elif model.lower().startswith('base'):
        ## generate simulated data
        base_growth=exp_growth(impute_df,'with_missing',2).shift(-1)
        # add to data
        baseline=impute_df.copy()
        baseline['hourly_growth']=base_growth
        # # set all negative growth to 0
        # baseline.loc[baseline['hourly_growth']<0, 'hourly_growth']=0
        # calculate productivity
        rates_df=calc_productivity(baseline, 'hourly_growth','Qc_hour')
        # calculate error metrics of day time values
        days_only = rates_df.loc[rates_df['par']>0]
        # set hour to index
        days_only = days_only.set_index('hour')
        # get rid of nans or infinities
        days_only=days_only.replace()
        # calculate error metrics (RMSE only)
        pred=days_only.productivity[:-1]
        actual=days_only.NPP[:-1]
        # list is just one value because not bootstrapped
        rmse_list = [mean_squared_error(actual, pred, squared=False)]
        # rename rates_df to return properly
        bagged=rates_df
        # no residuals bc no decomposition in baseline model
        snr_list=[np.nan]
        resid_percent=np.nan
    elif model.lower().startswith('roll'):
        # get components from rolling model
        pro_seasonal, pro_trend, pro_resid = rolling_tsd(impute_df.set_index('hour'), 'with_missing', period=12,
                                                        window=3, type='log additive', extrapolate=True)
        pro_all=summarize_rolling(pro_seasonal, pro_trend, pro_resid)
        pro_all.rename(columns={'seasonal':'diel'}, inplace=True)
        # get other necessary columns`a
        pro_all=pd.merge(pro_all, impute_df[['hour','Qc_hour','par','NPP']], on='hour')
        # calculate growth and productivity
        pro_all['hourly_growth']=exp_growth(pro_all, 'diel',2).shift(-1)
        rates_df=calc_productivity(pro_all, 'hourly_growth', 'Qc_hour')
        # bootstrap model
        rolling_mbb_df, mbb_data=run_bootstrapping(rates_df, model='rolling')
        # perform bagging on bootstrapped data
        bagged=bagging_results(mbb_data)
        # get error metrics
        rmse_list,snr_list=loop_error_metrics(mbb_data)
        # calculate residual analytics
        resid_hours=find_bad_resids(mbb_data)
        resid_percent=bad_resid_percent(resid_hours, bagged)
        # flag bad residuals in bagged data
        bagged['resid_flag']=0
        # flag with 1 if bad residual
        bagged.loc[bagged.index.isin(resid_hours), 'resid_flag']=1
    elif model.lower().startswith('n'):
        # get tsd components
        tsd_df=run_naive(impute_df, 'with_missing')
        
        # calculate hourly growth by exponential growth and maintain correct order
        tsd_df['hourly_growth']=exp_growth(tsd_df, 'diel',2).shift(-1)
        # calculate hourly productivity
        rates_df=calc_productivity(tsd_df,'hourly_growth','Qc_hour')
        
        # run bootstrapping to get list of new dataframes
        mbb_df, mbb_data=run_bootstrapping(rates_df, model='naive')

        # perform bagging on bootstrapped data
        bagged=bagging_results(mbb_data)
        # get error metrics
        rmse_list,snr_list=loop_error_metrics(mbb_data)
        # calculate residual analytics
        resid_hours=find_bad_resids(mbb_data)
        resid_percent=bad_resid_percent(resid_hours, bagged)
        # flag bad residuals in bagged data
        bagged['resid_flag']=0
        # flag with 1 if bad residual
        bagged.loc[bagged.index.isin(resid_hours), 'resid_flag']=1

    else: 
        return('Choose a valid model: baseline, naive, rolling, or STL')
    return(bagged, rmse_list, snr_list)