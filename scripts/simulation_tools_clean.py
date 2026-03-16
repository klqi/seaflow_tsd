import numpy as np

## helper function to add gaussian noise by varying std
# inputs: x=data, mu = mean, std=standard deviation
# returns: data with noise added
def gaussian_noise(x,mu,std):
    noise = np.random.normal(mu, std, size = x.shape)
    x_noisy = x + noise
    return x_noisy 


from scipy import stats
## helper function to calculate p-value for linear regression
def calc_pvalue(X,y,params,pred):
    new_X = np.append(np.ones((len(X),1)), X, axis=1)
    # calculate mean squared error
    MSE = (sum((y-pred)**2))/(len(new_X)-len(new_X[0]))
    # matrix algebra to get p-value
    v_b = MSE*(np.linalg.inv(np.dot(new_X.T,new_X)).diagonal())
    s_b = np.sqrt(v_b)
    t_b = params/ s_b
    p_val =[2*(1-stats.t.cdf(np.abs(i),(len(new_X)-len(new_X[0])))) for i in t_b]
    # return raw p-value
    return(p_val[1])

## helper function to calculate daily averaged hourly growth rate
from sklearn.linear_model import LinearRegression
from rate_functions import calc_se
import pandas as pd
def calc_daily_avg_growth(df, col, model='not base'):
    # check model type
    if model=='base':
        base_df=pd.concat([df.head(1), df.tail(1)])
        # fit data on only first and last values
        X = np.array([0, int(base_df['hour'].diff().values[1])]).reshape(-1, 1)
        y = np.log(base_df[col].values)
        # set y intercept as first value
        y_int=np.log(base_df.head(1)[col].values[0])
    else:
        # fit data on entire day
        X = np.arange(0, len(df)).reshape(-1, 1)
        y = np.log(df[col].values)
        # add y intercent, log Qc at sunrise
        y_int=np.log(df.loc[df['time_of_day']=='sunrise', col]).values[0]
        
    # Step 1: Subtract the known y-intercept from all y values
    y_centered = y - y_int
    # Step 2: Fit the linear regression model
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y_centered)
    # Step 3: Add the known y-intercept back to the model's intercept
    model.intercept_ += y_int
    # get slope and coefficients
    slope=model.coef_[0]
    params = np.append(model.intercept_,model.coef_)
    # step 4: predict new y values on original df's X and dY
    X = np.arange(0, len(df)).reshape(-1, 1)
    y = np.log(df[col].values)
    pred=model.predict(X)

    # calculate p-value and standard error
    pval=calc_pvalue(X,y,params,pred)
    se=calc_se(X,y,params,pred)
    return(slope, pval,se)

## helper function to calculate daily averaged hourly growth rate
from sklearn.linear_model import LinearRegression
from scipy import stats
def calc_daily_avg_growth(df, col, model='not base'):
    # check model type
    if model=='base':
        base_df=pd.concat([df.head(1), df.tail(1)])
        # fit data on only first and last values
        X = np.array([0, int(base_df['hour'].diff().values[1])]).reshape(-1, 1)
        y = np.log(base_df[col].values)
        # set y intercept as first value
        y_int=np.log(base_df.head(1)[col].values[0])
    else:
        # fit data on entire day
        X = np.arange(0, len(df)).reshape(-1, 1)
        y = np.log(df[col].values)
        # add y intercent, log Qc at sunrise
        y_int=np.log(df.loc[df['time_of_day']=='sunrise', col]).values[0]
        
    # Step 1: Subtract the known y-intercept from all y values
    y_centered = y - y_int
    # Step 2: Fit the linear regression model
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y_centered)
    # Step 3: Add the known y-intercept back to the model's intercept
    model.intercept_ += y_int
    # get slope and coefficients
    slope=model.coef_[0]
    params = np.append(model.intercept_,model.coef_)
    # step 4: predict new y values on original df's X and dY
    X = np.arange(0, len(df)).reshape(-1, 1)
    y = np.log(df[col].values)
    pred=model.predict(X)

    # calculate p-value and standard error
    pval=calc_pvalue(X,y,params,pred)
    se=calc_se(X,y,params,pred)
    return(slope, pval,se)

## helper function to find daily growth for each day in a cruise
# save calculated daily growth rates
def calc_diel_growth(df,col,model,day_len=12):
    # calculate number of days of time series
    days=df.loc[df['time_of_day']=='sunrise'].index.values-1
    # store growth and calculated p values
    daily_growth=[]
    pvals=[]
    ses=[]
    # sunrise 1 hour after minimum, sunset 1 hour after maximum
    for day in days:
        start=day+1
        # set sunrise, sunset, and day time 
        df.loc[start,'time_of_day']='sunrise'
        df[start+1:start+day_len]['time_of_day']='day'
        df.loc[start+day_len,'time_of_day']='sunset'
        # calcaulate daily averaged hourly growth since sunrise
        if model.startswith('base'):
            ## check if any rows are null
            base_day=df[start:start+day_len]
            if len(base_day.loc[base_day[col].isnull()]) >0:
                # find first and last null values
                base_good=base_day.loc[base_day[col].notnull()]
                # skip if whole day is empty or only has 1 hour
                if len(base_good)<2:
                    # save values as nan
                    daily_growth.append(np.nan)
                    pvals.append(np.nan)
                    ses.append(np.nan)
                    continue
                # else calculate growth on non-null values
                slope, pval, se=calc_daily_avg_growth(base_good, col,model='base')
                slope=slope/(len(base_good))
            else:
                # need to divide by the number of hours that have passed
                slope, pval, se=calc_daily_avg_growth(base_day, col, model='base')
                slope=slope/(day_len-1)
        else:
            slope, pval, se=calc_daily_avg_growth(df[start:start+day_len], col)
        daily_growth.append(slope)
        pvals.append(pval)
        ses.append(se)
    # return as dataframe
    growth_df=pd.DataFrame(data=[daily_growth,pvals,ses]).T
    growth_df.columns=['growth','pval','se']
    growth_df['model']=model
    return(growth_df)


import random
from itertools import product
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
    ids = random.sample(list(product(range(qc_only.shape[0]), range(qc_only.shape[1]))), n)
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

import math
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


from statsmodels.tsa.seasonal import seasonal_decompose
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


## running entire model w/o bootstrapping or zinser data
def run_full_model(df, col, remove=0, noise=0, blocks=False, model='STL', show_plots=True):
    # first generate noise if prompted
    if noise > 0:
        # calculate std to sample from noise
        x=df[col]
        # set mean to 0 for gaussian noise
        mu=0
        # multiply noise by Qc standard deviation
        std = noise * np.std(x) 
        # add noise to data
        df['data_with_noise']=gaussian_noise(x,mu,std)
    # don't generate noise    
    else:
        df['data_with_noise']=df[col]
    
    # generate missing data if prompted
    if remove > 0:
        # generate blocks of missing datam
        if blocks:
            # remove block length and amount of data to remove
            missing=generate_random_chunks(df,'data_with_noise',remove, missing_col='data_with_missing')
        else:
            # generate misisng data at random
            missing=generate_missing_data(df, 'data_with_noise', remove,missing_col='data_with_missing')
        # save missing data
        missing_data=missing['data_with_missing']
        # calculate imputed values
        impute_df=run_imputation(missing, 'data_with_noise', missing_col='data_with_missing',period=24, interval=1)
        # rename col
        impute_df.rename(columns={'with_missing':'data_with_missing'},inplace=True)
        # reset time of day columns
        impute_df['time_of_day']=df['time_of_day']
        # check if imputation ran
        if impute_df is None:
            print('Imputation Failed')
            return
    else: # run STL model on complete dataset (no missing values)
        # add necessary columns
        impute_df=df.copy()
        # replace with noise column (doesn't matter if noise was added or not)
        impute_df['with_missing']=impute_df['data_with_noise']
        # check if there are missing values, and replace with noise
        impute_df['data_with_missing']=np.where(impute_df['with_missing']>0, impute_df['with_missing'], 
         np.mean(impute_df['with_missing']))
        # save missing data
        missing_data=impute_df['data_with_missing']

    ## run STL model
    if model.lower().startswith('s'):
        # get tsd components
        tsd_df=run_STL(impute_df, col='data_with_missing',period=24)
        # calculate growth
        growth_df=calc_diel_growth(tsd_df, 'diel', model)
        growth_df['remove']=remove
        growth_df['noise']=noise
    ## run naive model
    elif model.lower().startswith('n'):
        # get tsd components
        tsd_df=run_naive(impute_df, col='data_with_missing',period=24)
        # save missing components 
        tsd_df['unfilled']=missing_data
        growth_df=calc_diel_growth(tsd_df, 'diel', model)
        growth_df['remove']=remove
        growth_df['noise']=noise
    ## run STL model
    elif model.lower().startswith('roll'):
        # get components from rolling model
        pro_seasonal, pro_trend, pro_resid = rolling_tsd(impute_df.set_index('hour'), 'data_with_missing', period=24,
                                                        window=3, type='log additive', extrapolate=True)
        pro_all=summarize_rolling(pro_seasonal, pro_trend, pro_resid)
        pro_all.rename(columns={'seasonal':'diel'}, inplace=True)
        # get other necessary columns`a
        tsd_df=pd.merge(pro_all, impute_df[['hour','time_of_day','data_with_missing']], on='hour')
        # calculate growth and productivity
        growth_df=calc_diel_growth(tsd_df, 'diel', model)
        growth_df['remove']=remove
        growth_df['noise']=noise
    ## run baseline model (just dusk+dawn, no decomposition)
    elif model.lower().startswith('base'):
        tsd_df=impute_df.copy()
        # replace with missing data (don't use imputed values) if remove >0; else leave as is
        if remove > 0:
            tsd_df['data_with_missing']=missing['data_with_missing']
        growth_df=calc_diel_growth(tsd_df, 'data_with_missing', model)
        growth_df['remove']=remove
        growth_df['noise']=noise
        # remove unnecessary columns
        #tsd_df.drop(columns = ['Qc', 'hourly_growth', 'pop'], inplace = True)
        
    else: 
        return('Choose a valid model: baseline, naive, rolling, or STL')
    return(tsd_df, growth_df)

from sklearn.metrics import mean_squared_error
### helper function to run simulated dataset trials
# inputs: df = control dataframe, model = string representation of model, trial_type = string representing "noise" or "remove", data_level = float representation of data removed or noise added
# outputs: tsd_df = dataframe with resulting tsd components from corrspnding model, growth_df = dataframe of calculated growth rates
def model_trial(df, model, trial_type, data_level):
    # try running full model, skip if error
    ### data removeal ###
    if trial_type == 'remove':
        try:
            tsd_df, growth_df=run_full_model(df, 'Qc', model=model, remove=data_level)
        except Exception as error:
            print(f'failed: {error}')
            return None, None
    ### add noise ###
    elif trial_type == 'noise':
        try:
            tsd_df, growth_df=run_full_model(df, 'Qc', model=model, noise=data_level)
        except Exception as error:
            print(f'failed: {error}')
            return None, None
    else: 
        print("Please choose either 'remove' or 'noise' trial.")
        return None, None
    # save data and model
    tsd_df['model']=model
    # show amount of data removed
    tsd_df[trial_type]=data_level
    # calculate estimated growth rate and error based on actual rates (any model but base will give same result)
    actual_growth_df=calc_diel_growth(df, 'Qc', 'naive', day_len = 12)
    # only get non-null values to calculate RMSE
    good_growth=growth_df.loc[growth_df['growth'].notnull()]
    good_actual=actual_growth_df.loc[good_growth.index]
    # calculate error
    rmse=mean_squared_error(good_actual['growth'].values, 
                            good_growth['growth'].values, squared=False)
    # save values
    tsd_df['rmse']=rmse
    scenario = df['scenario'].unique()[0]
    tsd_df['scenario']=scenario
    growth_df['scenario']=scenario

    # return tsd results and growth rates
    return tsd_df, growth_df