from collections import Counter
from scipy import stats
import pandas as pd


## helper to find autocorrelation in residuals by 48-hr rolling window
# input: dataframe with requires resid and cruise_day columns
# output: list of days to remove
def check_resids(df):
    sig_days=[]
    for day in pd.unique(df['cruise_day'])[:-1]:
        # grab 2 days
        check_days=[day, day+1]
        cruise_day=df.loc[(df['cruise_day'].isin(check_days))]
        ## df autocorrelation for 24 hours over past 3 days
        # make df to df significance
        resids=cruise_day['resid'].values
        shift_resid=cruise_day['resid'].shift(24).values
        resid_df=pd.DataFrame([resids, shift_resid]).T.dropna()
        pval=stats.pearsonr(resid_df[0], resid_df[1])[1]
        # save cruise day where residual pvalue is significant
        if pval<0.01:
            # save days
            for x in check_days:
                sig_days.append(x)
    # find duplicate days using counter data structure
    d = Counter(sig_days)
    # return duplicates for removal
    return (list([item for item in d if d[item]>1]))


from sklearn.metrics import mean_squared_error
def r2_rmse(g):
    rmse=mean_squared_error(g['data_with_missing'], g['trend']*g['diel'],squared=False)
    return pd.Series(dict(rmse = rmse))
