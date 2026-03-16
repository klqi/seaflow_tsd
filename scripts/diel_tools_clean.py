from datetime import datetime, timedelta, time
# use astral (pyephem not in python 3.8 yet)
from astral import Observer
from astral.sun import sunrise
from astral.sun import sunset
from astral.sun import night
from datetime import datetime
import pandas as pd


## helper function to round to nearest hour
# input: t = datetime object
# output: datetime object rounded to nearest hour
def hour_rounder(t):
    # returns datetime rounded to nearest hour by adding a timedelta hour if minute >= 30
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
               +timedelta(hours=t.minute//30))

## helper function to check whether a datetime is between two other dateteims
# input: begin_time = datetime object at start, end_time = datetime object at end, check_time = datetime object to check
# output: Boolean (True if between, False if not between)
def is_time_between(begin_time, end_time, check_time):
    # If check time is not given, default to current UTC time
    check_time = check_time
    if begin_time < end_time:
        return check_time >= begin_time and check_time <= end_time
    else: # crosses midnight
        return check_time >= begin_time or check_time <= end_time

## helper function to calculate sunrise and sunset times
# input: time = datetime object, obs = observer object (astral)
# output: sr = datetime of sunrise, ss = datetime of sunset
def sunrise_sunset(time, obs):
    # fist calculate sunrise no offset
    sr=hour_rounder(sunrise(obs, date=time))
    # next calculate sunrise with offset
    sr_offset=hour_rounder(sunrise(obs, date=time+pd.DateOffset(1)))
    # if sunrise with offset is before measured time, then calculate sunrise/set with offset
    if (sr_offset <= time):
        # needs offset
        ss=hour_rounder(sunset(obs, date=time+pd.DateOffset(1)))
        return sr_offset,ss
    else:
        # no offset
        ss=hour_rounder(sunset(obs, date=time))
        return sr,ss
    
## helper function to determine what time of day it is given sunrise/sunset times
# input: time = datetime (hourly), sunrise = datetime of sunrise, sunset = datetime of sunset
# output: returns string categorizing time of day
def label_daytime(time, sunrise, sunset):
    status='nan'
    # check if at sunrise
    if time==sunrise:
        status='sunrise'
    # check if at sunset
    elif time==sunset:
        status='sunset'
    # check if at day time (should be between sunrise and sunset)
    elif is_time_between(time, sunrise, sunset):
        status='day'
    # if not between should be night time
    else:
        status='night'
    return status


## helper function to calculate hours of daylight per cruise day using astral
# input: df=dataframe with time, time_of_day, cruise_day, lat, lon columns to calculate hours of daylight using
# output: datframe with new daylength column
def calc_daylength(df):
    # offset day
    df['round_day']=df['time'].dt.round('1d')-pd.DateOffset(1)
    sr_ss=df.loc[(df['time_of_day']=='sunrise')|
                  (df['time_of_day']=='sunset')].groupby('cruise_day').agg({
        'lat':'mean',
        'lon':'mean',
        'round_day':'first'
    }).reset_index()
    # # get exact sunrise and sunset times
    for i, row in sr_ss.iterrows():
        obs=Observer(row['lat'], row['lon'], 0)
        # get sunrise and sunset times
        sr=sunrise(obs, date=row['round_day'])
        ss=sunset(obs, date=row['round_day'])
        # calculate difference between times in hours
        diff=(ss-sr).total_seconds()/3600
        # save in df df
        df.loc[df['cruise_day']==row['cruise_day'], 'daylength']=diff
    return df


# cursed helper function to calculate day/night cycle using astral
# requires lat (float), lon (float), and time (datetime) columns as input
# returns dataframe with added columns, night (day, night, sunrise, sunset) and time_day (offset for calculation)
def find_night(df, offset=True):
    # drop annoying columns
    if 'index' in df:
        df.drop(columns=['index'], inplace=True)
    # offset by 1 day if true
    if offset:
        df['time_day'] = df['time'].dt.round('1d') - pd.DateOffset(1)
    else:
        df['time_day'] = df['time'].dt.round('1d')
    # initialize night col
    df['night'] = 'nan'

    # loooooop (using sunrise/sunset works but dawn/dusk doesnt- why?) IDK
    for index, row in df.iterrows():
        # is it night time?
        obs = Observer(row['lat'], row['lon'], 0)
        sr = sunrise(obs, date = row['time_day'])
        ss = sunset(obs, date = row['time_day'])
        # round to nearest hour
        night_time = [hour_rounder(pd.to_datetime(x)) for x in (sr, ss)]
        # say if time is at sunrise
        if row['time'] == night_time[0]:
            df.loc[index, 'night'] = 'sunrise'
        # sunset check
        elif row['time'] == night_time[1]:
            df.loc[index, 'night'] = 'sunset'
        # day check
        elif is_time_between(night_time[0], night_time[1], row['time']):
            # catch edge case where astral fails to find sunrise
            if (0 < index < len(df)):
                # change to sunrise if the row directly before is night
                if (df.loc[index-1, 'night']=='night'):
                    df.loc[index-1, 'night']='sunrise'
                    df.loc[index,'night']='day'
                else:
                    df.loc[index, 'night'] = 'day'
            else:
                df.loc[index, 'night'] = 'day'
        # night check
        else:
            # catch edge case where astral fails to find sunset
            if (0 < index < len(df)):
                # must be run after index check or will fail
                if (df.loc[index-1, 'night']=='day'):
                    # change previous row to sunrise if before is day
                    df.loc[index-1, 'night']='sunset'
                    # change present row
                    df.loc[index, 'night']='night'
                else:
                    df.loc[index, 'night'] = 'night'
            else:
                df.loc[index, 'night'] = 'night'
    return(df)


# helper function to mark each day by sunset/sunrise instead of by utc time
## input: df=dataframe with day/night/sunrise/sunset labels (run through find_night function)
## output: resulting df with cruise_day column
# get list of indices for sunrises
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

def get_complete_days(data,exclude_night=True):
    # only grab complete days
    day_counts=data.groupby(['cruise_day','pop']).agg({'time_day':'count'}).reset_index()
    complete_days=pd.unique(day_counts.loc[day_counts['time_day'].between(23,25),'cruise_day'])
    # exclude incopmlete days and night if true
    if (exclude_night):
        days=data.loc[data['cruise_day'].isin(complete_days)&(data['night']!='night')]
    else:
        # only exclude incomplete days
        days=data.loc[data['cruise_day'].isin(complete_days)]
    return(days)