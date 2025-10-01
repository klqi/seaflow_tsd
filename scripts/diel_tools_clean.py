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