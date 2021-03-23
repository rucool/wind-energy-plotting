#!/usr/bin/env python

"""
Author: Lori Garzio on 3/23/2021
Last Modified: 3/23/2021
Preliminary sea breeze identification using Tuckerton met tower data at 12m. Data gaps cannot be >1 hour.
"""

import os
import datetime as dt
import glob
import pandas as pd
import functions.common as cf
import functions.seabreeze as sb
pd.set_option('display.width', 320, "display.max_columns", 15)  # for display in pycharm console


def daterange(date1, date2):
    for n in range(int((date2 - date1).days)+1):
        yield date1 + dt.timedelta(n)


def main(stime, etime, sDir):
    #met_dir = /home/coolgroup/MetData/CMOMS/surface/daily  # on server
    met_dir = '/Users/lgarzio/Documents/rucool/bpu/wrf/mettower_data/'  # on local machine

    # find the file extensions for the date range of interest
    fext = []
    for tm in daterange(stime, etime):
        fext.append(tm.strftime('%Y%m%d'))

    # combine data from files into one dataframe
    df = pd.DataFrame()
    wind_height = ['12m']

    for f in fext:
        try:
            fname = glob.glob(met_dir + '*surface.' + f + '.dat')[0]
        except IndexError:
            continue
        try:
            df1 = pd.read_csv(fname)
        except pd.errors.EmptyDataError:
            continue

        # calculate wind direction
        wdir = cf.wind_uv_to_dir(df1['sonic_u(cm/s)'], df1['sonic_v(cm/s)'])
        df1['12m Wind Direction'] = wdir

        cols = ['time_stamp(utc)', 'avg(mph)', 'sonic_u(cm/s)', 'sonic_v(cm/s)', '12m Wind Direction']
        dfc = df1[cols]
        df = df.append(dfc)  # append to the overall dataframe

    if len(df) > 0:
        df['Date and Time'] = pd.to_datetime(df['time_stamp(utc)'])  # format time
        df.sort_values(by='Date and Time', inplace=True)  # make sure the dataframe is sorted by time
        df['DateEST'] = df['Date and Time'] - dt.timedelta(hours=5)

        seabreezes = sb.seabreeze_id_df(df, 'DateEST', '{} Wind Direction'.format(wind_height[0]), 'EST')

        save_file = os.path.join(sDir, 'seabreezes{}_{}-{}.csv'.format(wind_height[0], fext[0], fext[-1]))
        seabreezes.to_csv(save_file, index=False)


if __name__ == '__main__':
    start_time = dt.datetime(2016, 6, 9)  # first date: dt.datetime(2016, 6, 9)
    end_time = dt.datetime(2021, 3, 15)
    save_dir = '/Users/lgarzio/Documents/rucool/bpu/wrf/seabreeze_id'  # on local machine
    #save_dir = '/www/home/lgarzio/public_html'  # on server
    main(start_time, end_time, save_dir)
