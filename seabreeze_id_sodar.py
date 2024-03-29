#!/usr/bin/env python

"""
Author: Lori Garzio on 2/18/2021
Last Modified: 3/23/2021
Preliminary sea breeze identification using SODAR data at 100m. Data gaps cannot be >1 hour.
"""

import os
import datetime as dt
import glob
import pandas as pd
import functions.seabreeze as sb
pd.set_option('display.width', 320, "display.max_columns", 15)  # for display in pycharm console


def daterange(date1, date2):
    for n in range(int((date2 - date1).days)+1):
        yield date1 + dt.timedelta(n)


def main(stime, etime, sDir):
    sodar_dir = '/home/coolgroup/MetData/CMOMS/sodar/daily/'  # on server
    #sodar_dir = '/Users/lgarzio/Documents/rucool/bpu/wrf/sodar_plotting/data3/'  # on local machine

    # find the file extensions for the date range of interest
    fext = []
    for tm in daterange(stime, etime):
        fext.append(tm.strftime('%Y%m%d'))

    # combine data from files into one dataframe
    df = pd.DataFrame()
    #wind_height = ['100m']
    wind_height = ['30m']

    for f in fext:
        file_df = pd.DataFrame()
        try:
            fname = glob.glob(sodar_dir + '*sodar.' + f + '.dat')[0]
        except IndexError:
            continue
        df1 = pd.read_csv(fname)

        # for each wind height, grab all columns of data for that height, turn rows where quality is <60 to NaN,
        # drop columns that aren't needed, and append the wind speeds from that height as a new column in the
        # dataframe
        for wh in wind_height:
            cols = [x for x in df1.columns.tolist() if wh == x.split(' ')[0]]  # get all the columns for one height
            cols.append('Date and Time')
            dfc = df1[cols]
            qc_colname = '{} Quality'.format(wh)
            dfc = dfc.set_index('Date and Time')
            dfc = dfc.mask(dfc[qc_colname] < 60)  # turn rows where quality is <60 to NaN
            dropcols = [qc_colname, '{} Wind Vert'.format(wh), '{} Wind Turbulence'.format(wh)]
            dfc = dfc.drop(columns=dropcols)  # drop the columns that aren't needed

            if len(file_df) == 0:
                file_df = file_df.append(dfc)
            else:
                for c in dfc.columns.tolist():  # add next height as a new column in the dataframe for that day
                    file_df[c] = dfc[c]
        df = df.append(file_df)  # append to the overall dataframe

    if len(df) > 0:
        df = df.reset_index()
        df['Date and Time'] = pd.to_datetime(df['Date and Time'])  # format time
        df.sort_values(by='Date and Time', inplace=True)  # make sure the dataframe is sorted by time
        df['DateEST'] = df['Date and Time'] - dt.timedelta(hours=5)

        seabreezes = sb.seabreeze_id_df(df, 'DateEST', '{} Wind Direction'.format(wh), 'EST')

        save_file = os.path.join(sDir, 'seabreezes{}_{}-{}.csv'.format(wh, fext[0], fext[-1]))
        seabreezes.to_csv(save_file, index=False)


if __name__ == '__main__':
    start_time = dt.datetime(2015, 5, 13)  # first date: dt.datetime(2015, 5, 13)
    end_time = dt.datetime(2021, 3, 15)
    #save_dir = '/Users/lgarzio/Documents/rucool/bpu/wrf/seabreeze_id'  # on local machine
    save_dir = '/www/home/lgarzio/public_html'  # on server
    main(start_time, end_time, save_dir)
