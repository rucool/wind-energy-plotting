#!/usr/bin/env python

"""
Author: Lori Garzio on 5/18/2020
Last Modified: 9/2/2020
Creates pcolor plots of wind speed for the most current 3 days of SODAR data.
"""

import argparse
import sys
import datetime as dt
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
pd.set_option('display.width', 320, "display.max_columns", 15)  # for display in pycharm console
plt.rcParams.update({'font.size': 14})


def format_date_axis(axis, figure):
    datef = mdates.DateFormatter('%m-%d %H:%M')
    axis.xaxis.set_major_formatter(datef)
    figure.autofmt_xdate()


def main(args):
    save_file = args.save_file
    sodar_dir = '/home/coolgroup/MetData/CMOMS/sodar/daily/'
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    # define files for the 3 most recent days of data
    today = dt.datetime.utcnow()
    days = [2, 1]
    fext = []
    for d in days:
        fext.append((today - dt.timedelta(days=d)).strftime('%Y%m%d'))
    fext.append(today.strftime('%Y%m%d'))

    # combine data from files into one dataframe
    df_ws = pd.DataFrame()  # wind speed matrix for pcolor plot
    wind_height = ['30m', '40m', '50m', '60m', '80m', '100m', '120m', '140m', '160m', '180m', '200m']

    # check if there is a file for today, if not don't make a plot
    try:
        glob.glob(sodar_dir + '*sodar.' + today.strftime('%Y%m%d') + '.dat')[0]
    except IndexError:
        print('No SODAR file for today {} - skipping plot'.format(today.strftime('%Y%m%d')))
        quit()

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
            dropcols = [qc_colname, '{} Wind Vert'.format(wh), '{} Wind Turbulence'.format(wh),
                        '{} Wind Direction'.format(wh)]
            dfc = dfc.drop(columns=dropcols)  # drop the columns that aren't needed

            if len(file_df) == 0:
                file_df = file_df.append(dfc)
            else:
                for c in dfc.columns.tolist():  # add next height as a new column in the dataframe for that day
                    file_df[c] = dfc[c]
        df_ws = df_ws.append(file_df)  # append to the wind speed matrix to combine data for multiple days

    if len(df_ws) > 0:
        df_ws = df_ws.reset_index()
        df_ws['Date and Time'] = pd.to_datetime(df_ws['Date and Time'])  # format time

        # find data gaps > 30 mins
        for i, row in df_ws.iterrows():
            if i > 0:
                t1 = row['Date and Time']
                t0 = df_ws.iloc[i - 1]['Date and Time']

                # calculate the difference between two rows of data in minutes
                diff_mins = (t1 - t0).total_seconds() / 60

                # if the data gap is >30 minutes, add rows of data containing nans, which prevents the pcolor function
                # from filling in data gaps
                if diff_mins > 30:
                    line1 = [t0 + dt.timedelta(minutes=10)]
                    line2 = [t1 - dt.timedelta(minutes=10)]
                    for c in df_ws.columns:
                        if 'Wind Speed' in c:
                            line1.append(float('NaN'))
                            line2.append(float('NaN'))

                    # add the lines of NaNs to the dataframe
                    df_ws = df_ws.append(pd.DataFrame([line1], columns=df_ws.columns))
                    df_ws = df_ws.append(pd.DataFrame([line2], columns=df_ws.columns))

        # sort dataframe on time
        df_ws.sort_values(by='Date and Time', inplace=True)
        df_ws = df_ws.set_index('Date and Time')

        # make pcolor plot
        ws = df_ws.values * 1.94384  # convert wind speeds from m/s to knots
        tm = np.array(df_ws.index)
        ht = np.array([30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200])

        plt.set_cmap('BuPu')

        fig, ax = plt.subplots(figsize=(12, 6))
        plt.subplots_adjust(right=0.98)
        ax.tick_params('both', length=5, width=2, which='major')

        # define color limits based on max wind speeds
        # if np.nanmax(ws) >= 80:
        #     color_lims = [0, 100]
        # elif np.nanmax(ws) >= 60:
        #     color_lims = [0, 80]
        # elif np.nanmax(ws) >= 40:
        #     color_lims = [0, 60]
        # else:
        #     color_lims = [0, 40]

        h = ax.pcolor(tm, ht, ws.T, vmin=0, vmax=40, shading='auto')
        plt.colorbar(h, ax=ax, label='Wind Speed (knots)', extend='max', pad=0.02)

        ax.set_title('Tuckerton SODAR Wind Speed')
        ax.text(.81, -.26, 'rucool.marine.rutgers.edu', size=12, transform=ax.transAxes)
        ax.set_ylabel('Height (m)')
        ax.set_xlabel('Time (GMT)')

        format_date_axis(ax, fig)

        plt.savefig(save_file, dpi=200)
        plt.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-s', '--save_file',
                            dest='save_file',
                            type=str,
                            help='Full file path to save directory and save filename')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
