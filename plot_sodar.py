#!/usr/bin/env python

"""
Author: Lori Garzio on 5/18/2020
Last Modified: 6/15/2020
Creates two wind barb plots: 1) for the most current 3 days, and 2) for the current day of SODAR data.
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


def define_barb_subset(df):
    # determine if the barbs should be subset for the plot based on the number of data points available
    if len(df) >= 7500:
        sub = 5
    elif len(df) >= 6000:
        sub = 2
    else:
        sub = 1  # no subset

    return sub


def define_color_lims(speed):
    # define color limits based on max wind speeds
    if np.nanmax(speed) >= 80:
        color_lims = [0, 100]
    elif np.nanmax(speed) >= 60:
        color_lims = [0, 80]
    elif np.nanmax(speed) >= 40:
        color_lims = [0, 60]
    else:
        color_lims = [0, 40]

    return color_lims


def format_date_axis(axis, figure):
    datef = mdates.DateFormatter('%m-%d %H:%M')
    axis.xaxis.set_major_formatter(datef)
    figure.autofmt_xdate()


def plot_barbs(tm, ht, u, v, sub, clims, ttl, figname):
    plt.set_cmap('jet')
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(right=0.98)
    ax.set_facecolor('grey')
    ax.tick_params('both', length=5, width=2, which='major')

    b = ax.barbs(tm[::sub], ht[::sub], u[::sub], v[::sub], np.sqrt(u * u + v * v)[::sub],
                 fill_empty=True, rounding=False, sizes=dict(emptybarb=0.1, spacing=0.2, height=.3),
                 clim=clims)

    # add colorbar
    plt.colorbar(b, ax=ax, label='Wind Speed (knots)', extend='both', pad=0.02)

    ax.set_title('Tuckerton SODAR Winds: {}'.format(ttl))
    ax.set_ylabel('Height (m)')
    ax.set_xlabel('Time (GMT)')
    plt.ylim(10, 220)

    format_date_axis(ax, fig)

    plt.savefig(figname, dpi=200)
    plt.close()


def format_data(df):
    # separate wind speed from wind direction
    df_ws = df[df['variable'].str.contains('Wind Speed')]
    df_direction = df[df['variable'].str.contains('Wind Direction')]

    tm = pd.to_datetime(np.array(df_ws['Date and Time']))
    ht = np.array(df_ws['height'])

    # calculate u and v from speed and direction
    speed = df_ws['value'] * 1.94384  # convert wind speeds from m/s to knots
    direction = df_direction['value']

    u, v = wind_spddir_to_uv(np.array(speed), np.array(direction))

    return tm, ht, u, v, speed


def wind_spddir_to_uv(wspd, wdir):
    """
    calculate the u and v wind components from wind speed and direction
    Input:
        wspd: wind speed
        wdir: wind direction
    Output:
        u: u wind component
        v: v wind component
    """

    rad = 4.0 * np.arctan(1) / 180.
    u = -wspd * np.sin(rad * wdir)
    v = -wspd * np.cos(rad * wdir)

    return u, v


def main(args):
    save_file1 = args.save_file1
    save_file3 = args.save_file3
    sodar_dir = '/home/coolgroup/MetData/CMOMS/sodar/daily/'
    os.makedirs(os.path.dirname(save_file1), exist_ok=True)
    os.makedirs(os.path.dirname(save_file3), exist_ok=True)

    # define files for the 3 most recent days of data
    today = dt.datetime.now()
    days = [2, 1]
    fext = []
    for d in days:
        fext.append((today - dt.timedelta(days=d)).strftime('%Y%m%d'))
    fext.append(today.strftime('%Y%m%d'))

    # combine data from files for the most current 3 days into one dataframe
    df3day = pd.DataFrame()
    wind_height = ['30m', '40m', '50m', '60m', '80m', '100m', '120m', '140m', '160m', '180m', '200m']

    # check if there is a file for today, if not don't make a plot
    try:
        glob.glob(sodar_dir + '*sodar.' + today.strftime('%Y%m%d') + '.dat')[0]
    except IndexError:
        print('No SODAR file for today - skipping plot')
        quit()

    for f in fext:
        try:
            fname = glob.glob(sodar_dir + '*sodar.' + f + '.dat')[0]
        except IndexError:
            continue
        df1 = pd.read_csv(fname)

        # for each wind height, grab all of the columns of data, drop all rows where quality is <60, un-pivot the
        # dataframe, and append the remaining data to the main dataframe
        for wh in wind_height:
            cols = [x for x in df1.columns.tolist() if wh == x.split(' ')[0]]  # get all the columns for one height
            cols.append('Date and Time')
            dfc = df1[cols]
            qc_colname = '{} Quality'.format(wh)
            dfc = dfc[dfc[qc_colname] >= 60]  # drop all rows where quality is <60
            dfc = dfc.drop(columns=qc_colname)  # drop the column for quality since it's no longer needed
            dfc = dfc.melt(id_vars='Date and Time')  # un-pivot the dataframe
            df3day = df3day.append(dfc)  # append to the main dataframe

    if len(df3day) > 0:
        # add height as a column in the dataframe
        df3day['height'] = df3day['variable'].map(lambda x: int(x.split(' ')[0][:-1]))
        df3day['tm'] = pd.to_datetime(df3day['Date and Time'])

        # get the data for the full 3 days, define the color limits and subsetting for the barbs, and create a plot
        tm3, ht3, u3, v3, wspd3 = format_data(df3day)
        color_lims3 = define_color_lims(wspd3)
        sub3 = define_barb_subset(df3day)

        plot_barbs(tm3, ht3, u3, v3, sub3, color_lims3, '3-day', save_file3)

        # get the data for the current day only, define the color limits, do not subset the barbs, and create a plot
        df1day = df3day.loc[df3day['tm'] >= pd.to_datetime(today.strftime('%Y-%m-%d'))]
        tm1, ht1, u1, v1, wspd1 = format_data(df1day)
        color_lims1 = define_color_lims(wspd1)
        sub1 = 1

        plot_barbs(tm1, ht1, u1, v1, sub1, color_lims1, '1-day', save_file1)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-s1', '--save_file1',
                            dest='save_file1',
                            type=str,
                            help='Full file path to save directory and save filename for 1 day plot')

    arg_parser.add_argument('-s3', '--save_file3',
                            dest='save_file3',
                            type=str,
                            help='Full file path to save directory and save filename for 3 day plot')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
