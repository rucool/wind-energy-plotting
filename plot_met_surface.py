#!/usr/bin/env python

"""
Author: Lori Garzio on 9/1/2020
Last Modified: 4/18/2021
Creates timeseries plots of wind speed, air temperature, and sea level pressure from RUCOOL's meteorological tower for
the three most recent days of available data.
These plots are used to populate RUCOOL's Coastal Metocean Monitoring Station webpage:
https://rucool.marine.rutgers.edu/data/meteorological-modeling/coastal-metocean-monitoring-station/
"""

import argparse
import datetime as dt
import os
import glob
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
pd.set_option('display.width', 320, "display.max_columns", 15)  # for display in pycharm console
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def format_date_axis(axis):
    datef = mdates.DateFormatter('%m-%d %H:%M')
    axis.xaxis.set_major_formatter(datef)


def format_plot(axis, date, ttl, ylabel, legend, ylims=None, yticks=None):
    axis.set_title(ttl, fontsize=14)
    axis.text(.81, -.12, 'rucool.marine.rutgers.edu', size=10, transform=axis.transAxes)
    axis.set_ylabel(ylabel, labelpad=10)
    axis.set_xlabel('Time (GMT)', labelpad=10)
    if ylims is not None:
        axis.set_ylim(ylims)
    if yticks is not None:
        plt.yticks(yticks)
    t0 = pd.to_datetime((date - dt.timedelta(days=2)).strftime('%Y%m%d'))
    t1 = pd.to_datetime((date + dt.timedelta(days=1)).strftime('%Y%m%d'))
    plt.xlim([t0, t1])
    plt.grid(color='lightgray')

    if legend == 'yes':
        axis.legend(loc='upper left', fontsize=10)


def wind_uv_to_spd(u, v):
    """
    Calculates the wind speed from the u and v wind components
    :param u: west/east direction (wind from the west is positive, from the east is negative)
    :param v: south/noth direction (wind from the south is positive, from the north is negative)
    :returns WSPD: wind speed calculated from the u and v wind components
    """
    WSPD = np.sqrt(np.square(u) + np.square(v))

    return WSPD


def main(args):
    save_dir = args.save_dir
    met_dir = '/home/coolgroup/MetData/CMOMS/surface_temp/surface'
    os.makedirs(save_dir, exist_ok=True)

    # find the most recent file
    last_file = sorted(glob.glob(os.path.join(met_dir, '*.csv')))[-1]
    last_file_day = dt.datetime.strptime(last_file.split('_')[-2], '%Y%m%d')

    # define files for the 3 most recent available days of data
    # today = dt.datetime.utcnow()
    int_days = [2, 1]
    fext = []
    for d in int_days:
        fext.append((last_file_day - dt.timedelta(days=d)).strftime('%Y%m%d'))
    fext.append(last_file_day.strftime('%Y%m%d'))

    # initialize empty dataframe to append data from all files
    df = pd.DataFrame()

    # # check if there is a file for today, if not don't make a plot
    # try:
    #     glob.glob(met_dir + '/*surface.' + today.strftime('%Y%m%d') + '.dat')[0]
    # except IndexError:
    #     print('No file for today - skipping plot')
    #     quit()

    for dd in fext:
        try:
            day_files = sorted(glob.glob(met_dir + '/RUT_' + dd + '*.csv'))
        except IndexError:
            continue

        for f in day_files:
            df1 = pd.read_csv(f)
            cols = ['time_stamp(utc)', 'avg(mph)', 'gust(mph)', 'sonic_u(cm/s)', 'sonic_v(cm/s)', '12m_air_temp(f)',
                    '2m_air_temp(f)', 'pressure']
            dfc = df1[cols]
            if len(dfc) > 0:
                df = df.append(dfc)

    tm = pd.to_datetime(np.array(df['time_stamp(utc)']))

    # plot windspeed
    # calculate sonic windspeed from u and v, and convert to m/s
    sonic_ws = wind_uv_to_spd(np.array(df['sonic_u(cm/s)']), np.array(df['sonic_v(cm/s)'])) * .01

    # convert avg and gusts fro mph to m/s
    avg_ws = np.array(df['avg(mph)']) * 0.44704
    gust = np.array(df['gust(mph)']) * 0.44704

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(tm, avg_ws, color='#33a02c', label='Wind Speed - Cup')  # green
    ax.plot(tm, gust, color='#fc8d62', label='Wind Gust - Cup')  # orange
    ax.plot(tm, sonic_ws, color='#1f78b4', label='Wind Speed - Sonic')  # blue

    title = 'Met Tower Winds at 12m'
    ylab = r'Wind Speed (m $\rm s^{-1}$)'

    if np.nanmax(gust) > 30:
        y_limits = [0, 40]
        y_ticks = np.linspace(0, 40, 11)
    elif np.nanmax(gust) > 20:
        y_limits = [0, 30]
        y_ticks = np.linspace(0, 30, 11)
    elif np.nanmax(gust) > 10:
        y_limits = [0, 20]
        y_ticks = np.linspace(0, 20, 11)
    else:
        y_limits = [0, 10]
        y_ticks = np.linspace(0, 10, 11)

    format_plot(ax, last_file_day, title, ylab, legend='yes', ylims=y_limits, yticks=y_ticks)

    format_date_axis(ax)

    figname = os.path.join(save_dir, 'bpu_windspeed.png')
    plt.savefig(figname, dpi=200)
    plt.close()

    # plot air temperature
    # convert F to C
    temp2 = (np.array(df['2m_air_temp(f)']) - 32) * 5/9
    temp12 = (np.array(df['12m_air_temp(f)']) - 32) * 5 / 9

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(tm, temp2, color='#d95f02', label='2m')  # orange
    ax.plot(tm, temp12, color='#7570b3', label='12m')  # purple

    title = 'Met Tower Air Temperature'
    ylab = 'Air Temperature ($^\circ$C)'
    ymin = np.floor(np.nanmin([np.nanmin(temp2), np.nanmin(temp12)])) - 1
    ymax = np.ceil(np.nanmax([np.nanmax(temp2), np.nanmax(temp12)])) + 1

    format_plot(ax, last_file_day, title, ylab, legend='yes', ylims=[ymin, ymax])

    format_date_axis(ax)

    figname = os.path.join(save_dir, 'bpu_temp.png')
    plt.savefig(figname, dpi=200)
    plt.close()

    # plot sea level pressure
    pressure = np.array(df['pressure'])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(tm, pressure, color='#1b9e77')  # green

    title = 'Met Tower Sea Level Pressure'
    ylab = 'Pressure (mb)'
    ymin = np.floor(np.nanmin(pressure) - 1.5)
    ymax = np.ceil(np.nanmax(pressure) + 1.5)

    format_plot(ax, last_file_day, title, ylab, legend='no', ylims=[ymin, ymax])

    format_date_axis(ax)

    figname = os.path.join(save_dir, 'bpu_pressure.png')
    plt.savefig(figname, dpi=200)
    plt.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Plot wind speeds, air temperature, and sea level pressure',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-s', '--save_dir',
                            dest='save_dir',
                            default='/www/cool/mrs/weather/cmoms_imagery',
                            type=str,
                            help='Full file path to save directory')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
