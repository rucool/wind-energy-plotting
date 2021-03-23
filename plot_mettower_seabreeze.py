#!/usr/bin/env python

"""
Author: Lori Garzio on 3/23/2021
Last Modified: 3/23/2021
Create timeseries plots of wind speed and direction from the Tuckerton met tower data at 12m.
"""

import os
import datetime as dt
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import functions.common as cf
pd.set_option('display.width', 320, "display.max_columns", 15)  # for display in pycharm console
plt.rcParams.update({'font.size': 12})


def daterange(date1, date2):
    for n in range(int((date2 - date1).days)+1):
        yield date1 + dt.timedelta(n)


def format_date_axis(axis, figure):
    datef = mdates.DateFormatter('%Y-%m-%d')
    axis.xaxis.set_major_formatter(datef)
    figure.autofmt_xdate()


def main(stime, etime, plotting_interval, hts, sDir):
    #met_dir = '/home/coolgroup/MetData/CMOMS/surface/daily/'  # on server
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

        # seabreeze file for the 12m met tower algorithm
        seabreeze_id12 = '/Users/lgarzio/Documents/rucool/bpu/wrf/seabreeze_id/seabreezes12m_20160609-20210315.csv'
        sbid12 = pd.read_csv(seabreeze_id12)
        sbid12['start_timeEST'] = pd.to_datetime(sbid12.start_timeEST)
        sbid12['end_timeEST'] = pd.to_datetime(sbid12.end_timeEST)

        # break into intervals (e.g. monthly, weekly)
        if plotting_interval == 'monthly':
            # create list of start and end dates
            start_dates = [stime]
            end_dates = []
            ts1 = stime
            while ts1 <= etime:
                ts2 = ts1 + dt.timedelta(days=1)
                if ts2.month != ts1.month:
                    if ts2 <= etime:
                        start_dates.append(ts2)
                    end_dates.append(ts1)
                ts1 = ts2

        elif plotting_interval == 'weekly':
            dtrange = pd.date_range(stime, etime, freq='W')
            start_dates = []
            end_dates = []
            for dr_idx, dr in enumerate(dtrange):
                if dr_idx == 0:  # first iteration
                    start_dates.append(stime)
                    end_dates.append(dr - dt.timedelta(minutes=1))
                elif dr_idx == len(dtrange) - 1:  # last iteration
                    end_dates.append(dr - dt.timedelta(minutes=1))
                    start_dates.append(dtrange[dr_idx - 1])
                    end_dates.append(etime)
                    start_dates.append(dr)
                else:
                    end_dates.append(dr - dt.timedelta(minutes=1))
                    start_dates.append(dtrange[dr_idx - 1])

        for sd, ed in zip(start_dates, end_dates):
            if plotting_interval == 'monthly':
                endtime = ed + dt.timedelta(hours=23, minutes=59)
            else:
                endtime = ed
            df_interval = df[(df.DateEST >= sd) & (df.DateEST <= endtime)]

            # get seabreeze intervals
            sbid12_interval = sbid12[(sbid12.start_timeEST >= sd) & (sbid12.start_timeEST <= endtime)]

            for wh in hts:
                fig, ax = plt.subplots(figsize=(12, 6))
                cols = [x for x in df.columns.tolist() if wh == x.split(' ')[0]]
                ax.plot(df_interval.DateEST, df_interval['avg(mph)'], c='tab:orange')
                ax.set_ylabel('Wind Speed (mph)', c='tab:orange')
                ax2 = ax.twinx()
                ax2.plot(df_interval.DateEST, df_interval['12m Wind Direction'], marker='.', ms=3, c='tab:blue',
                         ls='None')
                ax2.set_ylabel('Wind Direction', c='tab:blue')
                ax2.set_ylim([0, 360])

                # add horizontal line delimiting seabreeze criteria for wind direction
                ax2.axhline(y=95, ls='-', c='k', zorder=1)
                ax2.axhline(y=195, ls='-', c='k', zorder=1)

                ax.set_xlabel('Date (EST)')

                # shade each time range for seabreezes
                for idx, row in sbid12_interval.iterrows():
                    ax.axvspan(row.start_timeEST, row.end_timeEST, facecolor='tab:green', label='12m algorithm', alpha=0.3)

                if plotting_interval == 'monthly':
                    savefile = 'mettower_wspd_dir_{}_{}'.format(sd.strftime('%Y%m'), wh)
                    ttl = '{}: {}'.format(sd.strftime('%b-%Y'), wh)
                    format_date_axis(ax, fig)
                else:
                    savefile = 'mettower_wspd_dir_{}_{}'.format(sd.strftime('%Y%m%d'), wh)
                    ttl = '{}: {}'.format(sd.strftime('%b %d %Y'), wh)
                    xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
                    ax.xaxis.set_major_formatter(xfmt)
                    #ax.xaxis.set_major_locator(plt.MaxNLocator(9))  # reduce number of x ticks

                plt.title(ttl, fontsize=12)
                plt.savefig(os.path.join(sDir, savefile), dpi=200)
                plt.close()


if __name__ == '__main__':
    start_time = dt.datetime(2020, 1, 1)  # first date: dt.datetime(2015, 5, 13)
    end_time = dt.datetime(2020, 12, 31)
    plt_int = 'weekly'  # 'monthly' or 'weekly'
    heights = ['12m']
    save_dir = '/Users/lgarzio/Documents/rucool/bpu/wrf/seabreeze_id/mettower_imgs'  # on local machine
    #save_dir = '/www/home/lgarzio/public_html'  # on server
    main(start_time, end_time, plt_int, heights, save_dir)
