#!/usr/bin/env python

"""
Author: Lori Garzio on 3/15/2021
Last Modified: 3/23/2021
Create timeseries plots of wind speed and direction from the Tuckerton SODAR data at heights defined by the user.
"""

import os
import datetime as dt
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
    #sodar_dir = '/home/coolgroup/MetData/CMOMS/sodar/daily/'  # on server
    sodar_dir = '/Users/lgarzio/Documents/rucool/bpu/wrf/sodar_plotting/data3/'  # on local machine

    # find the file extensions for the date range of interest
    fext = []
    for tm in daterange(stime, etime):
        fext.append(tm.strftime('%Y%m%d'))

    # combine data from files into one dataframe
    df = pd.DataFrame()
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
        for wh in hts:
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

        # seabreeze file for the 100m algorithm
        seabreeze_id100 = '/Users/lgarzio/Documents/rucool/bpu/wrf/seabreeze_id/seabreezes100m_20150513-20210315.csv'
        sbid100 = pd.read_csv(seabreeze_id100)
        sbid100['start_timeEST'] = pd.to_datetime(sbid100.start_timeEST)
        sbid100['end_timeEST'] = pd.to_datetime(sbid100.end_timeEST)

        # seabreeze file for the 30m algorithm
        seabreeze_id30 = '/Users/lgarzio/Documents/rucool/bpu/wrf/seabreeze_id/seabreezes30m_20150513-20210315.csv'
        sbid30 = pd.read_csv(seabreeze_id30)
        sbid30['start_timeEST'] = pd.to_datetime(sbid30.start_timeEST)
        sbid30['end_timeEST'] = pd.to_datetime(sbid30.end_timeEST)

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
            # savedf = 'sodar_wspd_dir_{}.csv'.format(sd.strftime('%Y%m%d'))
            # df_interval.to_csv(os.path.join(sDir, savedf), index=False)

            # get seabreeze intervals
            sbid100_interval = sbid100[(sbid100.start_timeEST >= sd) & (sbid100.start_timeEST <= endtime)]
            sbid30_interval = sbid30[(sbid30.start_timeEST >= sd) & (sbid30.start_timeEST <= endtime)]

            for wh in hts:
                fig, ax = plt.subplots(figsize=(12, 6))
                cols = [x for x in df.columns.tolist() if wh == x.split(' ')[0]]
                ax.plot(df_interval.DateEST, df_interval[cols[-1]], c='tab:orange')
                ax.set_ylabel('Wind Speed', c='tab:orange')
                ax2 = ax.twinx()
                ax2.plot(df_interval.DateEST, df_interval[cols[0]], marker='.', ms=3, c='tab:blue', ls='None')
                ax2.set_ylabel('Wind Direction', c='tab:blue')
                ax2.set_ylim([0, 360])

                # add horizontal line delimiting seabreeze criteria for wind direction
                ax2.axhline(y=95, ls='-', c='k', zorder=1)
                ax2.axhline(y=195, ls='-', c='k', zorder=1)

                ax.set_xlabel('Date (EST)')

                # shade each time range for seabreezes
                for idx, row in sbid30_interval.iterrows():
                    ax.axvspan(row.start_timeEST, row.end_timeEST, facecolor='tab:blue', label='30m algorithm', alpha=0.3)

                for idx, row in sbid100_interval.iterrows():
                    ax.axvspan(row.start_timeEST, row.end_timeEST, facecolor='gray', label='100m algorithm', alpha=0.5)

                #ax.legend(fontsize=8)

                if plotting_interval == 'monthly':
                    savefile = 'sodar_wspd_dir_{}_{}'.format(sd.strftime('%Y%m'), wh)
                    ttl = '{}: {}\nblue shading 30m algorithm, gray shading 100m algorithm'.format(sd.strftime('%b-%Y'), wh)
                    format_date_axis(ax, fig)
                else:
                    savefile = 'sodar_wspd_dir_{}_{}'.format(sd.strftime('%Y%m%d'), wh)
                    ttl = '{}: {}\nblue shading 30m algorithm, gray shading 100m algorithm'.format(sd.strftime('%b %d %Y'), wh)
                    xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
                    ax.xaxis.set_major_formatter(xfmt)
                    #ax.xaxis.set_major_locator(plt.MaxNLocator(9))  # reduce number of x ticks

                plt.title(ttl, fontsize=12)
                plt.savefig(os.path.join(sDir, '{}'.format(wh), savefile), dpi=200)
                plt.close()


if __name__ == '__main__':
    start_time = dt.datetime(2020, 1, 1)  # first date: dt.datetime(2015, 5, 13)
    end_time = dt.datetime(2020, 12, 31)
    plt_int = 'monthly'  # 'monthly' or 'weekly'
    heights = ['30m', '100m']
    save_dir = '/Users/lgarzio/Documents/rucool/bpu/wrf/sodar_plotting/seabreeze_figs'  # on local machine
    #save_dir = '/www/home/lgarzio/public_html'  # on server
    main(start_time, end_time, plt_int, heights, save_dir)
