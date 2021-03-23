#! /usr/bin/env python

"""
Author: Lori Garzio on 3/23/2021
Last modified: 3/23/2021
"""

import numpy as np
import pandas as pd
import datetime as dt


def seabreeze_id_df(df, dt_colname, wd_colname, timezone):
    """
    Identifies seabreezes from wind direction data stored as a dataframe
    :param df: dataframe containing data
    :param dt_colname: column name for date
    :param wd_colname: column name for wind direction
    :param timezone: e.g. EST
    :return:
    """
    seabreezes = []  # master list to keep track of seabreezes lasting >= 4 hours
    sb = []  # initial list to track each potential seabreeze
    for i, row in df.iterrows():
        t1 = row[dt_colname]
        winddir_test = 95 <= row[wd_colname] <= 195
        if len(sb) == 0:
            # for the beginning of the seabreeze, check that the time is between 10am and and 4pm EST
            if np.logical_and(10 <= t1.hour < 16, winddir_test):
                sb.append(row[dt_colname])
        else:
            if pd.isnull(row[wd_colname]):  # keep going if wind direction is missing
                continue
            if winddir_test:
                # check that this timestamp is <=1 hour from the previous timestamp
                # this will catch gaps in the data >1 hour
                if t1 - sb[-1] <= dt.timedelta(hours=1):
                    if len(sb) == 1:
                        sb.append(row[dt_colname])
                    else:  # replace second timestamp with this one
                        sb[-1] = row[dt_colname]
                else:
                    # if the gap is >1 hour, check that the interval already identified is >= 4 hours
                    delta = sb[-1] - sb[0]
                    if delta >= dt.timedelta(hours=4):
                        # calculate seabreeze duration in hours
                        dur_hours = (delta.days * 24) + (delta.seconds / 3600)
                        sb.append(round(dur_hours, 2))
                        # append to master seabreeze list, then start over
                        seabreezes.append(sb)
                        sb = []  # then start over
                    else:
                        sb = []
            else:
                if len(sb) == 2:  # if there are 2 timestamps, check if the interval is >= 4 hours
                    delta = sb[-1] - sb[0]
                    if delta >= dt.timedelta(hours=4):
                        # calculate seabreeze duration in hours
                        dur_hours = (delta.days * 24) + (delta.seconds / 3600)
                        sb.append(round(dur_hours, 2))
                        # append to master seabreeze list, then start over
                        seabreezes.append(sb)
                        sb = []  # then start over
                    else:  # if false, don't append to master seabreeze list
                        sb = []
                else:  # if there's only 1 timestamp, start over
                    sb = []

    cols = ['start_time{}'.format(timezone), 'end_time{}'.format(timezone), 'duration_hours']
    sb_df = pd.DataFrame(seabreezes, columns=cols)
    return sb_df
