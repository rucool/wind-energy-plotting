#!/usr/bin/env python

"""
Author: Lori Garzio on 6/15/2020
Last modified: 7/28/2021
Creates profile plots of wind speed from RU-WRF 4.1 at native model levels for hours 1-24 and 25-48 at 2 locations:
1) NYSERDA North LiDAR Buoy
2) NYSERDA South LiDAR Buoy
"""

import argparse
import numpy as np
import os
import glob
import pandas as pd
import datetime as dt
import sys
import xarray as xr
import matplotlib.pyplot as plt
import functions.common as cf
plt.rcParams.update({'font.size': 14})  # set the font size for all plots


def append_model_data(nc_filepath, buoy_locations, data_dict):
    """
    Append model data from a specific lat/lon to data dictionary
    nc_filepath: file path to NetCDF file containing data
    buoy_locations: dictionary containing buoy latitude and longitude
    data_dict: dictionary with keys 't', 'height', and 'ws' to which data are appended
    """
    ncfile = xr.open_dataset(nc_filepath, mask_and_scale=False)

    lats = ncfile['XLAT']
    lons = ncfile['XLONG']

    # Find the closest model point
    # calculate the sum of the absolute value distance between the model location and buoy location
    a = abs(lats - buoy_locations['lat']) + abs(lons - buoy_locations['lon'])

    # find the indices of the minimum value in the array calculated above
    i, j = np.unravel_index(a.argmin(), a.shape)

    # grab the data at that location/index
    height = np.squeeze(ncfile['height_agl'])[:, i, j]
    u = np.squeeze(ncfile['u'])[:, i, j]
    v = np.squeeze(ncfile['v'])[:, i, j]

    # calculate wind speed (m/s) from u and v
    ws = cf.wind_uv_to_spd(u, v)

    # append data to array
    data_dict['t'] = np.append(data_dict['t'], ncfile['Time'].values)
    if len(data_dict['height']) > 0:
        data_dict['height'] = np.vstack((data_dict['height'], height.values))
        data_dict['ws'] = np.vstack((data_dict['ws'], ws.values))
    else:
        data_dict['height'] = height.values
        data_dict['ws'] = ws.values


def plot_wndsp_profile(data_dict, hour_info, plt_ttl, model_init_dt, save_filepath, hmax=None):
    """
    Profile plots of wind speeds, colored by time
    data_dict: dictionary containing wind speed data at multiple heights
    hour_info: dictionary containing the minimum and maximum hours being plotted
    plt_ttl: plot title
    model_init_dt: model initialized date string
    save_filepath: full file path to save directory and save filename
    hmax: optional, maximum height to plot
    """
    n = len(data_dict['t'])
    colors = plt.cm.rainbow(np.linspace(0, 1, n))

    # specify the colorbar tick labels
    if hour_info['max_hour'] == 24:
        cbar_labs = ['01:00', '06:00', '12:00', '18:00', '24:00']
    else:
        cbar_labs = ['25:00', '30:00', '36:00', '42:00', '48:00']

    fig, ax = plt.subplots(figsize=(8, 9))
    plt.subplots_adjust(right=0.88, left=0.15)
    plt.grid()
    for i in range(n):
        if hmax is not None:
            height_ind = np.where(data_dict['height'][i] <= hmax)
            ax.plot(data_dict['ws'][i][height_ind], data_dict['height'][i][height_ind], c=colors[i])
        else:
            ax.plot(data_dict['ws'][i], data_dict['height'][i], c=colors[i])
        if i == (n - 1):
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=None, cmap='rainbow'),
                                ax=ax, orientation='vertical', fraction=0.09, pad=0.03, label='Model Forecast Hour (GMT)')
            cbar.set_ticks([0, .25, .5, .75, 1])
            cbar.ax.set_yticklabels(cbar_labs)
            ax.set_xlabel('Wind Speed (m/s)')
            ax.set_ylabel('Height (m)')
            ax.set_title(plt_ttl)
            if hmax is not None:
                ax.set_xlim(0, 30)
            else:
                ax.set_xlim(0, 40)

            # add text to the bottom of the plot
            insert_text1 = 'RU-WRF (v4.1) 3km Model: Initialized {}'.format(model_init_dt)
            ax.text(.5, -.12, insert_text1, size=10, transform=ax.transAxes)

            plt.savefig(save_filepath, dpi=200)
            plt.close()


def main(args):
    wrf_dir = args.wrf_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # locations of NYSERDA LIDAR buoys
    nyserda_buoys = dict(nyserda_north=dict(lon=-72.7173, lat=39.9686),
                         nyserda_south=dict(lon=-73.4295, lat=39.5465))

    # for each NYSERDA buoy location, append wind speeds for each model run for hours 1-24 and 25-48
    files = sorted(glob.glob(wrf_dir + '*.nc'))
    run1 = [x for x in files if 0 < int(x.split('.nc')[0][-3:]) < 25]  # plot hours 1-24
    run2 = [x for x in files if 24 < int(x.split('.nc')[0][-3:]) < 49]  # plot hours 25-48
    run_dict = dict(run1=dict(file_lst=run1, min_hour=1, max_hour=24),
                    run2=dict(file_lst=run2, min_hour=25, max_hour=48))

    for run, info in run_dict.items():
        # initialize empty dictionaries for each buoy location
        data = dict(nyserda_north=dict(t=np.array([], dtype='datetime64[ns]'), height=np.array([]), ws=np.array([])),
                    nyserda_south=dict(t=np.array([], dtype='datetime64[ns]'), height=np.array([]), ws=np.array([])))
        for f in info['file_lst']:
            for nb, bloc, in nyserda_buoys.items():
                if 'north' in nb:
                    append_model_data(f, bloc, data['nyserda_north'])
                else:
                    append_model_data(f, bloc, data['nyserda_south'])

        # plot data for each NYSERDA buoy location
        if run == 'run1':
            mdate = pd.to_datetime(data['nyserda_north']['t'][0])
        else:
            mdate = pd.to_datetime(data['nyserda_north']['t'][0]) - dt.timedelta(days=1)
        datestr = mdate.strftime('%Y%m%d')
        datestr2 = mdate.strftime('%Y-%m-%d')
        init_dt_str = '00Z{}'.format(mdate.strftime('%d%b%Y'))

        for loc, d in data.items():
            if 'north' in loc:
                buoy = 'NYSERDA North'
                buoy_code = 'NYNE05'
            elif 'south' in loc:
                buoy = 'NYSERDA South'
                buoy_code = 'NYSE06'
            ttl = 'RU-WRF 4.1 Wind Profiles at {}\n{}: Hours {:03d}-{:03d}'.format(buoy, datestr2, info['min_hour'], info['max_hour'])

            # plot entire profile
            sf = 'WRF_wsprofiles_{}_{}_H{:03d}-{:03d}.png'.format(buoy_code, datestr, info['min_hour'], info['max_hour'])
            sfpath = os.path.join(save_dir, sf)
            plot_wndsp_profile(d, info, ttl, init_dt_str, sfpath)

            # plot profile 0-1000m
            sf = 'WRF_wsprofiles_{}_{}_H{:03d}-{:03d}_abl.png'.format(buoy_code, datestr, info['min_hour'], info['max_hour'])
            sfpath = os.path.join(save_dir, sf)
            plot_wndsp_profile(d, info, ttl, init_dt_str, sfpath, 1200)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-wd', '--wrf_dir',
                            dest='wrf_dir',
                            default='/home/coolgroup/ru-wrf/real-time/v4.1_parallel/processed/modlevs/3km/20200615',
                            type=str,
                            help='Full path to subset WRF native model level directory')

    arg_parser.add_argument('-sd', '--save_dir',
                            dest='save_dir',
                            type=str,
                            help='Full path to save directory')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
