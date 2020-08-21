#!/usr/bin/env python

"""
Author: Lori Garzio on 8/17/2020
Last modified: 8/19/2020
Creates hourly plots of RU-WRF 4.1 output variables: hourly snowfall, and daily accumulated snowfall.
The plots are used to populate RUCOOL's RU-WRF webpage:
https://rucool.marine.rutgers.edu/data/meteorological-modeling/ruwrf-mesoscale-meteorological-model-forecast/
"""

import argparse
import numpy as np
import os
import glob
import sys
import time
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def plt_snow(nc, model, figname, snowtype, ncprev=None):
    """
    Create filled contour surface maps of hourly and accumulated snowfall
    :param nc: netcdf file
    :param model: the model version that is being plotted, e.g. 3km or 9km
    :param figname: full file path to save directory and save filename
    :param snowtype: plot type to make, e.g. 'acc' (accumulated) or 'hourly'
    :param ncprev: optional, netcdf file from the previous model hour to calculate hourly snowfall
    """
    # SNOWNC = water equivalent of total accumulated snowfall
    snow = nc['SNOWNC']
    plot_types = ['full_grid', 'bight']

    if snowtype == 'acc':
        new_fname = 'acc{}'.format(figname.split('/')[-1])
        figname = '/{}/{}'.format(os.path.join(*figname.split('/')[0:-1]), new_fname)
        color_label = 'Total Accumulated Snow 10:1 (in)'

        # specify colorbar level demarcations and contour levels
        levels = [-0.5, 0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20]
        cbar_ticks = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20]
        contour_list = [0, 2, 4, 6, 10, 20]

    elif snowtype == 'hourly':
        color_label = 'Hourly Snowfall 10:1 (in)'
        # calculate hourly snowfall for every model hour by subtracting the snowfall for the previous hour from
        # the snowfall for the current hour
        if ncprev is not None:
            snowprev = ncprev['SNOWNC']
            snow = np.subtract(np.squeeze(snow), np.squeeze(snowprev))
        else:
            snow = snow

            # specify colorbar level demarcations and contour levels
        levels = [-0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]
        cbar_ticks = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]
        contour_list = [0, 1, 2, 3, 4, 5]

    for pt in plot_types:
        if pt == 'full_grid':  # subset the entire grid
            snow_sub, ax_lims = cf.subset_grid(snow, model)
        else:  # subset just NY Bight
            new_fname = 'bight_{}'.format(figname.split('/')[-1])
            figname = '/{}/{}'.format(os.path.join(*figname.split('/')[0:-1]), new_fname)
            snow_sub, ax_lims = cf.subset_grid(snow, 'bight')

        fig, ax, lat, lon = cf.set_map(snow_sub)

        # convert mm to in then multiply by 10 since the output is water equivalent
        snow_in = snow_sub * 0.0393701 * 10

        # add text to the bottom of the plot
        cf.add_text(ax, nc.SIMULATION_START_DATE, nc.time_coverage_start, model)

        cf.add_map_features(ax, ax_lims)

        snow_colors = [
            "#ffffff",
            "#deebf7",
            "#c6dbef",
            "#9ecae1",
            "#6baed6",
            "#4292c6",
            "#2171b5",
            "#08519c",
            "#9e9ac8",
            "#8c6bb1",
            "#88419d",
            "#810f7c",
            "#4d004b"
        ]
        snow_colormap = mpl.colors.ListedColormap(snow_colors)

        # plot data
        pf.plot_contourf(fig, ax, color_label, lon, lat, snow_in, levels, snow_colormap, color_label, var_min=None,
                         var_max=None, normalize='yes', cbar_ticks=cbar_ticks)

        # add contours
        pf.add_contours(ax, lon, lat, snow_in, contour_list)

        plt.savefig(figname, dpi=200)
        plt.close()


def main(args):
    start_time = time.time()
    wrf_procdir = args.wrf_dir
    save_dir = args.save_dir

    if wrf_procdir.endswith('/'):
        ext = '*.nc'
    else:
        ext = '/*.nc'
    files = sorted(glob.glob(wrf_procdir + ext))

    # get the model version (3km or 9km) from the filename
    f0 = files[0]
    model_ver = f0.split('/')[-1].split('_')[1]  # 3km or 9km
    os.makedirs(save_dir, exist_ok=True)

    for i, f in enumerate(files):
        fname = f.split('/')[-1].split('.')[0]
        splitter = fname.split('/')[-1].split('_')
        ncfile = xr.open_dataset(f, mask_and_scale=False)
        sfile = cf.save_filepath(save_dir, 'snow', splitter)

        # plot hourly and accumulated snowfall (each .nc file contains accumulated snowfall)
        # plot accumulated snowfall
        plt_snow(ncfile, model_ver, sfile, 'acc')

        # plot hourly snowfall
        if i > 0:
            fprev = files[i - 1]
            nc_prev = xr.open_dataset(fprev, mask_and_scale=False)
        else:
            nc_prev = None
        plt_snow(ncfile, model_ver, sfile, 'hourly', nc_prev)

    print('')
    print('Script run time: {} minutes'.format(round(((time.time() - start_time) / 60), 2)))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Plot hourly snowfall, and daily accumulated snowfall',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-wrf_dir',
                            dest='wrf_dir',
                            default='/home/coolgroup/ru-wrf/real-time/v4.1_parallel/processed/3km/20200101',
                            type=str,
                            help='Full directory path to subset WRF netCDF files.')

    arg_parser.add_argument('-save_dir',
                            dest='save_dir',
                            default='/home/coolgroup/ru-wrf/real-time/v4.1_parallel/scripts/webfigs/3km',
                            type=str,
                            help='Full directory path to save output plots.')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
