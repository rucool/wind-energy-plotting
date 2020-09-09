#!/usr/bin/env python

"""
Author: Lori Garzio on 5/28/2020
Last modified: 9/9/2020
Creates hourly plots of RU-WRF 4.1 output variables: hourly rainfall + sea level pressure, and daily accumulated
rainfall. The plots are used to populate RUCOOL's RU-WRF webpage:
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


def plt_rain(nc, model, figname, raintype, lease_areas, ncprev=None):
    """
    Create filled contour surface maps of hourly and accumulated rain
    :param nc: netcdf file
    :param model: the model version that is being plotted, e.g. 3km or 9km
    :param figname: full file path to save directory and save filename
    :param raintype: plot type to make, e.g. 'acc' (accumulated) or 'hourly'
    :param lease_areas: dictionary containing lat/lon coordinates for wind energy lease area polygon
    :param ncprev: optional, netcdf file from the previous model hour to calculate hourly rainfall
    """
    # RAINNC = total accumulated rainfall
    rn = nc['RAINNC']
    plot_types = ['full_grid', 'bight']

    if raintype == 'acc':
        new_fname = 'acc{}'.format(figname.split('/')[-1])
        figname = '/{}/{}'.format(os.path.join(*figname.split('/')[0:-1]), new_fname)
        color_label = 'Total Accumulated Precipitation (in)'
        title = color_label
        slp = None  # don't plot sea level pressure for accumulated rain maps
    elif raintype == 'hourly':
        color_label = 'Hourly Precipitation (in)'
        title = '{}, Sea Level Pressure (mb)'.format(color_label)
        slp = nc['SLP']
        # calculate hourly rainfall for every model hour by subtracting the rainfall for the previous hour from
        # the rainfall for the current hour
        if ncprev is not None:
            preciprev = ncprev['RAINNC']
            rn = np.subtract(np.squeeze(rn), np.squeeze(preciprev))
        else:
            rn = rn

    for pt in plot_types:
        if pt == 'full_grid':  # subset the entire grid
            if slp is not None:
                slp_sub, __ = cf.subset_grid(slp, model)
            rn_sub, ax_lims = cf.subset_grid(rn, model)
        else:  # subset just NY Bight
            new_fname = 'bight_{}'.format(figname.split('/')[-1])
            figname = '/{}/{}'.format(os.path.join(*figname.split('/')[0:-1]), new_fname)
            if slp is not None:
                slp_sub, __ = cf.subset_grid(slp, 'bight')
            rn_sub, ax_lims = cf.subset_grid(rn, 'bight')

        fig, ax, lat, lon = cf.set_map(rn_sub)

        # add text to the bottom of the plot
        cf.add_text(ax, nc.SIMULATION_START_DATE, nc.time_coverage_start, model)

        cf.add_map_features(ax, ax_lims)

        # pf.add_lease_area_polygon(ax, lease_areas, 'magenta')

        # convert mm to inches
        rn_sub = rn_sub * 0.0394

        # modified NWS colormap, from http://jjhelmus.github.io/blog/2013/09/17/plotting-nsw-precipitation-data/
        nws_precip_colors = [
            "#fdfdfd",  # 0.01 - 0.10 inches
            "#019ff4",  # 0.10 - 0.25 inches
            "#0300f4",  # 0.25 - 0.50 inches
            "#02fd02",  # 0.50 - 0.75 inches
            "#01c501",  # 0.75 - 1.00 inches
            "#008e00",  # 1.00 - 1.50 inches
            "#fdf802",  # 1.50 - 2.00 inches
            "#e5bc00",  # 2.00 - 2.50 inches
            "#fd9500",  # 2.50 - 3.00 inches
            "#fd0000",  # 3.00 - 4.00 inches
            "#d40000",  # 4.00 - 5.00 inches
            "#bc0000",  # 5.00 - 6.00 inches
            "#f800fd",  # 6.00 - 8.00 inches
            "#9854c6",  # 8.00 - 10.00 inches
            "#4B0082"  # 10.00+
        ]
        precip_colormap = mpl.colors.ListedColormap(nws_precip_colors)

        # specify colorbar level demarcations
        levels = [0.01, 0.1, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10., 12.]

        # plot data
        pf.plot_contourf(fig, ax, title, lon, lat, rn_sub, levels, precip_colormap, color_label, var_min=None,
                         var_max=None, normalize='yes')

        # add slp as contours if provided
        if slp is not None:
            contour_list = [940, 944, 948, 952, 956, 960, 964, 968, 972, 976, 980, 984, 988, 992, 996, 1000, 1004, 1008,
                            1012, 1016, 1020, 1024, 1028, 1032, 1036, 1040]
            pf.add_contours(ax, lon, lat, slp_sub, contour_list)

        plt.savefig(figname, dpi=200)
        plt.close()


def main(args):
    start_time = time.time()
    wrf_procdir = args.wrf_dir
    save_dir = args.save_dir

    la_polygon = cf.extract_lease_areas()

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
        sfile = cf.save_filepath(save_dir, 'rain', splitter)

        # plot hourly and accumulated rainfall (each .nc file contains accumulated precipitation)
        # plot accumulated rainfall
        plt_rain(ncfile, model_ver, sfile, 'acc', la_polygon)

        # plot hourly rainfall
        if i > 0:
            fprev = files[i - 1]
            nc_prev = xr.open_dataset(fprev, mask_and_scale=False)
        else:
            nc_prev = None
        plt_rain(ncfile, model_ver, sfile, 'hourly', la_polygon, nc_prev)

    print('')
    print('Script run time: {} minutes'.format(round(((time.time() - start_time) / 60), 2)))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Plot hourly rainfall + sea level pressure, and daily accumulated rainfall',
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
