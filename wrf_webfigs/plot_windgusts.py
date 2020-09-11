#!/usr/bin/env python

"""
Author: Lori Garzio on 9/11/2020
Last modified: 9/11/2020
Creates hourly plots of RU-WRF 4.1 calculated wind gusts. The plots are used to populate RUCOOL's RU-WRF webpage:
https://rucool.marine.rutgers.edu/data/meteorological-modeling/ruwrf-mesoscale-meteorological-model-forecast/
"""

import argparse
import numpy as np
import os
import glob
import sys
import time
import xarray as xr
import matplotlib.pyplot as plt
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def plt_gust(nc, model, figname, lease_areas):
    """
    Create pseudocolor surface maps of wind gust.
    :param nc: netcdf file
    :param model: the model version that is being plotted, e.g. 3km or 9km
    :param figname: full file path to save directory and save filename
    :param lease_areas: dictionary containing lat/lon coordinates for wind energy lease area polygon
    """
    gust = nc['WINDGUST']

    color_label = 'Wind Gust (knots)'

    plot_types = ['full_grid', 'bight']
    for pt in plot_types:
        if pt == 'full_grid':  # subset the entire grid
            gust_sub, ax_lims = cf.subset_grid(gust, model)
        else:  # subset just NY Bight
            new_fname = 'bight_{}'.format(figname.split('/')[-1])
            figname = '/{}/{}'.format(os.path.join(*figname.split('/')[0:-1]), new_fname)
            gust_sub, ax_lims = cf.subset_grid(gust, 'bight')

        fig, ax, lat, lon = cf.set_map(gust_sub)

        # add text to the bottom of the plot
        cf.add_text(ax, nc.SIMULATION_START_DATE, nc.time_coverage_start, model)

        cf.add_map_features(ax, ax_lims)

        # pf.add_lease_area_polygon(ax, lease_areas, 'magenta')

        # convert wind gust from m/s to knots
        gust_sub = np.squeeze(gust_sub.values) * 1.94384

        # convert nans to zero
        gust_sub[np.isnan(gust_sub)] = 0

        # add contours
        contour_list = [22, 34, 48, 64]
        pf.add_contours(ax, lon, lat, gust_sub, contour_list)

        # plot data
        # pcolormesh: coarser resolution, shows the actual resolution of the model data
        pf.plot_pcolormesh(fig, ax, color_label, lon, lat, gust_sub, 0, 40, 'BuPu', color_label)

        # contourf: smooths the resolution of the model data, plots are less pixelated
        # levels = np.arange(0, 40.1, .1)
        # pf.plot_contourf(fig, ax, color_label, lon, lat, gust_sub, levels, 'BuPu', color_label, var_min=0, var_max=40,
        #                  normalize='no', cbar_ticks=np.linspace(0, 40, 9))

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
        sfile = cf.save_filepath(save_dir, 'gusts', splitter)
        plt_gust(ncfile, model_ver, sfile, la_polygon)

    print('')
    print('Script run time: {} minutes'.format(round(((time.time() - start_time) / 60), 2)))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Plot wind gusts',
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
