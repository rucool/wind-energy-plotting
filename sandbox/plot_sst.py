#!/usr/bin/env python

"""
Author: Lori Garzio on 8/21/2020
Last modified: 8/25/2020
Creates plot of H000 sea surface temperature from RU-WRF 4.1 subset files.
Note: This script requires fiona to be installed to work properly!
"""

import argparse
import numpy as np
import os
import glob
import sys
import time
import xarray as xr
import matplotlib.pyplot as plt
import cmocean as cmo
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def plt_sst(nc, model, figname):
    """
    Create a pcolor surface map of sea surface temperature
    :param nc: netcdf file
    :param model: the model version that is being plotted, e.g. 3km or 9km
    :param figname: full file path to save directory and save filename
    """
    sst = nc['SST']
    landmask = nc['LANDMASK']  # 1=land, 0=water
    lakemask = nc['LAKEMASK']  # 1=lake, 0=non-lake
    color_label = 'SST ($^\circ$C)'
    title = 'Sea Surface Temperature ($^\circ$C)'

    plot_types = ['full_grid', 'bight']  # plot the full grid and just NY Bight area
    for pt in plot_types:
        if pt == 'full_grid':  # subset the entire grid
            sst_sub, ax_lims = cf.subset_grid(sst, model)
            landmask_sub, __ = cf.subset_grid(landmask, model)
            lakemask_sub, __ = cf.subset_grid(lakemask, model)
        else:  # subset just NY Bight
            new_fname = 'bight_{}'.format(figname.split('/')[-1])
            figname = '/{}/{}'.format(os.path.join(*figname.split('/')[0:-1]), new_fname)
            sst_sub, ax_lims = cf.subset_grid(sst, 'bight')
            landmask_sub, __ = cf.subset_grid(landmask, 'bight')
            lakemask_sub, __ = cf.subset_grid(lakemask, 'bight')

        fig, ax, lat, lon = cf.set_map(sst_sub)

        # add text to the bottom of the plot
        cf.add_text(ax, nc.SIMULATION_START_DATE, nc.time_coverage_start, model)

        # convert values over land and lakes to nans
        ldmask = np.logical_and(landmask_sub == 1, landmask_sub == 1)
        sst_sub.values[ldmask] = np.nan

        lkmask = np.logical_and(lakemask_sub == 1, lakemask_sub == 1)
        sst_sub.values[lkmask] = np.nan

        # convert degrees K to degrees C
        sst_sub_c = sst_sub.values - 273.15

        # add contour lines
        contour_list = np.linspace(0, 30, 7)
        pf.add_contours(ax, lon, lat, sst_sub_c, contour_list)

        # plot data
        # pcolormesh: coarser resolution, shows the actual resolution of the model data
        pf.plot_pcolormesh(fig, ax, title, lon, lat, sst_sub_c, 0, 32, cmo.cm.thermal, color_label)

        # contourf: smooths the resolution of the model data, plots are less pixelated
        # levels = np.arange(0, 32.05, .05)
        # pf.plot_contourf(fig, ax, title, lon, lat, sst_sub_c, levels, cmo.cm.thermal, color_label, var_min=0,
        #                  var_max=32, normalize='no', cbar_ticks=np.linspace(0, 30, 7))

        cf.add_map_features(ax, ax_lims, landcolor='lightgray')

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

    # plot data only from the first hour because SST doesn't change for each model run
    fname = f0.split('/')[-1].split('.')[0]
    splitter = fname.split('/')[-1].split('_')
    ncfile = xr.open_dataset(f0, mask_and_scale=False)
    sfile = cf.save_filepath(save_dir, 'sst', splitter)
    plt_sst(ncfile, model_ver, sfile)

    print('')
    print('Script run time: {} minutes'.format(round(((time.time() - start_time) / 60), 2)))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Plot SST at model H000',
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
