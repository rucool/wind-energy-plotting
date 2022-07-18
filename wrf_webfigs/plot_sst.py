#!/usr/bin/env python

"""
Author: Lori Garzio on 8/21/2020
Last modified: 7/18/2022
Creates a plot of sea surface temperature from RU-WRF 4.1 subset files from H001 (SST does not change throughout the day)
"""

import argparse
import numpy as np
import os
import glob
import sys
import time
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
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
    color_label = 'SST (\N{DEGREE SIGN}C)'
    title = 'Sea Surface Temperature (\N{DEGREE SIGN}C)'

    plot_types = ['full_grid', 'bight']  # plot the full grid and just NY Bight area
    for pt in plot_types:
        if pt == 'full_grid':  # subset the entire grid
            sst_sub, ax_lims, xticks, yticks = cf.subset_grid(sst, model)
            landmask_sub, _, _, _ = cf.subset_grid(landmask, model)
            lakemask_sub, _, _, _ = cf.subset_grid(lakemask, model)
        else:  # subset just NY Bight
            new_fname = 'bight_{}'.format(figname.split('/')[-1])
            figname = '/{}/{}'.format(os.path.join(*figname.split('/')[0:-1]), new_fname)
            sst_sub, ax_lims, xticks, yticks = cf.subset_grid(sst, 'bight')
            landmask_sub, _, _, _ = cf.subset_grid(landmask, 'bight')
            lakemask_sub, _, _, _ = cf.subset_grid(lakemask, 'bight')

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
        #vlims = [0, 32]
        #bins = 16
        vlims = [20, 31]
        bins = 12
        #vlims = [np.nanmin(sst_sub_c), np.nanmax(sst_sub_c)]
        cmap = cmo.cm.thermal
        levels = MaxNLocator(nbins=bins).tick_values(vlims[0], vlims[1])  # levels every x degrees C
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        kwargs = dict()
        kwargs['ttl'] = title
        kwargs['clab'] = color_label
        # kwargs['var_lims'] = vlims
        kwargs['norm_clevs'] = norm
        kwargs['extend'] = 'both'
        kwargs['cmap'] = cmap
        pf.plot_pcolormesh(fig, ax, lon, lat, sst_sub_c, **kwargs)

        # initialize keyword arguments for map features
        kwargs = dict()
        kwargs['xticks'] = xticks
        kwargs['yticks'] = yticks
        kwargs['landcolor'] = 'tan'
        cf.add_map_features(ax, ax_lims, **kwargs)

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
        if splitter[-1] == 'H001':
            ncfile = xr.open_dataset(f, mask_and_scale=False)
            sfile = cf.save_filepath(save_dir, 'sst', splitter)
            plt_sst(ncfile, model_ver, sfile)

    print('')
    print('Script run time: {} minutes'.format(round(((time.time() - start_time) / 60), 2)))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Plot SST from WRF output',
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
