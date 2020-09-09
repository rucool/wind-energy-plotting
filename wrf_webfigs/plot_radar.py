#!/usr/bin/env python

"""
Author: Lori Garzio on 8/21/2020
Last modified: 9/9/2020
Creates hourly plots of RU-WRF 4.1 output variables: radar composite reflectivity.
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
import matplotlib.pyplot as plt
import pyart  # used for the colormap 'pyart_NWSRef'
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def plt_radar(nc, model, figname, lease_areas):
    """
    Create filled contour surface maps of radar reflectivity
    :param nc: netcdf file
    :param model: the model version that is being plotted, e.g. 3km or 9km
    :param figname: full file path to save directory and save filename
    :param lease_areas: dictionary containing lat/lon coordinates for wind energy lease area polygon
    """
    # MDBZ = max radar reflectivity
    radar = nc['MDBZ']

    plot_types = ['full_grid', 'bight']  # plot the full grid and just NY Bight area
    for pt in plot_types:
        if pt == 'full_grid':  # subset the entire grid
            radar_sub, ax_lims = cf.subset_grid(radar, model)
        else:  # subset just NY Bight
            new_fname = 'bight_{}'.format(figname.split('/')[-1])
            figname = '/{}/{}'.format(os.path.join(*figname.split('/')[0:-1]), new_fname)
            radar_sub, ax_lims = cf.subset_grid(radar, 'bight')

        fig, ax, lat, lon = cf.set_map(radar_sub)

        # add text to the bottom of the plot
        cf.add_text(ax, nc.SIMULATION_START_DATE, nc.time_coverage_start, model)

        cf.add_map_features(ax, ax_lims)

        # pf.add_lease_area_polygon(ax, lease_areas, 'magenta')

        title = 'Radar Composite Reflectivity ({})'.format(radar.units)

        vmin = 0
        vmax = 72
        levels = np.linspace(vmin, vmax, 145)
        ticklevs = np.linspace(vmin, 70, 15)

        # If the array is all zeros, turn the zeros to nans. Otherwise the plot will be all teal instead of white.
        if np.nanmax(radar_sub) == 0.0:
            radar_sub.values[radar_sub == 0] = np.nan

        pf.plot_contourf(fig, ax, title, lon, lat, radar_sub, levels, 'pyart_NWSRef', title, var_min=vmin, var_max=vmax,
                         normalize='no', cbar_ticks=ticklevs)

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
        sfile = cf.save_filepath(save_dir, 'radar', splitter)
        plt_radar(ncfile, model_ver, sfile, la_polygon)

    print('')
    print('Script run time: {} minutes'.format(round(((time.time() - start_time) / 60), 2)))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Plot maximum radar reflectivity',
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
