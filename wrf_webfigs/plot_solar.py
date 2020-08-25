#!/usr/bin/env python

"""
Author: Lori Garzio on 8/24/2020
Last modified: 8/25/2020
Creates hourly plots of RU-WRF 4.1 output variables: downwelling shortwave irradiance and diffuse fraction of shortwave
irradiance. The plots are used to populate RUCOOL's RU-WRF webpage:
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


def plt_solar(nc, model, figname):
    """
    Create a pcolor surface map of air temperature at 2m with contours
    :param nc: netcdf file
    :param model: the model version that is being plotted, e.g. 3km or 9km
    :param figname: full file path to save directory and save filename
    """
    varname = figname.split('/')[-1].split('_')[0]
    if varname == 'srad':
        solar = nc['SWDOWN']
        color_label = r'Surface Downwelling Shortwave Flux (W $\rm m^{-2}$)'  # \rm removes the italics'
        mingray = 50  # minimum value for making the state/coastlines gray
        vmin = 0.0
        vmax = 1000
        contour_list = np.linspace(200, 1000, 5)
        lab_format = None  # defaults to '%d' in function
    elif varname == 'diffuse':
        solar = nc['DIFFUSE_FRAC']
        color_label = 'Diffuse Fraction of Surface Shortwave Irradiance'
        mingray = 0.1  # minimum value for making the state/coastlines gray
        vmin = 0.0
        vmax = 1.0
        contour_list = np.linspace(.2, 1, 5)
        lab_format = '%.1f'

    plot_types = ['full_grid', 'bight']  # plot the full grid and just NY Bight area
    for pt in plot_types:
        if pt == 'full_grid':  # subset the entire grid
            solar_sub, ax_lims = cf.subset_grid(solar, model)
        else:  # subset just NY Bight
            new_fname = 'bight_{}'.format(figname.split('/')[-1])
            figname = '/{}/{}'.format(os.path.join(*figname.split('/')[0:-1]), new_fname)
            solar_sub, ax_lims = cf.subset_grid(solar, 'bight')

        fig, ax, lat, lon = cf.set_map(solar_sub)

        # add text to the bottom of the plot
        cf.add_text(ax, nc.SIMULATION_START_DATE, nc.time_coverage_start, model)

        # change the state and coastline edges to gray if there is no solar radiation
        if np.nanmax(solar_sub) < mingray:
            cf.add_map_features(ax, ax_lims, ecolor='#525252')
        else:
            cf.add_map_features(ax, ax_lims)

        # add contour lines
        pf.add_contours(ax, lon, lat, solar_sub, contour_list, label_format=lab_format)

        # plot data
        pf.plot_pcolormesh(fig, ax, color_label, lon, lat, solar_sub, vmin, vmax, plt.cm.CMRmap, color_label)

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

    # List of variables to plot
    plt_vars = ['srad', 'diffuse_srad']

    for i, f in enumerate(files):
        fname = f.split('/')[-1].split('.')[0]
        splitter = fname.split('/')[-1].split('_')
        ncfile = xr.open_dataset(f, mask_and_scale=False)
        for pv in plt_vars:
            sfile = cf.save_filepath(save_dir, pv, splitter)
            plt_solar(ncfile, model_ver, sfile)

    print('')
    print('Script run time: {} minutes'.format(round(((time.time() - start_time) / 60), 2)))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Plot downwelling shortwave irradiance and diffuse fraction of shortwave irradiance',
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
