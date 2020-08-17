#!/usr/bin/env python

"""
Author: Lori Garzio on 5/28/2020
Last modified: 8/17/2020
Creates hourly plots of RU-WRF 4.1 output variables: air temperature at 2m. The plots are used to populate
RUCOOL's RU-WRF webpage:
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


def plt_2m_temp(nc, model, figname):
    """
    Create a pcolor surface map of air temperature at 2m with contours
    :param nc: netcdf file
    :param model: the model version that is being plotted, e.g. 3km or 9km
    :param figname: full file path to save directory and save filename
    """
    t2 = nc['T2']
    color_label = 'Air Temperature ($^\circ$F)'

    plot_types = ['full_grid', 'bight']  # plot the full grid and just NY Bight area
    for pt in plot_types:
        if pt == 'full_grid':  # subset the entire grid
            t2_sub, ax_lims = cf.subset_grid(t2, model)
        else:  # subset just NY Bight
            new_fname = 'bight_{}'.format(figname.split('/')[-1])
            figname = '/{}/{}'.format(os.path.join(*figname.split('/')[0:-1]), new_fname)
            t2_sub, ax_lims = cf.subset_grid(t2, 'bight')

        fig, ax, lat, lon = cf.set_map(t2_sub)

        # add text to the bottom of the plot
        cf.add_text(ax, nc.SIMULATION_START_DATE, nc.time_coverage_start, model)

        cf.add_map_features(ax, ax_lims)

        # convert K to F
        d = np.squeeze(t2_sub.values) * 9/5 - 459.67

        # add contour lines
        contour_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        pf.add_contours(ax, lon, lat, d, contour_list)

        # plot data
        # pcolormesh: coarser resolution, shows the actual resolution of the model data
        pf.plot_pcolormesh(fig, ax, '2m {}'.format(color_label), lon, lat, d, -20, 110, 'jet', color_label)

        # contourf: smooths the resolution of the model data, plots are less pixelated
        # levels = np.arange(-20, 110.5, .5)
        # pf.plot_contourf(fig, ax, color_label, lon, lat, d, levels, 'jet', color_label, var_min=-20, var_max=110,
        #                  normalize='no', cbar_ticks=np.linspace(-20, 100, 7))

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

    # get the model version (3km or 9km) from the directory path
    f0 = files[0]
    model_ver = f0.split('/')[-3]  # 3km or 9km
    os.makedirs(save_dir, exist_ok=True)

    for i, f in enumerate(files):
        fname = f.split('/')[-1].split('.')[0]
        splitter = fname.split('/')[-1].split('_')
        ncfile = xr.open_dataset(f, mask_and_scale=False)
        sfile = cf.save_filepath(save_dir, 'T2', splitter)
        plt_2m_temp(ncfile, model_ver, sfile)

    print('')
    print('Script run time: {} minutes'.format(round(((time.time() - start_time) / 60), 2)))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Plot 2m Air Temperature',
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
