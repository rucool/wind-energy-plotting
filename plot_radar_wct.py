#!/usr/bin/env python

"""
Author: Lori Garzio on 7/13/2021
Last modified: 7/15/2021
Creates a surface map of radar reflectivity from files downloaded from NOAA's Weather and Climate Toolkit:
https://www.ncdc.noaa.gov/wct/
"""

import argparse
import numpy as np
import os
import sys
import xarray as xr
import matplotlib.pyplot as plt
import pyart  # used for the colormap 'pyart_NWSRef'
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def plt_radar(nc, subset_domain, figname):
    """
    Create filled contour surface maps of radar reflectivity
    :param nc: netcdf file
    :param subset_domain: the plotting limits to match WRF output, e.g. 3km, 9km, or bight (NY Bight)
    :param figname: full file path to save directory and save filename
    """

    radar = nc['Reflectivity']

    ax_lims, _ = cf.define_axis_limits(subset_domain)

    fig, ax, lat, lon = cf.set_map(radar)
    cf.add_map_features(ax, ax_lims)
    title = 'Radar Reflectivity ({})'.format(radar.units)
    vmin = 0
    vmax = 72

    # If the array is all zeros, turn the zeros to nans. Otherwise the plot will be all teal instead of white.
    if np.nanmax(radar) == 0.0:
        radar.values[radar == 0] = np.nan

    pf.plot_pcolormesh(fig, ax, title, lon, lat, np.squeeze(radar.values), vmin, vmax, 'pyart_NWSRef',
                       'Radar Reflectivity ({})'.format(radar.units))

    plt.savefig(figname, dpi=200)
    plt.close()


def main(args):
    ncfile = args.ncfile
    save_dir = args.save_dir
    wrf_domain = args.wrf_domain

    os.makedirs(save_dir, exist_ok=True)

    fname = ncfile.split('/')[-1].split('.')[0]
    sfile = os.path.join(save_dir, f'{fname}_radar_{wrf_domain}.png')
    nc = xr.open_dataset(ncfile, mask_and_scale=False)
    plt_radar(nc, wrf_domain, sfile)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Plot radar reflectivity downloaded from NOAA's Weather and Climate Toolkit",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-f',
                            dest='ncfile',
                            type=str,
                            help='Full file path to NWS Gridded NetCDF file.')

    arg_parser.add_argument('-s',
                            dest='save_dir',
                            type=str,
                            help='Full directory path to save output plot.')
    arg_parser.add_argument('-wd',
                            dest='wrf_domain',
                            type=str,
                            help='WRF domain limits for plotting, e.g. 3km, 9km, bight')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
