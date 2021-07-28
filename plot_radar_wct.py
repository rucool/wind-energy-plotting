#!/usr/bin/env python

"""
Author: Lori Garzio on 7/13/2021
Last modified: 7/28/2021
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
    :param subset_domain: the plotting limit domain, e.g. 3km, 9km, bight (NY Bight), full_grid, mab, nj, snj
    :param figname: full file path to save directory and save filename
    """
    radar = nc['Reflectivity']

    radar_sub, ax_lims, xticks, yticks = cf.subset_grid_wct(radar, subset_domain)

    fig, ax, lat, lon = cf.set_map(radar_sub)

    # initialize keyword arguments for map features
    kwargs = dict()
    kwargs['xticks'] = xticks
    kwargs['yticks'] = yticks

    cf.add_map_features(ax, ax_lims, **kwargs)

    title = 'Radar Reflectivity ({})'.format(radar_sub.units)

    kwargs = dict()
    kwargs['ttl'] = title
    kwargs['cmap'] = 'pyart_NWSRef'
    kwargs['clab'] = title
    kwargs['var_lims'] = [0, 72]

    # If the array is all zeros, turn the zeros to nans. Otherwise the plot will be all teal instead of white.
    if np.nanmax(radar_sub) == 0.0:
        radar_sub.values[radar_sub == 0] = np.nan

    pf.plot_pcolormesh(fig, ax, lon, lat, np.squeeze(radar_sub.values), **kwargs)

    plt.savefig(figname, dpi=200)
    plt.close()


def main(args):
    ncfile = args.ncfile
    save_dir = args.save_dir
    wrf_domain = args.wrf_domain

    os.makedirs(save_dir, exist_ok=True)

    fname = ncfile.split('/')[-1].split('.')[0]
    nc = xr.open_dataset(ncfile, mask_and_scale=False)
    sfile = os.path.join(save_dir, f'{fname}_radar_{wrf_domain}.png')
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
                            help='WRF domain limits for plotting, e.g. 3km, 9km, bight, full_grid, mab, nj, snj')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
