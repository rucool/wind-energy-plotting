#!/usr/bin/env python

"""
Author: Lori Garzio on 7/6/2022
Last modified: 10/6/2022
Creates a surface map of sea surface temperature from the RU-WRF 4.1 input files (GOES Spike Filter and RTG
composite, "SST_raw_yesterday.nc")
"""

import argparse
import numpy as np
import pandas as pd
import os
import sys
import xarray as xr
import yaml
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
import cmocean as cmo
import functions.common as cf
import functions.plotting as pf
import functions.hurricanes_plotting as hp
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def subset_grid(ext, dataset, lon_name, lat_name):
    if len(np.shape(dataset[lon_name])) == 1:
        lonx, laty = np.meshgrid(dataset[lon_name], dataset[lat_name])
    else:
        lonx = dataset[lon_name]
        laty = dataset[lat_name]

    if dataset.name == 'TMP_P0_L1_GLL0':  # RTG dataset
        lonx[lonx > 180] = lonx[lonx > 180] - 360  # convert longitude from 0 to 360 to -180 to 180

    lon_idx = np.logical_and(lonx > ext[0], lonx < ext[1])
    lat_idx = np.logical_and(laty > ext[2], laty < ext[3])

    # find i and j indices of lon/lat in boundaries
    ind = np.where(np.logical_and(lat_idx, lon_idx))

    # subset data from min i,j corner to max i,j corner
    # there will be some points outside of defined boundaries because grid is not rectangular
    data_sub = np.squeeze(dataset)[range(np.min(ind[0]), np.max(ind[0]) + 1), range(np.min(ind[1]), np.max(ind[1]) + 1)]
    lon = data_sub[lon_name]
    lat = data_sub[lat_name]

    return data_sub, lon, lat


def main(args):
    ymd = args.ymd
    save_dir = args.save_dir

    yr = pd.to_datetime(ymd).year
    ym = ymd[0:6]
    month = pd.to_datetime(ymd).month

    save_dir_3km_zoom_in = os.path.join(save_dir, str(yr), 'ruwrf_input', ym)
    os.makedirs(save_dir_3km_zoom_in, exist_ok=True)

    sst_inputs_dir = os.path.join('/home/coolgroup/ru-wrf/real-time/sst-input', ymd)

    extent_bight, __, __, __, = cf.define_axis_limits('bight')
    extents = dict(
        extent_bight_3km=dict(
            lims=extent_bight,
            save=save_dir_3km_zoom_in
        )
    )

    wrf_sst_input_file = os.path.join(sst_inputs_dir, 'SST_raw_yesterday.nc')

    # get WRF SST input file
    try:
        ds_wrf_input = xr.open_dataset(wrf_sst_input_file)
        sst_wrf_input = np.squeeze(ds_wrf_input.sst)
    except FileNotFoundError:
        print(f'No such file or directory: {wrf_sst_input_file}')
        sst_wrf_input = None

    # get colorbar limits from configuration file
    configfile = cf.sst_surface_map_config()
    with open(configfile) as config:
        config_info = yaml.full_load(config)
        for k, v in config_info.items():
            if month in v['months']:
                color_lims = v['color_lims']

    bins = color_lims[1] - color_lims[0]
    cmap = cmo.cm.thermal
    levels = MaxNLocator(nbins=bins).tick_values(color_lims[0], color_lims[1])
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    for key, values in extents.items():
        save_file = os.path.join(values['save'], f'ru-wrf_sst_input_{ymd}')
        kwargs = dict()
        kwargs['zoom_coastline'] = False

        fig, ax = hp.map_create(values['lims'], **kwargs)

        if type(sst_wrf_input) == xr.core.dataarray.DataArray:
            sst_wrf_input_sub, lon_sst_wrf_input, lat_sst_wrf_input = subset_grid(values['lims'], sst_wrf_input, 'lon', 'lat')
        else:
            sst_wrf_input_sub = None

        contour_list = [5, 10, 15, 20, 25, 30]
        if type(sst_wrf_input_sub) == xr.core.dataarray.DataArray:
            pf.add_contours(ax, lon_sst_wrf_input, lat_sst_wrf_input, sst_wrf_input_sub.values, contour_list)

        kwargs = dict()
        kwargs['ttl'] = f'GOES-SF + RTG Composite {pd.to_datetime(ymd).strftime("%Y-%m-%d")}'
        kwargs['norm_clevs'] = norm
        kwargs['extend'] = 'both'
        kwargs['cmap'] = cmap
        kwargs['clab'] = 'SST (\N{DEGREE SIGN}C)'
        if type(sst_wrf_input_sub) == xr.core.dataarray.DataArray:
            pf.plot_pcolormesh(fig, ax, lon_sst_wrf_input, lat_sst_wrf_input, sst_wrf_input_sub.values, **kwargs)
        else:
            ax.set_title(kwargs['ttl'], fontsize=15)

        plt.savefig(save_file, dpi=200)
        plt.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Plot daily WRF SST input',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('ymd',
                            type=str,
                            help='Year-month-day to plot in the format YYYYmmdd (e.g. 20220101.')

    arg_parser.add_argument('-save_dir',
                            default='/www/web/rucool/windenergy/ru-wrf/images/daily/sst-input',
                            type=str,
                            help='Full directory path to save output plots.')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
