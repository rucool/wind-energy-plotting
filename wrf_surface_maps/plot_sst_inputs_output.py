#!/usr/bin/env python

"""
Author: Lori Garzio on 7/6/2022
Last modified: 7/8/2022
Creates 4-panel surface maps of sea surface temperature from RU-WRF 4.1 input files (GOES Spike Filter and RTG), GFS
SST data from 000Z, and RU-WRF 4.1 output SST
"""

import argparse
import numpy as np
import pandas as pd
import os
import glob
import sys
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
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
#def main(ymd, save_dir):
    ymd = args.ymd
    save_dir = args.save_dir

    extent_3km, __, __, __ = cf.define_axis_limits('3km')
    extent_bight, __, __, __, = cf.define_axis_limits('bight')
    extents = dict(extent_3km=extent_3km,
                   extent_bight=extent_bight)

    yr = pd.to_datetime(ymd).year

    wrf_dir = os.path.join('/home/coolgroup/ru-wrf/real-time/v4.1_parallel/processed/3km', ymd)
    sst_inputs_dir = os.path.join('/home/coolgroup/ru-wrf/real-time/sst-input', ymd)
    gfs_dir = os.path.join('/home/coolgroup/ru-wrf/real-time/validation_data/gfs', str(yr), ymd)

    wrf_file = glob.glob(os.path.join(wrf_dir, 'wrfproc_*_00Z_H000.nc'))[0]
    sf_file = os.path.join(sst_inputs_dir, 'SST_raw_yesterday.nc')
    rtg_file = glob.glob(os.path.join(sst_inputs_dir, 'rtgssthr_*.grib2'))[0]
    gfs_file = glob.glob(os.path.join(gfs_dir, 'gfs*.f000.grib2'))[0]

    save_dir_zoom_out = os.path.join(save_dir, str(yr), 'zoom_out')
    save_dir_zoom_in = os.path.join(save_dir, str(yr), 'zoom_in')
    os.makedirs(save_dir_zoom_out, exist_ok=True)
    os.makedirs(save_dir_zoom_in, exist_ok=True)

    # get WRF output SST
    ds = xr.open_dataset(wrf_file)
    sst_wrf = np.squeeze(ds.SST) - 273.15  # convert K to degrees C
    landmask = np.squeeze(ds.LANDMASK)  # 1=land, 0=water
    lakemask = np.squeeze(ds.LAKEMASK)  # 1=lake, 0=non-lake

    # convert values over land and lakes to nans
    ldmask = np.logical_and(landmask == 1, landmask == 1)
    #sst_wrf.values[ldmask] = np.nan

    lkmask = np.logical_and(lakemask == 1, lakemask == 1)
    #sst_wrf.values[lkmask] = np.nan

    # get spike filter input SST
    ds_sf = xr.open_dataset(sf_file)
    sst_sf = np.squeeze(ds_sf.sst)

    # get RTG input SST (file is from the previous day for WRF input)
    ds_rtg = xr.open_dataset(rtg_file, engine='pynio')
    sst_rtg = np.squeeze(ds_rtg.TMP_P0_L1_GLL0) - 273.15  # convert K to degrees C

    # get GFS output (DISCLAIMER: I don't know which hour WRF uses so plotting 000Z for now)
    ds_gfs = xr.open_dataset(gfs_file, engine='pynio')
    sst_gfs = np.squeeze(ds_gfs.TMP_P0_L1_GLL0) - 273.15  # convert K to degrees C

    vlims = [14, 30]
    cmap = cmo.cm.thermal
    levels = MaxNLocator(nbins=16).tick_values(vlims[0], vlims[1])  # levels every 1 degrees C
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    main_title = f'RU-WRF Sea Surface Temperature: {pd.to_datetime(ymd).strftime("%Y-%m-%d")}'

    for key, extent in extents.items():
        save_file = os.path.join(save_dir_zoom_out, f'ru-wrf_sst_{ymd}')
        kwargs = dict()
        kwargs['zoom_coastline'] = False
        kwargs['landcolor'] = 'none'
        if key == 'extent_bight':
            save_file = os.path.join(save_dir_zoom_in, f'ru-wrf_sst_bight_{ymd}')
            kwargs['zoom_coastline'] = True

        fig, axs = plt.subplots(2, 2, figsize=(9, 8), sharey=True, sharex=True,
                                subplot_kw=dict(projection=ccrs.Mercator()))
        ax1 = axs[0, 0]
        ax2 = axs[0, 1]
        ax3 = axs[1, 0]
        ax4 = axs[1, 1]
        fig.suptitle(main_title, fontsize=16, y=.98)

        sst_wrf_sub, lon_wrf, lat_wrf = subset_grid(extent, sst_wrf, 'XLONG', 'XLAT')
        sst_sf_sub, lon_sf, lat_sf = subset_grid(extent, sst_sf, 'lon', 'lat')
        sst_rtg_sub, lon_rtg, lat_rtg = subset_grid(extent, sst_rtg, 'lon_0', 'lat_0')
        sst_gfs_sub, lon_gfs, lat_gfs = subset_grid(extent, sst_gfs, 'lon_0', 'lat_0')

        kwargs['bottom_label'] = False
        hp.map_create(extent, ax=ax1, **kwargs)
        kwargs['bottom_label'] = True
        hp.map_create(extent, ax=ax3, **kwargs)
        kwargs['left_label'] = False
        kwargs['bottom_label'] = False
        hp.map_create(extent, ax=ax2, **kwargs)
        kwargs['bottom_label'] = True
        hp.map_create(extent, ax=ax4, **kwargs)

        contour_list = [15, 20, 25, 30]
        pf.add_contours(ax1, lon_sf, lat_sf, sst_sf_sub.values, contour_list)
        pf.add_contours(ax2, lon_rtg, lat_rtg, sst_rtg_sub.values, contour_list)
        pf.add_contours(ax3, lon_gfs, lat_gfs, sst_gfs_sub.values, contour_list)
        pf.add_contours(ax4, lon_wrf, lat_wrf, sst_wrf_sub.values, contour_list)

        kwargs = dict()
        kwargs['panel_title'] = 'Spike Filter Input'
        kwargs['norm_clevs'] = norm
        kwargs['extend'] = 'both'
        kwargs['cmap'] = cmap
        kwargs['title_pad'] = 8
        pf.plot_pcolormesh_panel(fig, ax1, lon_sf, lat_sf, sst_sf_sub.values, **kwargs)

        kwargs['panel_title'] = f'GFS {ymd} 000Z'
        pf.plot_pcolormesh_panel(fig, ax3, lon_gfs, lat_gfs, sst_gfs_sub.values, **kwargs)

        kwargs['panel_title'] = 'RTG Input'
        kwargs['clab'] = 'SST (\N{DEGREE SIGN}C)'
        pf.plot_pcolormesh_panel(fig, ax2, lon_rtg, lat_rtg, sst_rtg_sub.values, **kwargs)

        kwargs['panel_title'] = 'RU-WRF Output'
        pf.plot_pcolormesh_panel(fig, ax4, lon_wrf, lat_wrf, sst_wrf_sub.values, **kwargs)

        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.02, hspace=0.12)
        plt.savefig(save_file, dpi=200)
        plt.close()


if __name__ == '__main__':
    # ymd = '20220708'
    # sDir = '/Users/garzio/Documents/rucool/bpu/wrf/sst-input-output'
    # main(ymd, sDir)
    arg_parser = argparse.ArgumentParser(description='Plot WRF SST inputs and output',
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