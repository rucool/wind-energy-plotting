#!/usr/bin/env python

"""
Author: Lori Garzio on 1/19/2024
Last modified: 1/21/2024
Creates 4-panel surface maps of sea surface temperature inputs and outputs from the summer 2022 modified SST
experiment
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
import cool_maps.plot as cplt
import functions.plotting as pf
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
#def main(start, end, save_dir):
    start = args.start
    end = args.end
    save_dir = args.save_dir

    daterange = pd.date_range(pd.to_datetime(start), pd.to_datetime(end))
    for d in daterange:
        ymd = d.strftime('%Y%m%d')
        yr = pd.to_datetime(ymd).year
        ym = ymd[0:6]
        month = pd.to_datetime(ymd).month

        save_dir = os.path.join(save_dir, str(yr), ym)
        os.makedirs(save_dir, exist_ok=True)

        wrf_dir = os.path.join('/home/coolgroup/ru-wrf/real-time/v4.1_parallel/processed_windturbs/1km_wf2km_nyb', ymd)
        wrf_dir_mod = os.path.join('/home/coolgroup/ru-wrf/real-time/v4.1_parallel/processed_windturbs/1km_wf2km_nyb_modsst', ymd)
        sst_inputs_dir = os.path.join('/home/coolgroup/ru-wrf/real-time/sst-input', ymd)
        sst_inputs_dir_mod = os.path.join('/home/coolgroup/ru-wrf/real-time/sst-input/temp_archive_mod_SST', ymd)

        # wrf_dir = os.path.join('/Users/garzio/Documents/rucool/bpu/wrf/modified_sst_testcases/summer2022/processed_windturbs/1km_wf2km_nyb', ymd)
        # wrf_dir_mod = os.path.join('/Users/garzio/Documents/rucool/bpu/wrf/modified_sst_testcases/summer2022/processed_windturbs/1km_wf2km_nyb_modsst', ymd)
        # sst_inputs_dir = os.path.join('/Users/garzio/Documents/rucool/bpu/wrf/modified_sst_testcases/summer2022/sst-input', ymd)
        # sst_inputs_dir_mod = os.path.join('/Users/garzio/Documents/rucool/bpu/wrf/modified_sst_testcases/summer2022/sst-input/temp_archive_mod_SST', ymd)

        #extent, __, __, __ = cf.define_axis_limits('windturb_nyb')
        extent = [-75.3, -72.1, 38, 40.5]

        wrf = xr.open_dataset(os.path.join(wrf_dir, f'wrfproc_1km_{ymd}_00Z_H000.nc'))
        wrf_mod = xr.open_dataset(os.path.join(wrf_dir_mod, f'wrfproc_1km_{ymd}_00Z_H000.nc'))
        sst_input = xr.open_dataset(os.path.join(sst_inputs_dir, 'SST_raw_yesterday.nc'))
        sst_input_mod = xr.open_dataset(os.path.join(sst_inputs_dir_mod, 'SST_raw_yesterday.nc'))

        vlims = [14, 28]
        cmap = cmo.cm.thermal
        levels = MaxNLocator(nbins=16).tick_values(vlims[0], vlims[1])
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        # get SST from each data source
        sst_wrf = np.squeeze(wrf.SST) - 273.15  # convert K to degrees C
        sst_wrf_mod = np.squeeze(wrf_mod.SST) - 273.15  # convert K to degrees C
        landmask = np.squeeze(wrf.LANDMASK)  # 1=land, 0=water
        lakemask = np.squeeze(wrf.LAKEMASK)  # 1=lake, 0=non-lake

        # convert values over land and lakes to nans
        ldmask = np.logical_and(landmask == 1, landmask == 1)
        sst_wrf.values[ldmask] = np.nan
        sst_wrf_mod.values[ldmask] = np.nan

        lkmask = np.logical_and(lakemask == 1, lakemask == 1)
        sst_wrf.values[lkmask] = np.nan
        sst_wrf_mod.values[lkmask] = np.nan

        sst_input = np.squeeze(sst_input.sst)
        sst_input_mod = np.squeeze(sst_input_mod.sst)

        sst_wrf_sub, wrf_sub_lon, wrf_sub_lat = subset_grid(extent, sst_wrf, 'XLONG', 'XLAT')
        sst_wrf_mod_sub, wrf_sub_mod_lon, wrf_sub_mod_lat = subset_grid(extent, sst_wrf_mod, 'XLONG', 'XLAT')
        sst_input_sub, input_lon, input_lat = subset_grid(extent, sst_input, 'lon', 'lat')
        sst_input_mod_sub, input_mod_lon, input_mod_lat = subset_grid(extent, sst_input_mod, 'lon', 'lat')

        main_title = f'RU-WRF Sea Surface Temperature: {pd.to_datetime(ymd).strftime("%Y-%m-%d")}'
        save_file = os.path.join(save_dir, f'ru-wrf_sst_inputs_outputs_{ymd}')
        kwargs = dict()
        kwargs['oceancolor'] = 'none'
        #kwargs['coast'] = 'low'

        fig, axs = plt.subplots(2, 2, figsize=(9, 8), sharey=True, sharex=True,
                                subplot_kw=dict(projection=ccrs.Mercator()))
        ax1 = axs[0, 0]  # input SST
        ax2 = axs[0, 1]  # output SST
        ax3 = axs[1, 0]  # input SST modified version
        ax4 = axs[1, 1]  # output SST modified version
        fig.suptitle(main_title, fontsize=16, y=.98)
        kwargs['tick_label_bottom'] = False
        cplt.create(extent, ax=ax1, **kwargs)
        kwargs['tick_label_bottom'] = True
        cplt.create(extent, ax=ax3, **kwargs)
        kwargs['tick_label_left'] = False
        kwargs['tick_label_bottom'] = False
        cplt.create(extent, ax=ax2, **kwargs)
        kwargs['tick_label_bottom'] = True
        cplt.create(extent, ax=ax4, **kwargs)

        contour_list = [5, 10, 15, 20, 25, 30]
        pf.add_contours(ax1, input_lon, input_lat, sst_input_sub.values, contour_list)
        pf.add_contours(ax2, wrf_sub_lon, wrf_sub_lat, sst_wrf_sub.values, contour_list)
        pf.add_contours(ax3, input_mod_lon, input_mod_lat, sst_input_mod_sub.values, contour_list)
        pf.add_contours(ax4, wrf_sub_mod_lon, wrf_sub_mod_lat, sst_wrf_mod_sub.values, contour_list)

        kwargs = dict()
        kwargs['panel_title'] = 'Input SST'
        kwargs['norm_clevs'] = norm
        kwargs['extend'] = 'both'
        kwargs['cmap'] = cmap
        kwargs['title_pad'] = 8
        pf.plot_pcolormesh_panel(fig, ax1, input_lon, input_lat, sst_input_sub.values, **kwargs)

        kwargs['panel_title'] = f'Input Modified SST'
        pf.plot_pcolormesh_panel(fig, ax3, input_mod_lon, input_mod_lat, sst_input_mod_sub.values, **kwargs)

        kwargs['panel_title'] = 'RU-WRF Output SST'
        kwargs['clab'] = 'SST (\N{DEGREE SIGN}C)'
        pf.plot_pcolormesh_panel(fig, ax2, wrf_sub_lon, wrf_sub_lat, sst_wrf_sub.values, **kwargs)

        kwargs['panel_title'] = 'RU-WRF Output Modified SST'
        pf.plot_pcolormesh_panel(fig, ax4, wrf_sub_mod_lon, wrf_sub_mod_lat, sst_wrf_mod_sub.values, **kwargs)

        lease = glob.glob('/home/coolgroup/bpu/mapdata/shapefiles/BOEM-Renewable-Energy-Shapefiles-current/Wind_Lease_Outlines*.shp')[0]
        #lease = glob.glob('/Users/garzio/Documents/rucool/bpu/wrf/lease_areas/BOEM-Renewable-Energy-Shapefiles-current/Wind_Lease_Outlines*.shp')[0]
        kwargs = dict()
        kwargs['edgecolor'] = 'magenta'
        pf.map_add_boem_outlines(ax1, lease, **kwargs)
        pf.map_add_boem_outlines(ax2, lease, **kwargs)
        pf.map_add_boem_outlines(ax3, lease, **kwargs)
        pf.map_add_boem_outlines(ax4, lease, **kwargs)

        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.02, hspace=0.12)
        #plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.12)
        plt.savefig(save_file, dpi=200)
        plt.close()


if __name__ == '__main__':
    # start = '20220601'
    # end = '20220831'
    # savedir = '/Users/garzio/Documents/rucool/bpu/wrf/modified_sst_testcases/summer2022'
    # main(start, end, savedir)
    arg_parser = argparse.ArgumentParser(description='Plot WRF SST inputs and outputs',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('start',
                            type=str,
                            help='Start date to plot in the format YYYYmmdd (e.g. 20220601.')

    arg_parser.add_argument('end',
                            type=str,
                            help='End date to plot in the format YYYYmmdd (e.g. 20220831.')

    arg_parser.add_argument('-save_dir',
                            default='/www/web/rucool/windenergy/ru-wrf/images/upwelling_case/upwelling_sst_input_output',
                            type=str,
                            help='Full directory path to save output plots.')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
