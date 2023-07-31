# Author: James Kim
# date: 07/31/23

#this code will plot the RTG SST input used in WRF model Runs (Made for December 2018 backdate)


import argparse
import numpy as np
import pandas as pd
import os
import glob
import sys
import xarray as xr
import yaml
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
    ymd = args.ymd
    save_dir = args.save_dir

    yr = pd.to_datetime(ymd).year
    ym = ymd[0:6]
    month = pd.to_datetime(ymd).month

    save_dir_rtg = os.path.join(save_dir, str(yr), 'rtg_only','ym')
    os.makedirs(save_dir_rtg, exist_ok=True)

    sst_inputs_dir = os.path.join('/home/coolgroup/ru-wrf/real-time/sst-input', ymd)

    extent_9km = [-80, -60, 31, 46]
    extents = dict(
        extent_9km=dict(
            lims=extent_9km,
            save=save_dir_rtg,
        )
    )

    try:
        rtg_file = glob.glob(os.path.join(sst_inputs_dir, 'rtgssthr_*.grib2'))[0]
    except IndexError:
        rtg_file = 'no_file'
        
    # get RTG file (file is from the previous day for the WRF input)
    if rtg_file == 'no_file':
        print(f'No such file or directory: {rtg_file}')
        sst_rtg = None
    else:
        ds_rtg = xr.open_dataset(rtg_file, engine='pynio')
        sst_rtg = np.squeeze(ds_rtg.TMP_P0_L1_GLL0) - 273.15  # convert K to degrees C

    vlims = [5, 30]
    bins = color_lims[1] - color_lims[0]
    cmap = cmo.cm.thermal
    levels = MaxNLocator(nbins=bins).tick_values(vlims[0], vlims[1])
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    for key, values in extents.items():
        model = key.split("_")[-1]

        main_title = f'RTG Sea Surface Temperature {model}: {pd.to_datetime(ymd).strftime("%Y-%m-%d")}'
        save_file = os.path.join(values['save'], f'ru-wrf_{model}_sst_{ymd}')
        kwargs = dict()
        kwargs['zoom_coastline'] = False

        fig, ax = plt.subplots(figsize=(9, 8), subplot_kw=dict(projection=ccrs.Mercator()))
        fig.suptitle(main_title, fontsize=16, y=.98)

        if type(sst_rtg) == xr.core.dataarray.DataArray:
            sst_rtg_sub, lon_rtg, lat_rtg = subset_grid(values['lims'], sst_rtg, 'lon_0', 'lat_0')
        else:
            sst_rtg_sub = None

        kwargs['panel_title'] = f'RTG'
        if type(sst_rtg_sub) == xr.core.dataarray.DataArray:
            pf.plot_pcolormesh_panel(fig, ax, lon_rtg, lat_rtg, sst_rtg_sub.values, **kwargs)
        else:
            ax.set_title(kwargs['panel_title'], fontsize=15, pad=kwargs['title_pad'])

        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.02, hspace=0.12)
        plt.savefig(save_file, dpi=200)
        plt.close()
