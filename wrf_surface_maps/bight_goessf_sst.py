## Author: Lori Garzio , adapted by james kim
#Creates a zoomed in plot of Goes SST on the priamary upwelling zone along the new jersey bight area



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
        ##### ARGS ######
    ymd = args.ymd
    save_dir = args.save_dir

    yr = pd.to_datetime(ymd).year
    ym = ymd[0:6]
    month = pd.to_datetime(ymd).month

    save_dir = os.path.join(save_dir, str(yr), 'goes_only',ym)
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f'ru-wrf_sst_goes_{ymd}')

    extent_3km, __, __, __ = cf.define_axis_limits('bight')
    
    save_file = os.path.join(save_dir, f'ru-wrf_sst_goes_spike_filter_{ymd}.png')  # Modified save_file name

    yr = pd.to_datetime(ymd).year
    ym = ymd[0:6]
    month = pd.to_datetime(ymd).month


    goes_sf_dir = os.path.join('/home/coolgroup/bpu/wrf/data/goes_composites/composites', f'{yr}')
    goes_sf_file = os.path.join(goes_sf_dir, f'goes_stacked_composites_{ymd}T0000.nc')

    # get GOES Spike Filter file
    try:
        ds_goes = xr.open_dataset(goes_sf_file)
        sst_goes = np.squeeze(ds_goes.sst)
    except FileNotFoundError:
        print(f'No such file or directory: {goes_sf_file}')
        sst_goes = None

    #plotting 

    vlims = [16, 25]
    cmap = cmo.cm.thermal
    levels = MaxNLocator(nbins=10).tick_values(vlims[0], vlims[1])  # levels every 1 degrees C
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    if type(sst_goes) == xr.core.dataarray.DataArray:
        goes_sub, lon_goes, lat_goes = subset_grid(extent_3km, sst_goes, 'lon', 'lat')
    else:
        goes_sub = None

    main_title = f'RU-WRF GOES_Spike_Filter SST Input: {pd.to_datetime(ymd).strftime("%Y-%m-%d")}'

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection=ccrs.Mercator()))
    fig.suptitle(main_title, fontsize=16, y=.98)


    kwargs = dict()
    #kwargs['zoom_coastline'] = False
    #kwargs['bottom_label'] = False
    hp.map_create(extent_3km, ax=ax, **kwargs)
    kwargs['cmap'] = cmap
    kwargs['norm_clevs'] = norm
    kwargs['title_pad'] = 9
    kwargs['extend'] = 'both'
    kwargs['clab'] = 'SST (\N{DEGREE SIGN}C)'

    contour_list = [5, 10, 15, 20, 25, 30]

    if type(goes_sub) == xr.core.dataarray.DataArray:
        if np.sum(~np.isnan(goes_sub.values)) > 0:  # check if the GOES-SF file has any data
            pf.add_contours(ax, lon_goes, lat_goes, goes_sub.values, contour_list)

    if type(goes_sub) == xr.core.dataarray.DataArray:
                pf.plot_pcolormesh_panel(fig, ax, lon_goes, lat_goes, goes_sub.values, **kwargs)
    plt.savefig(save_file, dpi=200)
    plt.close()
    
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Plot GOES Spike SST',
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
    
