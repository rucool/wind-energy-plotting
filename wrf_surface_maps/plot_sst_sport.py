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

##### ARGS ######
def main(args):
    ymd = args.ymd
    save_dir = args.save_dir

    yr = pd.to_datetime(ymd).year
    ym = ymd[0:6]
    month = pd.to_datetime(ymd).month

    save_dir = os.path.join(save_dir, str(yr), 'sportavhrr_only',ym)
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f'ru-wrf_sst_sportavhrr_{ymd}')
    
    sst_inputs_dir = os.path.join('/users/jameskim/Documents/rucool/SSTcodetest',ymd)
    sat_file = os.path.join(sst_inputs_dir, 'SST_raw_yesterday.nc')

    extent_3km, __, __, __ = cf.define_axis_limits('3km')

    ds_sf = xr.open_dataset(sat_file)
    sst_sf = np.squeeze(ds_sf.sst)

    vlims = [5, 30]
    cmap = cmo.cm.thermal
    levels = MaxNLocator(nbins=16).tick_values(vlims[0], vlims[1])  # levels every 1 degrees C
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    main_title = f'RU-WRF SPORT/AVHRR SST Input: {pd.to_datetime(ymd).strftime("%Y-%m-%d")}'

    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw=dict(projection=ccrs.Mercator()))
    fig.suptitle(main_title, fontsize=16, y=.98)

    sst_sf_sub, lon_sf, lat_sf = subset_grid(extent_3km, sst_sf, 'lon', 'lat')

    hp.map_create(extent_3km, ax=ax)

    contour_list = [10,15, 20, 25, 30]
    try:
        pf.add_contours(ax, lon_sf, lat_sf, sst_sf_sub.values, contour_list)
    except TypeError:
        sst_sf = np.squeeze(ds_sf.sst).transpose()
        sst_sf_sub, lon_sf, lat_sf = subset_grid(extent_3km, sst_sf, 'lon', 'lat')
        pf.add_contours(ax, lon_sf, lat_sf, sst_sf_sub.values, contour_list)

    kwargs = dict()
    kwargs['norm_clevs'] = norm
    kwargs['extend'] = 'both'
    kwargs['cmap'] = cmap
    kwargs['title_pad'] = 8
    kwargs['clab'] = 'SST (\N{DEGREE SIGN}C)'
    pf.plot_pcolormesh_panel(fig, ax, lon_sf, lat_sf, sst_sf_sub.values, **kwargs)

    try:
        plt.savefig(save_file, dpi=200)
    except Exception as e:
        print(f"Error occurred while saving the file: {e}")
        plt.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Plot WRF SST RTG input',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('ymd',
                        type=str,
                        help='Year-month-day to plot in the format YYYYmmdd (e.g. 20220101.')

    arg_parser.add_argument('-save_dir',
                        default='/www/web/rucool/windenergy/ru-wrf/images/daily/sst-input/',
                        type=str,
                        help='Full directory path to save output plots.')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))