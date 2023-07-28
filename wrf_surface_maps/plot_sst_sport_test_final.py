import argparse
import numpy as np
import pandas as pd
import os
import glob
import datetime as dt
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
    ymd = args.ymd
    save_dir = args.save_dir

    extent_3km, __, __, __ = cf.define_axis_limits('3km')
    extent_bight, __, __, __, = cf.define_axis_limits('bight')
    extents = dict(extent_3km=extent_3km,
                   extent_bight=extent_bight)

    yr = pd.to_datetime(ymd).year

    sport_dd = (pd.to_datetime(ymd) - dt.timedelta(days=1)).strftime('%Y%m%d')

    wrf_dir = os.path.join('/home/coolgroup/ru-wrf/real-time/v4.1_parallel/processed/3km', ymd)
    sst_inputs_dir = os.path.join('/home/coolgroup/ru-wrf/real-time/sst-input', ymd)
    sport_avhrr_dir = os.path.join('/home/coolgroup/bpu/wrf/data/composites')

    wrf_file = glob.glob(os.path.join(wrf_dir, 'wrfproc_*_00Z_H000.nc'))[0]
    sat_file = os.path.join(sst_inputs_dir, 'SST_raw_yesterday.nc')
    
    try:
        SST_WRF_file = glob.glob(os.path.join(sst_inputs_dir, f'SST_WRF_{ymd}.grb'))[0]
    except IndexError:
        print(f'No such file or directory: SST_WRF_{ymd}.grb')
        SST_WRF_file = 'no_file'
        
    try:
        sport_avhrr_file = glob.glob(os.path.join(sport_avhrr_dir, f'procdate_{sport_dd}*.nc'))[0]
    except IndexError:
        print(f'No such file or directory: procdate_{sport_dd}*.nc')
        sport_avhrr_file = 'no_file'

    save_dir_zoom_out = os.path.join(save_dir, f'{str(yr)}_sport', 'zoom_out')
    save_dir_zoom_in = os.path.join(save_dir, f'{str(yr)}_sport', 'zoom_in')
    os.makedirs(save_dir_zoom_out, exist_ok=True)
    os.makedirs(save_dir_zoom_in, exist_ok=True)

    # get WRF output SST
    ds = xr.open_dataset(wrf_file)
    sst_wrf = np.squeeze(ds.SST) - 273.15  # convert K to degrees C
    landmask = np.squeeze(ds.LANDMASK)  # 1=land, 0=water
    lakemask = np.squeeze(ds.LAKEMASK)  # 1=lake, 0=non-lake

    # convert values over land and lakes to nans
    ldmask = np.logical_and(landmask == 1, landmask == 1)
    sst_wrf.values[ldmask] = np.nan  # Moved this line inside the try-except block

    lkmask = np.logical_and(lakemask == 1, lakemask == 1)
    sst_wrf.values[lkmask] = np.nan  # Moved this line inside the try-except block

    # get satellite input SST
    try:
        ds_sf = xr.open_dataset(sat_file)
        sst_sf = np.squeeze(ds_sf.sst)
    except FileNotFoundError:
        print(f'No such file or directory: {sat_file}')
        sst_sf = None

    # get the WRF grib file
    try:
        ds_SST_WRF = xr.open_dataset(SST_WRF_file, engine='pynio')
        SST_WRF_grib = np.squeeze(ds_SST_WRF.TMP_P0_L1_GLL0) - 273.15  # convert K to degrees C
    except FileNotFoundError:
        print(f'No such file or directory: {SST_WRF_file}')
        SST_WRF_grib = None

    # get GFS output (DISCLAIMER: I don't know which hour WRF uses so plotting 000Z for now)
    try:
        ds_sport = xr.open_dataset(sport_avhrr_file, engine='pynio')
        sst_sport = np.squeeze(ds_sport.sst.transpose())
    except FileNotFoundError:
        print(f'No such file or directory: {sport_avhrr_file}')
        sst_sport = None
        
    vlims = [5, 30]
    cmap = cmo.cm.thermal
    levels = MaxNLocator(nbins=16).tick_values(vlims[0], vlims[1])  # levels every 1 degrees C
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    main_title = f'RU-WRF Sea Surface Temperature: {pd.to_datetime(ymd).strftime("%Y-%m-%d")}'

    for key, extent in extents.items():
        save_file = os.path.join(save_dir_zoom_out, f'ru-wrf_sst_sport_{ymd}')
        kwargs = dict()
        kwargs['zoom_coastline'] = False
        kwargs['landcolor'] = 'none'
        if key == 'extent_bight':
            save_file = os.path.join(save_dir_zoom_in, f'ru-wrf_sst_sport_bight_{ymd}')
            #kwargs['zoom_coastline'] = True
        
        fig, axs = plt.subplots(2, 2, figsize=(9, 8), sharey=True, sharex=True,
                                subplot_kw=dict(projection=ccrs.Mercator()))
        ax1 = axs[0, 0]
        ax2 = axs[0, 1]
        ax3 = axs[1, 0]
        ax4 = axs[1, 1]
        fig.suptitle(main_title, fontsize=16, y=.98)
        
        sst_wrf_sub, lon_wrf, lat_wrf = subset_grid(extent, sst_wrf, 'XLONG', 'XLAT')
        sst_sf_sub, lon_sf, lat_sf = subset_grid(extent, sst_sf, 'lon', 'lat')
        SST_WRF_grib_sub, lon_SST_WRF_grib, lat_SST_WRF_grib = subset_grid(extent, SST_WRF_grib, 'lon_0', 'lat_0')
        sst_sport_sub, lon_sport, lat_sport = subset_grid(extent, sst_sport, 'lon', 'lat')
        
        kwargs['bottom_label'] = False
        hp.map_create(extent, ax=ax1, **kwargs)
        kwargs['bottom_label'] = True
        hp.map_create(extent, ax=ax3, **kwargs)
        kwargs['left_label'] = False
        kwargs['bottom_label'] = False
        hp.map_create(extent, ax=ax2, **kwargs)
        kwargs['bottom_label'] = True
        hp.map_create(extent, ax=ax4, **kwargs)
        
        contour_list = [10, 15, 20, 25, 30]
        try:
            pf.add_contours(ax1, lon_sf, lat_sf, sst_sf_sub.values, contour_list)
        except TypeError:
            sst_sf = np.squeeze(ds_sf.sst).transpose()
            sst_sf_sub, lon_sf, lat_sf = subset_grid(extent, sst_sf, 'lon', 'lat')
            pf.add_contours(ax1, lon_sf, lat_sf, sst_sf_sub.values, contour_list)
        
        pf.add_contours(ax2, lon_SST_WRF_grib, lat_SST_WRF_grib, SST_WRF_grib_sub.values, contour_list)
        
        try:
            pf.add_contours(ax3, lon_sport, lat_sport, sst_sport_sub.values, contour_list)
        except TypeError:
            sst_sport = np.squeeze(ds_sport.sst.transpose())
            sst_sport_sub, lon_sport, lat_sport = subset_grid(extent, sst_sport, 'lon', 'lat')
            pf.add_contours(ax3, lon_sport, lat_sport, sst_sport_sub.values, contour_list)
        
        pf.add_contours(ax4, lon_wrf, lat_wrf, sst_wrf_sub.values, contour_list)
        
        kwargs['panel_title'] = 'SST_raw_yesterday.nc'
        kwargs['norm_clevs'] = norm
        kwargs['extend'] = 'both'
        kwargs['cmap'] = cmap
        kwargs['title_pad'] = 8
        try:
            pf.plot_pcolormesh_panel(fig, ax1, lon_sf, lat_sf, sst_sf_sub.values, **kwargs)
        except Exception as e:
            print(f"Error plotting satellite panel: {e}")
        
        kwargs['panel_title'] = f'Sport/AVHRR {sport_dd}'
        try:
            pf.plot_pcolormesh_panel(fig, ax3, lon_sport, lat_sport, sst_sport_sub.values, **kwargs)
        except Exception as e:
            print(f"Error plotting Sport/AVHRR panel: {e}")
        
        kwargs['panel_title'] = f'SST_WRF_{ymd}.grb'
        kwargs['clab'] = 'SST (\N{DEGREE SIGN}C)'
        pf.plot_pcolormesh_panel(fig, ax2, lon_SST_WRF_grib, lat_SST_WRF_grib, SST_WRF_grib_sub.values, **kwargs)
        
        kwargs['panel_title'] = 'RU-WRF Output'
        pf.plot_pcolormesh_panel(fig, ax4, lon_wrf, lat_wrf, sst_wrf_sub.values, **kwargs)
        
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.02, hspace=0.12)
        plt.savefig(save_file, dpi=200)
        plt.close()

        print('Plot generated')


if __name__ == '__main__':
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
