import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
import cmocean as cmo
import argparse
import sys

def subset_grid(ext, dataset, lon_name, lat_name):
    """
    Subsets the grid based on the provided geographical extent.

    Args:
        ext (list): Geographical extent [lon_min, lon_max, lat_min, lat_max].
        dataset (xarray.Dataset): The dataset containing SST data.
        lon_name (str): Name of the longitude variable in the dataset.
        lat_name (str): Name of the latitude variable in the dataset.

    Returns:
        tuple: Subsets of the dataset and corresponding longitude and latitude.
    """
    if len(np.shape(dataset[lon_name])) == 1:
        lonx, laty = np.meshgrid(dataset[lon_name], dataset[lat_name])
    else:
        lonx = dataset[lon_name]
        laty = dataset[lat_name]

    lon_idx = np.logical_and(lonx > ext[0], lonx < ext[1])
    lat_idx = np.logical_and(laty > ext[2], laty < ext[3])

    ind = np.where(np.logical_and(lat_idx, lon_idx))
    data_sub = np.squeeze(dataset)[range(np.min(ind[0]), np.max(ind[0]) + 1), range(np.min(ind[1]), np.max(ind[1]) + 1)]
    lon = data_sub[lon_name]
    lat = data_sub[lat_name]

    return data_sub, lon, lat

def main(args):
    ymd = args.ymd
    save_dir = args.save_dir

    yr = pd.to_datetime(ymd).year
    ym = ymd[0:6]

    save_dir_wrf = os.path.join(save_dir, str(yr), 'wrf_output', ym)
    os.makedirs(save_dir_wrf, exist_ok=True)

    wrf_dir = os.path.join('/home/coolgroup/ru-wrf/real-time/v4.1_parallel/processed_windturbs/1km_wf2km_nyb', ymd)  # Update the path as needed
    new_data_dir = os.path.join('/home/coolgroup/ru-wrf/real-time/v4.1_parallel/processed_windturbs/1km_wf2km_nyb_modsst',ymd)  # Update this path for the new dataset

    print(wrf_dir)
    print(new_data_dir)

    extent = [-80, -60, 31, 46]  # Define the geographical extent as needed

    # Process the first dataset
    wrf_file = glob.glob(os.path.join(wrf_dir, 'wrfproc_1km_*_00Z_H000.nc'))[0]
    print(wrf_file)
    ds = xr.open_dataset(wrf_file)
    sst_wrf = np.squeeze(ds.SST) - 273.15  # convert K to degrees C
    
    landmask = np.squeeze(ds.LANDMASK)  # 1=land, 0=water
    lakemask = np.squeeze(ds.LAKEMASK)  # 1=lake, 0=non-lake

    # Convert values over land and lakes to nans
    ldmask = np.logical_and(landmask == 1, landmask == 1)
    sst_wrf.values[ldmask] = np.nan

    lkmask = np.logical_and(lakemask == 1, lakemask == 1)
    sst_wrf.values[lkmask] = np.nan

    # Load and process the new dataset
    new_wrf_file = glob.glob(os.path.join(new_data_dir, 'wrfproc_1km_*_00Z_H000.nc'))[0]
    new_ds = xr.open_dataset(new_wrf_file)
    new_sst_wrf = np.squeeze(new_ds.SST) - 273.15  # Convert K to degrees C

    # Convert values over land and lakes to nans for the new dataset
    new_landmask = np.squeeze(new_ds.LANDMASK)  # 1=land, 0=water
    new_lakemask = np.squeeze(new_ds.LAKEMASK)  # 1=lake, 0=non-lake

    new_ldmask = np.logical_and(new_landmask == 1, new_landmask == 1)
    new_sst_wrf.values[new_ldmask] = np.nan

    new_lkmask = np.logical_and(new_lakemask == 1, new_lakemask == 1)
    new_sst_wrf.values[new_lkmask] = np.nan

    # Setup plot with 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(18, 9), subplot_kw=dict(projection=ccrs.Mercator()))
    main_title = f'RU-WRF Sea Surface Temperature: {pd.to_datetime(ymd).strftime("%Y-%m-%d")}'
    fig.suptitle(main_title, fontsize=16, y=.98)

    # Plotting for the first dataset
    vlims = [5, 30]
    cmap = cmo.cm.thermal
    levels = MaxNLocator(nbins=16).tick_values(vlims[0], vlims[1])
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    sst_wrf_sub, lon_wrf, lat_wrf = subset_grid(extent, sst_wrf, 'XLONG', 'XLAT')
    cf = axs[0].pcolormesh(lon_wrf, lat_wrf, sst_wrf_sub, norm=norm, cmap=cmap, transform=ccrs.PlateCarree())
    axs[0].coastlines()
    axs[0].set_title('Upwelling')
    fig.colorbar(cf, ax=axs[0], orientation='vertical', label='SST (\N{DEGREE SIGN}C)',shrink=0.75, aspect=20, pad=0.05)
    
    land_color = 'tan'  # Replace 'tan' with any color you prefer for land
    axs[0].pcolormesh(lon_wrf, lat_wrf, landmask, color=land_color, transform=ccrs.PlateCarree(), zorder=0)

# Plotting for the second dataset
    new_sst_wrf_sub, new_lon_wrf, new_lat_wrf = subset_grid(extent, new_sst_wrf, 'XLONG', 'XLAT')
    cf2 = axs[1].pcolormesh(new_lon_wrf, new_lat_wrf, new_sst_wrf_sub, norm=norm, cmap=cmap, transform=ccrs.PlateCarree())
    axs[1].coastlines()
    axs[1].set_title('No Upwelling')
    fig.colorbar(cf2, ax=axs[1], orientation='vertical', label='SST (\N{DEGREE SIGN}C)',shrink=0.75, aspect=20, pad=0.05)
   
    land_color = 'tan'  # Replace 'tan' with any color you prefer for land
    axs[1].pcolormesh(lon_wrf, lat_wrf, landmask, color=land_color, transform=ccrs.PlateCarree(), zorder=0)

    # Save the figure with two panels
    plt.savefig(os.path.join(save_dir_wrf, f'ru-wrf_sst_comparison_{ymd}.png'), dpi=200)
    plt.close()

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Plot WRF SST output',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('ymd',
                            type=str,
                            help='Year-month-day to plot in the format YYYYmmdd (e.g. 20220101.')

    arg_parser.add_argument('-save_dir',
                            default='/www/web/rucool/windenergy/ru-wrf/images/upwelling_case/upwelling_sst_output',  # Update the path as needed
                            type=str,
                            help='Full directory path to save output plots.')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))

