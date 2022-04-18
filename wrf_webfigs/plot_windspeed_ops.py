#!/usr/bin/env python

"""
Author: Lori Garzio on 4/14/2022
Last modified: 4/18/2022
Creates hourly plots of RU-WRF 4.1 wind speed at 10m for operations purposes. The plots are used to populate
RUCOOL's RU-WRF webpage:
https://rucool.marine.rutgers.edu/data/meteorological-modeling/ruwrf-mesoscale-meteorological-model-forecast/
"""

import argparse
import numpy as np
import os
import glob
import sys
import time
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def plt_windsp(nc, model, ht, figname, lease_areas=None):
    """
    Create pseudocolor surface maps of wind speed with quivers indicating wind direction.
    :param nc: netcdf file
    :param model: the model version that is being plotted, e.g. 3km or 9km
    :param ht: wind speed height to plot, e.g. 10m, 80m, 160m
    :param figname: full file path to save directory and save filename
    :param lease_areas: optional dictionary containing lat/lon coordinates for wind energy lease area polygon
    """
    lease_areas = lease_areas or None

    if ht == '10m':
        u = nc['U10']
        v = nc['V10']
    else:
        u = nc.sel(height=int(ht[0:-1]))['U']
        v = nc.sel(height=int(ht[0:-1]))['V']

    color_label = 'Wind Speed (knots)'

    # define the subsetting for the quivers on the map based on model and height
    quiver_subset = dict(_3km=dict(_10m=11, _80m=12, _160m=13),
                         _9km=dict(_10m=4, _80m=5, _160m=6),
                         bight_3km=dict(_10m=6, _80m=6, _160m=7),
                         bight_9km=dict(_10m=2, _80m=2, _160m=3))

    plot_types = ['full_grid', 'bight']
    for pt in plot_types:
        if pt == 'full_grid':  # subset the entire grid
            u_sub, _, _, _ = cf.subset_grid(u, model)
            v_sub, ax_lims, xticks, yticks = cf.subset_grid(v, model)
            qs = quiver_subset['_{}'.format(model)]['_{}'.format(ht)]
        else:  # subset just NY Bight
            new_fname = 'bight_{}'.format(figname.split('/')[-1])
            figname = '/{}/{}'.format(os.path.join(*figname.split('/')[0:-1]), new_fname)
            u_sub, _, _, _ = cf.subset_grid(u, 'bight')
            v_sub, ax_lims, xticks, yticks = cf.subset_grid(v, 'bight')
            qs = quiver_subset['bight_{}'.format(model)]['_{}'.format(ht)]

        fig, ax, lat, lon = cf.set_map(u_sub)

        # add text to the bottom of the plot
        cf.add_text(ax, nc.SIMULATION_START_DATE, nc.time_coverage_start, model)

        # initialize keyword arguments for map features
        kwargs = dict()
        kwargs['xticks'] = xticks
        kwargs['yticks'] = yticks
        cf.add_map_features(ax, ax_lims, **kwargs)

        if lease_areas:
            pf.add_lease_area_polygon(ax, lease_areas, 'magenta')

        # convert wind speeds from m/s to knots
        u_sub = np.squeeze(u_sub.values) * 1.94384
        v_sub = np.squeeze(v_sub.values) * 1.94384

        # standardize the vectors so they only represent direction
        u_sub_standardize = u_sub / cf.wind_uv_to_spd(u_sub, v_sub)
        v_sub_standardize = v_sub / cf.wind_uv_to_spd(u_sub, v_sub)

        # calculate wind speed from u and v
        speed = cf.wind_uv_to_spd(u_sub, v_sub)

        # mask vectors if wind speed is < 2
        mask = speed < 2
        u_sub_standardize[mask] = np.nan
        v_sub_standardize[mask] = np.nan

        # add contours
        contour_list = [5, 10, 15, 20]
        pf.add_contours(ax, lon, lat, speed, contour_list)

        # plot data
        # pcolormesh: coarser resolution, shows the actual resolution of the model data
        cmap = plt.get_cmap('jet')  # 'turbo'
        vlims = [0, 20]
        levels = MaxNLocator(nbins=20).tick_values(vlims[0], vlims[1])  # every 2 knots
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        kwargs = dict()
        kwargs['ttl'] = 'Wind Speed ({}) Wind Ops'.format(ht)
        kwargs['cmap'] = cmap
        kwargs['clab'] = color_label
        kwargs['norm_clevs'] = norm
        kwargs['extend'] = 'neither'
        pf.plot_pcolormesh(fig, ax, lon, lat, speed, **kwargs)

        ax.quiver(lon[::qs, ::qs], lat[::qs, ::qs], u_sub_standardize[::qs, ::qs], v_sub_standardize[::qs, ::qs],
                  scale=50, width=.002, headlength=4, transform=ccrs.PlateCarree())

        plt.savefig(figname, dpi=200)
        plt.close()


def main(args):
    start_time = time.time()
    wrf_procdir = args.wrf_dir
    save_dir = args.save_dir

    if wrf_procdir.endswith('/'):
        ext = '*.nc'
    else:
        ext = '/*.nc'
    files = sorted(glob.glob(wrf_procdir + ext))

    # get the model version (3km or 9km) from the filename
    f0 = files[0]
    model_ver = f0.split('/')[-1].split('_')[1]  # 3km or 9km
    os.makedirs(save_dir, exist_ok=True)

    # List of variables to plot
    plt_vars = ['ws10_ops']

    kwargs = dict()
    # kwargs['lease_areas'] = cf.extract_lease_areas()

    for i, f in enumerate(files):
        fname = f.split('/')[-1].split('.')[0]
        splitter = fname.split('/')[-1].split('_')
        ncfile = xr.open_dataset(f, mask_and_scale=False)
        sfile = cf.save_filepath(save_dir, plt_vars[0], splitter)
        plt_windsp(ncfile, model_ver, '10m', sfile, **kwargs)

    print('')
    print('Script run time: {} minutes'.format(round(((time.time() - start_time) / 60), 2)))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Plot windspeed for operations',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-wrf_dir',
                            dest='wrf_dir',
                            default='/home/coolgroup/ru-wrf/real-time/v4.1_parallel/processed/3km/20200101',
                            type=str,
                            help='Full directory path to subset WRF netCDF files.')

    arg_parser.add_argument('-save_dir',
                            dest='save_dir',
                            default='/home/coolgroup/ru-wrf/real-time/v4.1_parallel/scripts/webfigs/3km',
                            type=str,
                            help='Full directory path to save output plots.')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
