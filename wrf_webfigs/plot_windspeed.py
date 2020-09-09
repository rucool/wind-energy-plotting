#!/usr/bin/env python

"""
Author: Lori Garzio on 5/28/2020
Last modified: 9/9/2020
Creates hourly plots of RU-WRF 4.1 output variables: wind speed at 10m, 80m and 160m. The plots are used to populate
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
import cartopy.crs as ccrs
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def plt_windsp(nc, model, ht, figname, lease_areas):
    """
    Create pseudocolor surface maps of wind speed with quivers indicating wind direction.
    :param nc: netcdf file
    :param model: the model version that is being plotted, e.g. 3km or 9km
    :param ht: wind speed height to plot, e.g. 10m, 80m, 160m
    :param figname: full file path to save directory and save filename
    :param lease_areas: dictionary containing lat/lon coordinates for wind energy lease area polygon
    """
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
            u_sub, __ = cf.subset_grid(u, model)
            v_sub, ax_lims = cf.subset_grid(v, model)
            qs = quiver_subset['_{}'.format(model)]['_{}'.format(ht)]
        else:  # subset just NY Bight
            new_fname = 'bight_{}'.format(figname.split('/')[-1])
            figname = '/{}/{}'.format(os.path.join(*figname.split('/')[0:-1]), new_fname)
            u_sub, __ = cf.subset_grid(u, 'bight')
            v_sub, ax_lims = cf.subset_grid(v, 'bight')
            qs = quiver_subset['bight_{}'.format(model)]['_{}'.format(ht)]

        fig, ax, lat, lon = cf.set_map(u_sub)

        # add text to the bottom of the plot
        cf.add_text(ax, nc.SIMULATION_START_DATE, nc.time_coverage_start, model)

        cf.add_map_features(ax, ax_lims)

        # pf.add_lease_area_polygon(ax, lease_areas, 'magenta')

        # convert wind speeds from m/s to knots
        u_sub = np.squeeze(u_sub.values) * 1.94384
        v_sub = np.squeeze(v_sub.values) * 1.94384

        # calculate wind speed from u and v
        speed = wind_uv_to_spd(u_sub, v_sub)

        # add contours
        contour_list = [22, 34, 48, 64]
        pf.add_contours(ax, lon, lat, speed, contour_list)

        # plot data
        # pcolormesh: coarser resolution, shows the actual resolution of the model data
        pf.plot_pcolormesh(fig, ax, '{} {}'.format(ht, color_label), lon, lat, speed, 0, 40, 'BuPu', color_label)

        # contourf: smooths the resolution of the model data, plots are less pixelated
        # levels = np.arange(0, 40.1, .1)
        # pf.plot_contourf(fig, ax, color_label, lon, lat, speed, levels, 'BuPu', color_label, var_min=0, var_max=40,
        #                  normalize='no', cbar_ticks=np.linspace(0, 40, 9))

        # subset the quivers and add as a layer
        ax.quiver(lon[::qs, ::qs], lat[::qs, ::qs], u_sub[::qs, ::qs], v_sub[::qs, ::qs], scale=1000,
                  width=.002, headlength=4, transform=ccrs.PlateCarree())

        plt.savefig(figname, dpi=200)
        plt.close()


def wind_uv_to_spd(u, v):
    """
    Calculates the wind speed from the u and v wind components
    :param u: west/east direction (wind from the west is positive, from the east is negative)
    :param v: south/noth direction (wind from the south is positive, from the north is negative)
    :returns WSPD: wind speed calculated from the u and v wind components
    """
    WSPD = np.sqrt(np.square(u) + np.square(v))

    return WSPD


def main(args):
    start_time = time.time()
    wrf_procdir = args.wrf_dir
    save_dir = args.save_dir

    la_polygon = cf.extract_lease_areas()

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
    plt_vars = ['ws10', 'ws80', 'ws160']

    for i, f in enumerate(files):
        fname = f.split('/')[-1].split('.')[0]
        splitter = fname.split('/')[-1].split('_')
        ncfile = xr.open_dataset(f, mask_and_scale=False)
        for pv in plt_vars:
            sfile = cf.save_filepath(save_dir, pv, splitter)
            if pv == 'ws10':
                plt_windsp(ncfile, model_ver, '10m', sfile, la_polygon)
            elif pv == 'ws80':
                plt_windsp(ncfile, model_ver, '80m', sfile, la_polygon)
            elif pv == 'ws160':
                plt_windsp(ncfile, model_ver, '160m', sfile, la_polygon)

    print('')
    print('Script run time: {} minutes'.format(round(((time.time() - start_time) / 60), 2)))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Plot windspeed',
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
