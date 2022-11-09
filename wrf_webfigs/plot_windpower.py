#!/usr/bin/env python

"""
Author: Lori Garzio on 1/14/2022
Last modified: 11/9/2022
Creates hourly plots of RU-WRF 4.1 output variables: estimated wind power at 160m. The plots are used to populate
RUCOOL's RU-WRF webpage:
https://rucool.marine.rutgers.edu/data/meteorological-modeling/ruwrf-mesoscale-meteorological-model-forecast/
"""

import argparse
import numpy as np
import pandas as pd
import os
import glob
import sys
import time
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def plt_power(nc, model, ht, figname, lease_area_outlines=None):
    """
    Create pseudocolor surface maps of estimated wind power at 160m.
    :param nc: netcdf file
    :param model: the model version that is being plotted, e.g. 3km or 9km
    :param ht: wind speed height to plot, e.g. 160m
    :param figname: full file path to save directory and save filename
    :param lease_areas: optional dictionary containing lat/lon coordinates for wind energy lease area polygon
    """
    power_curve = '/home/lgarzio/rucool/bpu/wrf/wrf_lw15mw_power_15001max.csv'  # on server, max is set to 15001
    pc = pd.read_csv(power_curve)
    turbine = power_curve.split('/')[-1].split('_')[1].split('lw')[-1].upper()
    lease_area_outlines = lease_area_outlines or None

    if ht == '10m':
        u = nc['U10']
        v = nc['V10']
    else:
        u = nc.sel(height=int(ht[0:-1]))['U']
        v = nc.sel(height=int(ht[0:-1]))['V']

    color_label = f'Estimated {turbine} Wind Power (kW)'

    plot_types = ['full_grid', 'bight']
    for pt in plot_types:
        if pt == 'full_grid':  # subset the entire grid
            u_sub, _, _, _ = cf.subset_grid(u, model)
            v_sub, ax_lims, xticks, yticks = cf.subset_grid(v, model)
        else:  # subset just NY Bight
            new_fname = 'bight_{}'.format(figname.split('/')[-1])
            figname = '/{}/{}'.format(os.path.join(*figname.split('/')[0:-1]), new_fname)
            u_sub, _, _, _ = cf.subset_grid(u, 'bight')
            v_sub, ax_lims, xticks, yticks = cf.subset_grid(v, 'bight')

        fig, ax, lat, lon = cf.set_map(u_sub)

        # add text to the bottom of the plot
        cf.add_text(ax, nc.SIMULATION_START_DATE, nc.time_coverage_start, model)

        # initialize keyword arguments for map features
        kwargs = dict()
        kwargs['xticks'] = xticks
        kwargs['yticks'] = yticks
        cf.add_map_features(ax, ax_lims, **kwargs)

        if lease_area_outlines:
            kwargs = dict()
            kwargs['edgecolor'] = 'cyan'  # 'dimgray'
            pf.map_add_boem_outlines(ax, **kwargs)

        # calculate wind speed from u and v
        speed = cf.wind_uv_to_spd(u_sub, v_sub)

        # calculate wind power
        power = xr.DataArray(np.interp(speed, pc['Wind Speed'], pc['Power']), coords=speed.coords)

        # add contours
        contour_list = [15000]
        pf.add_contours(ax, lon, lat, power, contour_list)

        # set color map
        cmap = plt.get_cmap('OrRd')
        levels = list(np.arange(0, 15001, 1000))

        # plot data
        # pcolormesh: coarser resolution, shows the actual resolution of the model data
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        kwargs = dict()
        kwargs['ttl'] = '{} {}'.format(ht, color_label)
        kwargs['cmap'] = cmap
        kwargs['clab'] = color_label
        kwargs['norm_clevs'] = norm
        kwargs['extend'] = 'neither'
        pf.plot_pcolormesh(fig, ax, lon, lat, power, **kwargs)

        # add power values of 15000 as another layer
        power_copy = power.copy()
        custom_color = ["#67000d"]  # dark red
        custom_colormap = ListedColormap(custom_color)
        mask = np.logical_and(power_copy.values < 15001, power_copy.values < 15001)
        power_copy.values[mask] = np.nan
        ax.pcolormesh(lon, lat, power_copy, cmap=custom_colormap, transform=ccrs.PlateCarree())

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
    plt_vars = ['power160']

    kwargs = dict()
    kwargs['lease_area_outlines'] = True

    for i, f in enumerate(files):
        fname = f.split('/')[-1].split('.')[0]
        splitter = fname.split('/')[-1].split('_')
        ncfile = xr.open_dataset(f, mask_and_scale=False)
        for pv in plt_vars:
            sfile = cf.save_filepath(save_dir, pv, splitter)
            if pv == 'power160':
                plt_power(ncfile, model_ver, '160m', sfile, **kwargs)

    print('')
    print('Script run time: {} minutes'.format(round(((time.time() - start_time) / 60), 2)))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Plot estimated wind power',
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
