#!/usr/bin/env python

"""
Author: Lori Garzio on 8/28/2020
Last Modified: 8/28/2020
Creates surface map of local radar. These plots are used to populate RUCOOL's Coastal Metocean Monitoring Station
webpage: https://rucool.marine.rutgers.edu/data/meteorological-modeling/coastal-metocean-monitoring-station/
"""

import argparse
import numpy as np
import pandas as pd
import sys
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pyart  # used for the colormap 'pyart_NWSRef'
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def main(args):
    save_file = args.save_file
    f = '/home/coolgroup/MetData/radar/latest.nc'  # on server

    nc = xr.open_dataset(f, mask_and_scale=False)
    radar = np.squeeze(nc['bref'])

    lccproj = ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=lccproj))

    lat = nc['lat'].values
    lon = nc['lon'].values

    # lon and lat variables are 1D, use meshgrid to create 2D arrays
    lons, lats = np.meshgrid(lon, lat)

    # add text to the bottom of the plot
    tm = pd.to_datetime(nc['time'].values[0])
    hhmm = tm.strftime('%H:%M')
    dmy = tm.strftime('%d%b%Y')
    insert_text1 = 'Valid {}Z {}'.format(hhmm, dmy)
    ax.text(.7, -.12, insert_text1, size=10, transform=ax.transAxes)

    # define axis limits and add map layers
    ax_lims = [-77.5, -71.5, 37.5, 42.1]
    cf.add_map_features(ax, ax_lims)

    title = 'NWS Radar KDIX Base Reflectivity (dBZ)'
    color_label = 'Base Reflectivity (dBZ)'

    vmin = 0
    vmax = 72
    levels = np.linspace(vmin, vmax, 145)
    ticklevs = np.linspace(vmin, 70, 15)

    pf.plot_contourf(fig, ax, title, lons, lats, radar.values, levels, 'pyart_NWSRef', color_label, var_min=vmin,
                     var_max=vmax, normalize='no', cbar_ticks=ticklevs)

    plt.savefig(save_file, dpi=200)
    plt.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Plot local radar',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-s', '--save_file',
                            dest='save_file',
                            default='/home/coolgroup/MetData/radar',
                            type=str,
                            help='Full file path to save directory and save filename.')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
