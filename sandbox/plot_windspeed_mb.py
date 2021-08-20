#!/usr/bin/env python

"""
Author: Lori Garzio on 8/12/2021
Last modified: 8/12/2021
Creates plots of RU-WRF 4.1 windspeeds at 925mb at the top of the hour, and exports a summary .csv file of windspeed
and direction at specific locations for seabreeze classification.
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
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def plt_windsp(nc, model, ht, figname, lease_areas=None, summary=None, add_text=None):
    """
    Create pseudocolor surface maps of wind speed with quivers indicating wind direction.
    :param nc: netcdf file
    :param model: the model version that is being plotted, e.g. 3km or 9km
    :param ht: wind speed height in mb to plot, e.g. 925
    :param figname: full file path to save directory and save filename
    :param lease_areas: optional dictionary containing lat/lon coordinates for wind energy lease area polygon
    :param summary: optional dictionary containing locations of specific locations for seabreeze classification,
    and a list to append windspeeds at specific locations for summary output
    :param add_text: optional, add windspeed/direction values at specific locations to the figure
    """
    lease_areas = lease_areas or None
    summary = summary or None
    add_text = add_text or None

    u = nc.sel(pressure=ht)['UP']
    v = nc.sel(pressure=ht)['VP']

    color_label = 'Wind Speed (m/s)'
    quiver_subset = dict(_3km=dict(_925=12),
                         _9km=dict(_925=4),
                         bight_3km=dict(_925=6),
                         bight_9km=dict(_925=2))

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
        # u_sub = xr.DataArray(np.squeeze(u_sub.values) * 1.94384, coords=u_sub.coords)
        # v_sub = xr.DataArray(np.squeeze(v_sub.values) * 1.94384, coords=v_sub.coords)

        # standardize the vectors so they only represent direction
        u_sub_standardize = u_sub / cf.wind_uv_to_spd(u_sub, v_sub)
        v_sub_standardize = v_sub / cf.wind_uv_to_spd(u_sub, v_sub)

        # calculate wind speed and direction from u and v
        speed = cf.wind_uv_to_spd(u_sub, v_sub)
        direction = cf.wind_uv_to_dir(u_sub, v_sub)

        # write a summary for wind speeds/directions at specified locations for seabreeze classification
        # add the windspeeds/directions to the map
        map_values = dict()
        if summary:
            tmstr = pd.to_datetime(nc.Time.values[0]).strftime('%Y-%m-%dT%H:%M')
            for key, coords in summary['locations'].items():
                # find the closest model grid point to the location
                a = abs(speed.XLAT - coords['lat']) + abs(speed.XLONG - coords['lon'])
                i, j = np.unravel_index(a.argmin(), a.shape)
                sp = speed[i, j]
                d = direction[i, j]
                coords.update(ws=np.round(float(sp.values), 2))
                coords.update(direction=np.round(float(d.values), 2))
                map_values[key] = coords
                if pt == 'full_grid':
                    wrf_lat = np.round(float(sp.XLAT.values), 4)
                    wrf_lon = np.round(float(sp.XLONG.values), 4)
                    summary['rows'].append([tmstr, key, coords['lat'], coords['lon'], ht, wrf_lat, wrf_lon,
                                            np.round(float(sp.values), 4), np.round(float(d.values), 4)])

        # mask vectors if wind speed is < 1 m/s
        mask = speed.values < 1
        u_sub_standardize.values[mask] = np.nan
        v_sub_standardize.values[mask] = np.nan

        # add contours
        #contour_list = [10, 22, 34, 48, 64]
        contour_list = [5, 11, 17, 25, 32]
        pf.add_contours(ax, lon, lat, speed, contour_list)

        # plot data
        # pcolormesh: coarser resolution, shows the actual resolution of the model data
        cmap = plt.get_cmap('BuPu')
        # vlims = [0, 40]
        # levels = MaxNLocator(nbins=20).tick_values(vlims[0], vlims[1])  # every 2 knots
        vlims = [0, 25]
        levels = MaxNLocator(nbins=25).tick_values(vlims[0], vlims[1])  # every 1 m/s
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        kwargs = dict()
        kwargs['ttl'] = '{}mb {}'.format(ht, color_label)
        kwargs['cmap'] = cmap
        kwargs['clab'] = color_label
        #kwargs['var_lims'] = vlims
        kwargs['norm_clevs'] = norm
        kwargs['extend'] = 'max'
        pf.plot_pcolormesh(fig, ax, lon, lat, speed, **kwargs)

        # # contourf: smooths the resolution of the model data, plots are less pixelated
        # kwargs = dict()
        # kwargs['ttl'] = '{} {}'.format(ht, color_label)
        # kwargs['cmap'] = cmap
        # kwargs['clab'] = color_label
        # kwargs['var_lims'] = [0, 40]
        # kwargs['cbar_ticks'] = np.linspace(0, 40, 9)
        #
        # levels = np.arange(0, 40.1, .1)
        # pf.plot_contourf(fig, ax, lon, lat, speed, levels, **kwargs)

        # subset the quivers and add as a layer
        # ax.quiver(lon[::qs, ::qs], lat[::qs, ::qs], u_sub[::qs, ::qs], v_sub[::qs, ::qs], scale=1000,
        #           width=.002, headlength=4, transform=ccrs.PlateCarree())

        ax.quiver(lon[::qs, ::qs], lat[::qs, ::qs], u_sub_standardize.values[::qs, ::qs], v_sub_standardize.values[::qs, ::qs],
                  scale=50, width=.002, headlength=4, transform=ccrs.PlateCarree())

        # add the seabreeze classification locations to the map
        if summary:
            if pt == 'full_grid':
                offset = 1.75
            else:
                offset = .85
            for key, values in map_values.items():
                ax.scatter(values['lon'], values['lat'], c='magenta', s=40, zorder=15, transform=ccrs.PlateCarree())
                if add_text:
                    ax.text(values['lon'] - offset, values['lat'], '{} {}'.format(values['ws'], values['direction']),
                            transform=ccrs.PlateCarree(),
                            bbox=dict(facecolor='lightgray', alpha=1), fontsize=8, zorder=15)

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
    files = [f for f in files if f.endswith('_M00.nc')]

    # get the model version (3km or 9km) from the filename
    f0 = files[0]
    model_ver = f0.split('/')[-1].split('_')[1]  # 3km or 9km
    os.makedirs(save_dir, exist_ok=True)

    # List of variables to plot
    plt_vars = ['ws925']

    seabreeze_locations = {'atlantic_city': {'lat': 39.4520, 'lon': -74.5670},
                           'ocean_city_md': {'lat': 38.3083, 'lon': -75.1239},
                           'oyster_creek': {'lat': 39.817102, 'lon': -74.213449},
                           'sea_isle_city': {'lat': 39.191604, 'lon': -74.788602},
                           'asbury_park': {'lat': 40.233804, 'lon': -74.089406}
                           }

    kwargs = dict()
    # kwargs['lease_areas'] = cf.extract_lease_areas()
    summary = dict(rows=[], locations=seabreeze_locations)
    kwargs['summary'] = summary
    kwargs['add_text'] = True

    for i, f in enumerate(files):
        fname = f.split('/')[-1].split('.')[0]
        splitter = fname.split('/')[-1].split('_')
        ncfile = xr.open_dataset(f, mask_and_scale=False)
        for pv in plt_vars:
            sfile = cf.save_filepath(save_dir, pv, splitter)
            if pv == 'ws925':
                plt_windsp(ncfile, model_ver, 925, sfile, **kwargs)

    df_headers = ['time', 'location', 'lat', 'lon', 'pressure_mb', 'wrf_lat', 'wrf_lon', 'wind_speed_meterspersec',
                  'wind_direction']
    df = pd.DataFrame(summary['rows'], columns=df_headers)
    df.to_csv('{}/seabreeze_summary.csv'.format(save_dir), index=False)
    print('')
    print('Script run time: {} minutes'.format(round(((time.time() - start_time) / 60), 2)))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Plot windspeed at pressure levels at the top of the hour',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-wrf_dir',
                            dest='wrf_dir',
                            type=str,
                            help='Full directory path to subset WRF netCDF files.')

    arg_parser.add_argument('-save_dir',
                            dest='save_dir',
                            type=str,
                            help='Full directory path to save output plots.')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
