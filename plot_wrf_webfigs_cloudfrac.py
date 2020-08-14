#!/usr/bin/env python

"""
Author: Lori Garzio on 8/6/2020
Last modified: 8/14/2020
Creates 4-panel hourly plots of cloud fraction from RU-WRF 4.1 at low, medium, high levels, and Total Cloud Fraction.
The plots are used to populate RUCOOL's RU-WRF webpage:
https://rucool.marine.rutgers.edu/data/meteorological-modeling/ruwrf-mesoscale-meteorological-model-forecast/
"""

import argparse
import numpy as np
import os
import glob
import sys
import datetime as dt
import time
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1 import make_axes_locatable


def add_map_features(ax, axes_limits, bottom_labs, left_labs):
    """
    Adds latitude and longitude gridlines and labels, coastlines, and statelines to a cartopy map object
    :param ax: plotting axis object
    :param axes_limits: list of axis limits [min lon, max lon, min lat, max lat]
    :param bottom_labs: option to add bottom grid labels ('yes' or 'no')
    :param left_labs: option to add left grid labels ('yes' or 'no')
    """
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='dotted', x_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    if bottom_labs != 'yes':
        gl.bottom_labels = False
    if left_labs != 'yes':
        gl.left_labels = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}

    # add some space between the grid labels and bottom of plot
    gl.xpadding = 12
    gl.ypadding = 12

    # set axis limits
    ax.set_extent(axes_limits)

    land = cfeature.NaturalEarthFeature('physical', 'land', '10m')
    ax.add_feature(land, zorder=5, edgecolor='red', facecolor='none')

    state_lines = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        edgecolor='red',
        facecolor='none')

    ax.add_feature(cfeature.BORDERS, zorder=6)
    ax.add_feature(state_lines, zorder=7)


def add_text(ax, run_date, time_coverage_start, model):
    """
    Adds text regarding model run information to the bottom a map
    :param ax: plotting axis object
    :param run_date: date string that the model run was initialized from the .nc file, format '%Y-%m-%d_%H:%M:%S'
    :param time_coverage_start: date string of the data being plotted from the .nc file, format '%Y%m%dT%H%M%SZ'
    :param model: the model version that is being plotted, e.g. 3km or 9km
    """
    # format dates
    init_dt = dt.datetime.strptime(run_date, '%Y-%m-%d_%H:%M:%S')
    init_dt_str = '00Z{}'.format(dt.datetime.strftime(init_dt, '%d%b%Y'))

    valid_dt_gmt = dt.datetime.strptime(time_coverage_start, '%Y%m%dT%H%M%SZ')
    valid_dt_gmt_str = '{:02d}Z{}'.format(valid_dt_gmt.hour, dt.datetime.strftime(valid_dt_gmt, '%d%b%Y'))

    # calculate the forecast hour
    fcast_hour = int((valid_dt_gmt - init_dt).total_seconds() / 60 / 60)

    weekDays = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')
    valid_dt_edt = valid_dt_gmt - dt.timedelta(hours=4)
    wkday = weekDays[valid_dt_edt.weekday()]
    valid_dt_edt_str = '{} {} {:02d}:00EDT'.format(wkday, dt.datetime.strftime(valid_dt_edt, '%d%b%Y'), valid_dt_edt.hour)

    insert_text1 = 'RU-WRF (v4.1) {} Model: Initialized {}'.format(model, init_dt_str)
    ax.text(.86, -.15, insert_text1, size=12, transform=ax.transAxes)

    insert_text2 = 'Valid {} ({}) | Forecast Hr {}'.format(valid_dt_gmt_str, valid_dt_edt_str, fcast_hour)
    ax.text(.59, -.22, insert_text2, size=12, transform=ax.transAxes)


def plot_pcolormesh_panel(fig, ax, ttl, lon_data, lat_data, var_data, var_min, var_max, cmap, clab, add_cbar):
    """
    Create a pseudocolor plot
    :param fig: figure object
    :param ax: plotting axis object
    :param ttl: plot title
    :param lon_data: longitude data
    :param lat_data: latitude data
    :param var_data: variable data
    :param var_min: minimum value for plotting (for fixed colorbar)
    :param var_max: maximum value for plotting (for fixed colorbar)
    :param cmap: color map
    :param clab: colorbar label
    :param add_cbar: option to add the colorbar ('yes' or 'no')
    """
    ax.set_title(ttl, fontsize=15)
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size='5%', pad=0.1, axes_class=plt.Axes)

    h = ax.pcolormesh(lon_data, lat_data, var_data, vmin=var_min, vmax=var_max, cmap=cmap, transform=ccrs.PlateCarree())

    if add_cbar == 'yes':
        fig.add_axes(cax)
        cb = plt.colorbar(h, cax=cax)
        cb.set_label(label=clab, fontsize=14)
        cb.ax.tick_params(labelsize=12)


def plt_cloudfrac(nc, model, figname):
    """
    Create a pcolor surface map of cloud cover
    :param nc: netcdf file
    :param model: the model version that is being plotted, e.g. 3km or 9km
    :param figname: full file path to save directory and save filename
    """
    var = nc['cloudfrac']
    color_label = 'Cloud Cover (%)'
    splitter = figname.split('/')[-1].split('_')

    # calculate maximum cloud fraction among each level = Total Cloud Fraction
    max_cloudfrac = np.max(np.squeeze(var), axis=0)

    # plot_types = ['full_grid', 'bight']
    plot_types = ['full_grid']
    for pt in plot_types:
        if pt == 'full_grid':
            # create a new file path
            new_fname = '{}_{}'.format(splitter[0], '_'.join(splitter[1:len(splitter)]))
            save_fig = '/{}/{}'.format(os.path.join(*figname.split('/')[0:-1]), new_fname)

            # subset the entire grid
            max_cloudfrac_sub, ax_lims = subset_grid(max_cloudfrac, model)
        else:
            # create a new file path
            new_fname = 'bight_{}_{}'.format(splitter[0], '_'.join(splitter[1:len(splitter)]))
            save_fig = '/{}/{}'.format(os.path.join(*figname.split('/')[0:-1]), new_fname)

            # subset just NY Bight
            max_cloudfrac_sub, ax_lims = subset_grid(max_cloudfrac, 'bight')

        fig, axs, lat, lon = set_map_panel(max_cloudfrac_sub)
        fig.suptitle(color_label, fontsize=17, y=.94)

        d = max_cloudfrac_sub.values * 100

        # plot Total Cloud Fraction in the bottom right panel
        ax4 = axs[1, 1]
        plot_pcolormesh_panel(fig, ax4, 'Total Cloud Fraction', lon, lat, d, 0, 100, 'gray', color_label, 'yes')
        add_map_features(ax4, ax_lims, 'yes', 'no')

        # plot each level
        ax1 = axs[0, 0]
        ax2 = axs[0, 1]
        ax3 = axs[1, 0]
        axes_ind = [ax1, ax2, ax3]
        for i, level in enumerate(var['low_mid_high'].values):
            ds = var.sel(low_mid_high=level)
            if pt == 'full_grid':
                # subset the entire grid
                ds_sub, ax_lims = subset_grid(ds, model)
            else:
                # subset just NY Bight
                ds_sub, ax_lims = subset_grid(ds, 'bight')

            d = ds_sub.values * 100

            if level == 300:
                colorbar = 'no'
                bottomlabel = 'no'
                leftlabel = 'yes'
                title = 'Low Level ({} m)'.format(level)
            elif level == 2000:
                colorbar = 'yes'
                bottomlabel = 'no'
                leftlabel = 'no'
                title = 'Mid Level ({} m)'.format(level)
            elif level == 6000:
                colorbar = 'no'
                bottomlabel = 'yes'
                leftlabel = 'yes'
                title = 'High Level ({} m)'.format(level)

                # add text to the bottom of the plot
                add_text(axes_ind[i], nc.SIMULATION_START_DATE, nc.time_coverage_start, model)

            plot_pcolormesh_panel(fig, axes_ind[i], title, lon, lat, d, 0, 100, 'gray', color_label, colorbar)
            add_map_features(axes_ind[i], ax_lims, bottomlabel, leftlabel)

        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.02, hspace=0.01)
        plt.savefig(save_fig, dpi=200)
        plt.close()


def save_filepath(save_dir, varname, sp):
    """
    Create a full file path to save directory and save filename.
    :param save_dir: directory to which file is saved
    :param varname: variable name
    :param sp: list containing original netcdf file name split into components, e.g. ['wrfproc', '3km', '20200720', '00Z', 'H000']
    :returns savefilepath: full file path to save directory and save filename
    """
    sf = '{}_{}_{}_{}.png'.format(varname, sp[1], sp[2], sp[-1].split('.')[0])

    savefilepath = os.path.join(save_dir, sf)

    return savefilepath


def set_map_panel(data):
    """
    Set up the map and projection for multiple panels
    :param data: data from the netcdf file to be plotted, including latitude and longitude coordinates
    :returns fig, ax objects
    :returns dlat: latitude data
    returns dlon: longitude data
    """
    lccproj = ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8)
    fig, axs = plt.subplots(2, 2, figsize=(9, 10), subplot_kw=dict(projection=lccproj), sharex=True, sharey=True)

    dlat = data['XLAT'].values
    dlon = data['XLONG'].values

    return fig, axs, dlat, dlon


def subset_grid(data, model):
    """
    Subset the data according to defined latitudes and longitudes, and define the axis limits for the plots
    :param data: data from the netcdf file to be plotted, including latitude and longitude coordinates
    :param model: the model version that is being plotted, e.g. 3km, 9km, or bight (to plot just NY Bight region)
    :returns data: data subset to the desired grid region
    :returns axis_limits: axis limits to be used in the plotting function
    """
    if model == '3km':
        axis_limits = [-79.81, -69.18, 34.52, 43]
        model_lims = dict(minlon=-79.9, maxlon=-69, minlat=34.5, maxlat=43)
    elif model == '9km':
        axis_limits = [-80, -67.9, 33, 44]
        model_lims = dict(minlon=-80.05, maxlon=-67.9, minlat=33, maxlat=44.05)
    elif model == 'bight':
        axis_limits = [-77.5, -72, 37.5, 42.05]
        model_lims = dict(minlon=-77.55, maxlon=-71.95, minlat=37.45, maxlat=42.05)
    else:
        print('Model not recognized')

    mlon = data['XLONG']
    mlat = data['XLAT']
    lon_ind = np.logical_and(mlon > model_lims['minlon'], mlon < model_lims['maxlon'])
    lat_ind = np.logical_and(mlat > model_lims['minlat'], mlat < model_lims['maxlat'])

    # find i and j indices of lon/lat in boundaries
    ind = np.where(np.logical_and(lon_ind, lat_ind))

    # subset data from min i,j corner to max i,j corner
    # there will be some points outside of defined boundaries because grid is not rectangular
    data = np.squeeze(data)[range(np.min(ind[0]), np.max(ind[0]) + 1), range(np.min(ind[1]), np.max(ind[1]) + 1)]

    return data, axis_limits


def main(args):
    start_time = time.time()
    wrf_procdir = args.wrf_dir
    sDir = '/home/coolgroup/ru-wrf/real-time/v4.1_parallel/scripts/webfigs'

    if wrf_procdir.endswith('/'):
        ext = '*.nc'
    else:
        ext = '/*.nc'
    files = sorted(glob.glob(wrf_procdir + ext))

    # get the model version (3km or 9km) from the directory path
    f0 = files[0]
    model_ver = f0.split('/')[-3]  # 3km or 9km
    save_dir = os.path.join(sDir, model_ver)
    os.makedirs(save_dir, exist_ok=True)

    # List of variables for plotting
    plt_vars = ['cloud']

    for i, f in enumerate(files):
        fname = f.split('/')[-1].split('.')[0]
        print('\nPlotting {}'.format(fname))
        splitter = fname.split('/')[-1].split('_')
        ncfile = xr.open_dataset(f, mask_and_scale=False)
        for pv in plt_vars:
            sfile = save_filepath(save_dir, pv, splitter)
            if pv == 'cloud':
                plt_cloudfrac(ncfile, model_ver, sfile)

    print('')
    print('Script run time: {} minutes'.format(round(((time.time() - start_time) / 60), 2)))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-wrf_dir',
                            dest='wrf_dir',
                            default='/home/coolgroup/ru-wrf/real-time/v4.1_parallel/processed/modlevs/3km/20200601',
                            type=str,
                            help='Full directory path to subset WRF native model level netCDF files.')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))(main(parsed_args))
