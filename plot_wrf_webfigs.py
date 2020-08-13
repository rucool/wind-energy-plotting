#!/usr/bin/env python

"""
Author: Lori Garzio on 5/28/2020
Last modified: 8/13/2020
Creates hourly plots of RU-WRF output variables: air temperature at 2m, wind speed at 10m and 150m, hourly rainfall +
sea level pressure, and daily accumulated rainfall. The plots are used to populate RUCOOL's RU-WRF webpage:
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def add_map_features(ax, axes_limits):
    """
    Adds latitude and longitude gridlines and labels, coastlines, and statelines to a cartopy map object
    :param ax: plotting axis object
    :param axes_limits: list of axis limits [min lon, max lon, min lat, max lat]
    """
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='dotted', x_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 13}
    gl.ylabel_style = {'size': 13}

    # add some space between the grid labels and bottom of plot
    gl.xpadding = 12
    gl.ypadding = 12

    # set axis limits
    ax.set_extent(axes_limits)

    land = cfeature.NaturalEarthFeature('physical', 'land', '10m')
    ax.add_feature(land, zorder=5, edgecolor='black', facecolor='none')

    state_lines = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none')

    ax.add_feature(cfeature.BORDERS, zorder=6)
    ax.add_feature(state_lines, zorder=7, edgecolor='black')


def add_contours(ax, londata, latdata, vardata, clist):
    """
    Adds black contour lines with labels to a cartopy map object
    :param ax: plotting axis object
    :param londata: longitude data
    :param latdata: latitude data
    :param vardata: variable data
    :param clist: list of contour levels
    """
    CS = ax.contour(londata, latdata, vardata, clist, colors='black', linewidths=.5, transform=ccrs.PlateCarree())
    ax.clabel(CS, inline=True, fontsize=10.5, fmt='%d')


def add_text(ax, run_date, time_coverage_start, model):
    """
    Adds text regarding model run information to the bottom of a plot
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
    valid_dt_edt_str = '{} {} {:02d}:00EDT'.format(wkday, dt.datetime.strftime(valid_dt_edt, '%d%b%Y'),
                                                   valid_dt_edt.hour)

    insert_text1 = 'RU-WRF (v4.1) {} Model: Initialized {}'.format(model, init_dt_str)
    ax.text(.41, -.09, insert_text1, size=10, transform=ax.transAxes)

    insert_text2 = 'Valid {} ({}) | Forecast Hr {}'.format(valid_dt_gmt_str, valid_dt_edt_str, fcast_hour)
    ax.text(.275, -.13, insert_text2, size=10, transform=ax.transAxes)


def plot_contourf(fig, ax, ttl, lon_data, lat_data, var_data, clevs, cmap, norm, clab):
    """
    Create a filled contour plot with user-defined levels and colors
    :param fig: figure object
    :param ax: plotting axis object
    :param ttl: plot title
    :param lon_data: longitude data
    :param lat_data: latitude data
    :param var_data: variable data
    :param clevs: list of colorbar level demarcations
    :param cmap: colormap
    :param norm: object that normalizes the colorbar level demarcations
    :param clab: colorbar label
    :returns fig, ax objects
    """
    plt.subplots_adjust(right=0.88)
    plt.title(ttl, fontsize=17)
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size='5%', pad=0.1, axes_class=plt.Axes)
    fig.add_axes(cax)

    cs = ax.contourf(lon_data, lat_data, var_data, clevs, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), alpha=.9)
    cb = plt.colorbar(cs, cax=cax)
    cb.set_label(label=clab, fontsize=14)

    return fig, ax


def plot_pcolor(fig, ax, ttl, lon_data, lat_data, var_data, var_min, var_max, cmap, clab):
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
    :returns fig, ax objects
    """
    plt.subplots_adjust(right=0.88)
    plt.title(ttl, fontsize=17)
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size='5%', pad=0.1, axes_class=plt.Axes)
    fig.add_axes(cax)

    h = ax.pcolor(lon_data, lat_data, var_data, vmin=var_min, vmax=var_max, cmap=cmap, transform=ccrs.PlateCarree())

    cb = plt.colorbar(h, cax=cax)
    cb.set_label(label=clab, fontsize=14)

    return fig, ax


def plt_2m_temp(nc, model, figname):
    """
    Create a pcolor surface map of air temperature at 2m with contours
    :param nc: netcdf file
    :param model: the model version that is being plotted, e.g. 3km or 9km
    :param figname: full file path to save directory and save filename
    """
    t2 = nc['T2']
    color_label = 'Air Temperature ($^\circ$F)'

    plot_types = ['full_grid', 'bight']  # plot the full grid and just NY Bight area
    for pt in plot_types:
        if pt == 'full_grid':  # subset the entire grid
            t2_sub, ax_lims = subset_grid(t2, model)
        else:  # subset just NY Bight
            new_fname = 'bight_{}'.format(figname.split('/')[-1])
            figname = '/{}/{}'.format(os.path.join(*figname.split('/')[0:-1]), new_fname)
            t2_sub, ax_lims = subset_grid(t2, 'bight')

        fig, ax, lat, lon = set_map(t2_sub)
        d = np.squeeze(t2_sub.values) * 9/5 - 459.67  # convert K to F

        fig, ax = plot_pcolor(fig, ax, '2m {}'.format(color_label), lon, lat, d, -20, 110, 'jet', color_label)

        # add contour lines
        contour_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        add_contours(ax, lon, lat, d, contour_list)

        # add text to the bottom of the plot
        add_text(ax, nc.SIMULATION_START_DATE, nc.time_coverage_start, model)

        add_map_features(ax, ax_lims)

        plt.savefig(figname, dpi=200)
        plt.close()


def plt_rain(nc, model, figname, raintype, ncprev=None):
    """
    Create filled contour surface maps of hourly and accumulated rain
    :param nc: netcdf file
    :param model: the model version that is being plotted, e.g. 3km or 9km
    :param figname: full file path to save directory and save filename
    :param raintype: plot type to make, e.g. 'acc' (accumulated) or 'hourly'
    :param ncprev: optional, netcdf file from the previous model hour to calculate hourly rainfall
    """
    # RAINNC = total accumulated rainfall
    rn = nc['RAINNC']
    plot_types = ['full_grid', 'bight']

    if raintype == 'acc':
        new_fname = 'acc{}'.format(figname.split('/')[-1])
        figname = '/{}/{}'.format(os.path.join(*figname.split('/')[0:-1]), new_fname)
        color_label = 'Total Accumulated Precipitation (in)'
        title = color_label
        slp = None  # don't plot sea level pressure for accumulated rain maps
    elif raintype == 'hourly':
        color_label = 'Hourly Precipitation (in)'
        title = '{}, Sea Level Pressure (mb)'.format(color_label)
        slp = nc['SLP']
        # calculate hourly rainfall for every model hour by subtracting the rainfall for the previous hour from
        # the rainfall for the current hour
        if ncprev is not None:
            preciprev = ncprev['RAINNC']
            rn = np.subtract(np.squeeze(rn), np.squeeze(preciprev))
        else:
            rn = rn

    for pt in plot_types:
        if pt == 'full_grid':  # subset the entire grid
            if slp is not None:
                slp_sub, __ = subset_grid(slp, model)
            rn_sub, ax_lims = subset_grid(rn, model)
        else:  # subset just NY Bight
            new_fname = 'bight_{}'.format(figname.split('/')[-1])
            figname = '/{}/{}'.format(os.path.join(*figname.split('/')[0:-1]), new_fname)
            if slp is not None:
                slp_sub, __ = subset_grid(slp, 'bight')
            rn_sub, ax_lims = subset_grid(rn, 'bight')

        fig, ax, lat, lon = set_map(rn_sub)

        # convert mm to inches
        rn_sub = rn_sub * 0.0394

        # modified NWS colormap, from http://jjhelmus.github.io/blog/2013/09/17/plotting-nsw-precipitation-data/
        nws_precip_colors = [
            "#fdfdfd",  # 0.01 - 0.10 inches
            "#019ff4",  # 0.10 - 0.25 inches
            "#0300f4",  # 0.25 - 0.50 inches
            "#02fd02",  # 0.50 - 0.75 inches
            "#01c501",  # 0.75 - 1.00 inches
            "#008e00",  # 1.00 - 1.50 inches
            "#fdf802",  # 1.50 - 2.00 inches
            "#e5bc00",  # 2.00 - 2.50 inches
            "#fd9500",  # 2.50 - 3.00 inches
            "#fd0000",  # 3.00 - 4.00 inches
            "#d40000",  # 4.00 - 5.00 inches
            "#bc0000",  # 5.00 - 6.00 inches
            "#f800fd",  # 6.00 - 8.00 inches
            "#9854c6",  # 8.00 - 10.00 inches
            "#4B0082"  # 10.00+
        ]
        precip_colormap = mpl.colors.ListedColormap(nws_precip_colors)

        # specify colorbar level demarcations
        levels = [0.01, 0.1, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10., 12.]
        norm = mpl.colors.BoundaryNorm(levels, 15)

        # plot data
        plot_contourf(fig, ax, title, lon, lat, rn_sub, levels, precip_colormap, norm, color_label)

        # add slp as contours if provided
        if slp is not None:
            contour_list = [940, 944, 948, 952, 956, 960, 964, 968, 972, 976, 980, 984, 988, 992, 996, 1000, 1004, 1008,
                            1012, 1016, 1020, 1024, 1028, 1032, 1036, 1040]
            add_contours(ax, lon, lat, slp_sub, contour_list)

        # add text to the bottom of the plot
        add_text(ax, nc.SIMULATION_START_DATE, nc.time_coverage_start, model)

        add_map_features(ax, ax_lims)

        plt.savefig(figname, dpi=200)
        plt.close()


def plt_windsp(nc, model, ht, figname):
    """
    Create pseudocolor surface maps of wind speed with quivers indicating wind direction.
    :param nc: netcdf file
    :param model: the model version that is being plotted, e.g. 3km or 9km
    :param ht: wind speed height to plot, e.g. 10m, 150m
    :param figname: full file path to save directory and save filename
    """
    if ht == '10m':
        u = nc['U10']
        v = nc['V10']
    elif ht == '150m':
        u = nc.sel(height=150)['U']
        v = nc.sel(height=150)['V']

    color_label = 'Wind Speed (knots)'

    # define the subsetting for the quivers on the map based on model and height
    quiver_subset = dict(_3km=dict(_10m=11, _150m=13),
                         _9km=dict(_10m=4, _150m=6),
                         bight_3km=dict(_10m=6, _150m=7),
                         bight_9km=dict(_10m=2, _150m=3))

    plot_types = ['full_grid', 'bight']
    for pt in plot_types:
        if pt == 'full_grid':  # subset the entire grid
            u_sub, __ = subset_grid(u, model)
            v_sub, ax_lims = subset_grid(v, model)
            qs = quiver_subset['_{}'.format(model)]['_{}'.format(ht)]
        else:  # subset just NY Bight
            new_fname = 'bight_{}'.format(figname.split('/')[-1])
            figname = '/{}/{}'.format(os.path.join(*figname.split('/')[0:-1]), new_fname)
            u_sub, __ = subset_grid(u, 'bight')
            v_sub, ax_lims = subset_grid(v, 'bight')
            qs = quiver_subset['bight_{}'.format(model)]['_{}'.format(ht)]

        fig, ax, lat, lon = set_map(u_sub)
        u_sub = np.squeeze(u_sub.values) * 1.94384  # convert wind speeds from m/s to knots
        v_sub = np.squeeze(v_sub.values) * 1.94384

        # calculate wind speed from u and v
        speed = wind_uv_to_spd(u_sub, v_sub)

        fig, ax = plot_pcolor(fig, ax, '{} {}'.format(ht, color_label), lon, lat, speed, 0, 40, 'BuPu', color_label)

        # subset the quivers and add as a layer
        ax.quiver(lon[::qs, ::qs], lat[::qs, ::qs], u_sub[::qs, ::qs], v_sub[::qs, ::qs], scale=1000,
                  width=.002, headlength=4, transform=ccrs.PlateCarree())

        # add contours
        contour_list = [25, 34, 48]
        add_contours(ax, lon, lat, speed, contour_list)

        # add text to the bottom of the plot
        add_text(ax, nc.SIMULATION_START_DATE, nc.time_coverage_start, model)

        add_map_features(ax, ax_lims)

        plt.savefig(figname, dpi=200)
        plt.close()


def save_filepath(save_dir, varname, sp):
    """
    Create a full file path to save directory and save filename.
    :param save_dir: directory to which file is saved
    :param varname: variable name
    :param sp: list containing original netcdf file name split into components, e.g. ['wrfproc', '3km', '20200720', '00Z', 'H000']
    :returns savefilepath: full file path to save directory and save filename
    """
    if varname == 'U10':
        varname = 'ws10'
    elif varname == 'U150':
        varname = 'ws150'

    sf = '{}_{}_{}_{}.png'.format(varname, sp[1], sp[2], sp[-1].split('.')[0])

    savefilepath = os.path.join(save_dir, sf)

    return savefilepath


def set_map(data):
    """
    Set up the map and projection
    :param data: data from the netcdf file to be plotted, including latitude and longitude coordinates
    :returns fig, ax objects
    :returns dlat: latitude data values
    returns dlon: longitude data values
    """
    lccproj = ccrs.LambertConformal(central_longitude=-74.5, central_latitude=38.8)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=lccproj))

    dlat = data['XLAT'].values
    dlon = data['XLONG'].values

    return fig, ax, dlat, dlon


def subset_grid(data, model):
    """
    Subset the data according to defined latitudes and longitudes, and define the axis limits for the plots
    :param data: data from the netcdf file to be plotted, including latitude and longitude coordinates
    :param model: the model version that is being plotted, e.g. 3km, 9km, or bight (to plot just NY Bight region)
    :returns data: data subset to the desired grid region
    :returns axis_limits: axis limits to be used in the plotting function
    """
    if model == '3km':
        axis_limits = [-79.81, -69.18, 34.5, 43]
        model_lims = dict(minlon=-79.9, maxlon=-69, minlat=34.5, maxlat=43)
    elif model == '9km':
        axis_limits = [-80, -67.9, 33.05, 44]
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
    # sDir = '/home/coolgroup/ru-wrf/real-time/v4.1_parallel/scripts/webfigs'
    sDir = '/home/lgarzio/rucool/bpu/wrf/webfigs'

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

    # List of variables to plot
    plt_vars = ['T2', 'U10', 'U150', 'rain']

    for i, f in enumerate(files):
        fname = f.split('/')[-1].split('.')[0]
        print('\nPlotting {}'.format(fname))
        splitter = fname.split('/')[-1].split('_')
        ncfile = xr.open_dataset(f, mask_and_scale=False)
        for pv in plt_vars:
            sfile = save_filepath(save_dir, pv, splitter)
            if pv == 'T2':
                plt_2m_temp(ncfile, model_ver, sfile)
            elif pv == 'U10':
                plt_windsp(ncfile, model_ver, '10m', sfile)
            elif pv == 'U150':
                plt_windsp(ncfile, model_ver, '150m', sfile)
            elif pv == 'rain':
                # plot hourly and accumulated rainfall (each .nc file contains accumulated precipitation)
                # plot accumulated rainfall
                plt_rain(ncfile, model_ver, sfile, 'acc')

                # plot hourly rainfall
                if i > 0:
                    fprev = files[i - 1]
                    nc_prev = xr.open_dataset(fprev, mask_and_scale=False)
                else:
                    nc_prev = None
                plt_rain(ncfile, model_ver, sfile, 'hourly', nc_prev)

    print('')
    print('Script run time: {} minutes'.format(round((time.time() - start_time) / 60)))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-wrf_dir',
                            dest='wrf_dir',
                            default='/home/coolgroup/ru-wrf/real-time/v4.1_parallel/processed/3km/20200101',
                            type=str,
                            help='Full directory path to subset WRF netCDF files.')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
