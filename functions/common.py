#! /usr/bin/env python

"""
Author: Lori Garzio on 8/17/2020
Last modified: 9/8/2020
"""

import numpy as np
import os
import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xml.etree.ElementTree as ET  # for parsing kml files


def add_map_features(ax, axes_limits, landcolor=None, ecolor=None):
    """
    Adds latitude and longitude gridlines and labels, coastlines, and statelines to a cartopy map object
    :param ax: plotting axis object
    :param axes_limits: list of axis limits [min lon, max lon, min lat, max lat]
    :param landcolor: optional, specify land color
    :param ecolor: optional, specify edge color, default is black
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

    if landcolor is not None:
        lc = landcolor
    else:
        lc = 'none'

    if ecolor is not None:
        ec = ecolor
    else:
        ec = 'black'

    ax.add_feature(land, zorder=5, edgecolor=ec, facecolor=lc)

    state_lines = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none')

    ax.add_feature(cfeature.BORDERS, zorder=6, edgecolor=ec)
    ax.add_feature(state_lines, zorder=7, edgecolor=ec)


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


def extract_lease_areas():
    """
    Extracts polygon coordinates from a .kml file.
    :returns dictionary containing lat/lon coordinates for wind energy lease area polygons, separated by company
    """
    boem_lease_areas = '/home/coolgroup/bpu/mapdata/shapefiles/RU-WRF_Plotting_Shapefiles/boem_lease_areas_AS_OW_split.kml'
    nmsp = '{http://www.opengis.net/kml/2.2}'
    doc = ET.parse(boem_lease_areas)
    findouter = '{0}MultiGeometry/{0}Polygon/{0}outerBoundaryIs/{0}LinearRing/{0}coordinates'.format(nmsp)

    polygon_dict = dict()
    for pm in doc.iterfind('.//{0}Placemark'.format(nmsp)):
        for nm in pm.iterfind('{0}name'.format(nmsp)):  # find the company name
            polygon_dict[nm.text] = dict(outer=[], inner=[])
            for ls in pm.iterfind(findouter):
                coord_strlst = [x for x in ls.text.split(' ')]
                for coords in coord_strlst:
                    splitter = coords.split(',')
                    polygon_dict[nm.text]['outer'].append([np.float(splitter[0]), np.float(splitter[1])])

    return polygon_dict


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
