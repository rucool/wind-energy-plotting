#! /usr/bin/env python

"""
Author: Lori Garzio on 8/17/2020
Last modified: 9/14/2020
"""

import numpy as np
import os
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xml.etree.ElementTree as ET  # for parsing kml files


def add_map_features(ax, axes_limits, landcolor=None, ecolor=None, xticks=None, yticks=None):
    """
    Adds latitude and longitude gridlines and labels, coastlines, and statelines to a cartopy map object
    :param ax: plotting axis object
    :param axes_limits: list of axis limits [min lon, max lon, min lat, max lat]
    :param landcolor: optional, specify land color
    :param ecolor: optional, specify edge color, default is black
    :param xticks: optional, specify x-ticks
    :param yticks: optional, specify y-ticks
    """
    landcolor = landcolor or 'none'
    ecolor = ecolor or 'black'
    xticks = xticks or None
    yticks = yticks or None

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

    # set optional x and y ticks
    if xticks:
        gl.xlocator = mticker.FixedLocator(xticks)

    if yticks:
        gl.ylocator = mticker.FixedLocator(yticks)

    land = cfeature.NaturalEarthFeature('physical', 'land', '10m')

    ax.add_feature(land, zorder=5, edgecolor=ecolor, facecolor=landcolor)

    state_lines = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none')

    ax.add_feature(cfeature.BORDERS, zorder=6, edgecolor=ecolor)
    ax.add_feature(state_lines, zorder=7, edgecolor=ecolor)


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


def define_axis_limits(model):
    if model == '3km':
        axis_limits = [-79.81, -69.18, 34.5, 43]
        data_sub = dict(minlon=-79.9, maxlon=-69, minlat=34.5, maxlat=43)
        xticks = None
        yticks = None
    elif model == '9km':
        axis_limits = [-80, -67.9, 33.05, 44]
        data_sub = dict(minlon=-80.05, maxlon=-67.9, minlat=33, maxlat=44.05)
        xticks = None
        yticks = None
    elif model == 'bight':
        axis_limits = [-77.5, -72, 37.5, 42.05]
        data_sub = dict(minlon=-77.75, maxlon=-71.75, minlat=37.25, maxlat=42.25)
        xticks = None
        yticks = None
    elif model == 'full_grid':
        axis_limits = [-79.79, -69.2, 34.5, 43]
        data_sub = dict(minlon=-80, maxlon=-69, minlat=34.25, maxlat=43.25)
        xticks = [-78, -76, -74, -72, -70]
        yticks = [36, 38, 40, 42]
    elif model == 'mab':
        axis_limits = [-77.2, -69.6, 36, 41.8]
        data_sub = dict(minlon=-77.5, maxlon=-69.25, minlat=35.5, maxlat=42)
        xticks = [-75, -73, -71]
        yticks = [37, 39, 41]
    elif model == 'nj':
        axis_limits = [-75.7, -71.5, 38.1, 41.2]
        data_sub = dict(minlon=-75.9, maxlon=-71.25, minlat=37.9, maxlat=41.5)
        xticks = [-75, -74, -73, -72]
        yticks = [39, 40, 41]
    elif model == 'snj':
        axis_limits = [-75.6, -73, 38.6, 40.5]
        data_sub = dict(minlon=-75.8, maxlon=-73.25, minlat=38.25, maxlat=40.25)
        xticks = [-75, -74.5, -74, -73.5]
        yticks = [39, 39.5, 40]
    else:
        print('Model not recognized')

    return axis_limits, data_sub, xticks, yticks


def extract_lease_areas():
    """
    Extracts polygon coordinates from a .kml file.
    :returns dictionary containing lat/lon coordinates for wind energy lease area polygons, separated by company
    """
    #boem_lease_areas = '/Users/lgarzio/Documents/rucool/bpu/wrf/boem_lease_area_full.kml'  # on local machine
    #boem_lease_areas = '/Users/lgarzio/Documents/rucool/bpu/wrf/boem_lease_areas_AS_OW_split.kml'  # on local machine
    boem_lease_areas = '/home/coolgroup/bpu/mapdata/shapefiles/RU-WRF_Plotting_Shapefiles/boem_lease_areas_AS_OW_split.kml'
    nmsp = '{http://www.opengis.net/kml/2.2}'
    doc = ET.parse(boem_lease_areas)
    findouter = './/{0}outerBoundaryIs/{0}LinearRing/{0}coordinates'.format(nmsp)
    findinner = './/{0}innerBoundaryIs/{0}LinearRing/{0}coordinates'.format(nmsp)

    polygon_dict = dict()
    for pm in doc.iterfind('.//{0}Placemark'.format(nmsp)):
        for nm in pm.iterfind('{0}name'.format(nmsp)):  # find the company name
            polygon_dict[nm.text] = dict(outer=[], inner=[])
            polygon_dict[nm.text]['outer'] = find_coords(pm, findouter)
            polygon_dict[nm.text]['inner'] = find_coords(pm, findinner)

    return polygon_dict


def find_coords(elem, findstr):
    """
    Finds coordinates in an .xml file and appends them in pairs to a list
    :param elem: element of an .xml file
    :param findstr: string to find in the element
    :returns list of coordinates
    """
    coordlist = []
    for ls in elem.iterfind(findstr):
        coord_strlst = [x for x in ls.text.split(' ')]
        for coords in coord_strlst:
            splitter = coords.split(',')
            coordlist.append([np.float(splitter[0]), np.float(splitter[1])])

    return coordlist


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

    try:
        dlat = data['XLAT'].values
        dlon = data['XLONG'].values
    except KeyError:
        dlat = data['lat'].values
        dlon = data['lon'].values

    return fig, ax, dlat, dlon


def subset_grid(data, model):
    """
    Subset the data according to defined latitudes and longitudes, and define the axis limits for the plots
    :param data: data from the netcdf file to be plotted, including latitude and longitude coordinates
    :param model: the model version that is being plotted, e.g. 3km, 9km, or bight (to plot just NY Bight region)
    :returns data: data subset to the desired grid region
    :returns axis_limits: axis limits to be used in the plotting function
    """
    axis_limits, model_lims, _, _ = define_axis_limits(model)

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


def subset_grid_wct(data, model):
    """
    Subset the data according to defined latitudes and longitudes, and define the axis limits for the plots
    :param data: data from the netcdf file to be plotted, including latitude and longitude coordinates
    :param model: the model version that is being plotted, e.g. 3km, 9km, or bight (to plot just NY Bight region)
    :returns data: data subset to the desired grid region
    :returns axis_limits: axis limits to be used in the plotting function
    """
    axis_limits, model_lims, xticks, yticks = define_axis_limits(model)

    mlon = data['lon']
    mlat = data['lat']
    mlon, mlat = np.meshgrid(mlon, mlat)
    lon_ind = np.logical_and(mlon > model_lims['minlon'], mlon < model_lims['maxlon'])
    lat_ind = np.logical_and(mlat > model_lims['minlat'], mlat < model_lims['maxlat'])

    # find i and j indices of lon/lat in boundaries
    ind = np.where(np.logical_and(lon_ind, lat_ind))

    # subset data from min i,j corner to max i,j corner
    # there will be some points outside of defined boundaries because grid is not rectangular
    data = np.squeeze(data)[range(np.min(ind[0]), np.max(ind[0]) + 1), range(np.min(ind[1]), np.max(ind[1]) + 1)]

    return data, axis_limits, xticks, yticks


def wind_uv_to_dir(u, v):
    """
    Calculates the wind direction from the u and v component of wind.
    Takes into account the wind direction coordinates is different than the
    trig unit circle coordinate. If the wind direction is 360 then returns zero
    (by %360)
    Inputs:
    u = west/east direction (wind from the west is positive, from the east is negative)
    v = south/noth direction (wind from the south is positive, from the north is negative)
    """
    WDIR = (270-np.rad2deg(np.arctan2(v, u))) % 360
    return WDIR
