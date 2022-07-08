#! /usr/bin/env python

"""
Author: Lori Garzio on 8/17/2020
Last modified: 9/8/2020
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable


def add_contours(ax, londata, latdata, vardata, clist, label_format=None):
    """
    Adds black contour lines with labels to a cartopy map object
    :param ax: plotting axis object
    :param londata: longitude data
    :param latdata: latitude data
    :param vardata: variable data
    :param clist: list of contour levels
    :param label_format: optional format for contour labels (e.g. '%.1f')
    """
    label_format = label_format or '%d'

    CS = ax.contour(londata, latdata, vardata, clist, colors='black', linewidths=.5, transform=ccrs.PlateCarree())
    ax.clabel(CS, inline=True, fontsize=10.5, fmt=label_format)


def add_lease_area_polygon(ax, lease_area_dict, line_color):
    """
    Adds polygon outlines for wind energy lease areas to map
    :param ax: plotting axis object
    :param lease_area_dict: dictionary containing lat/lon coordinates for wind energy lease area polygons
    :param line_color: polygon line color
    """
    for key, value in lease_area_dict.items():
        for k, v in value.items():
            if len(v) > 0:
                for i, coord in enumerate(v):
                    if i > 0:
                        poly_lons = [v[i - 1][0], coord[0]]
                        poly_lats = [v[i - 1][1], coord[1]]
                        ax.plot(poly_lons, poly_lats, ls='-', lw=.4, color=line_color, transform=ccrs.PlateCarree())


def plot_contourf(fig, ax, lon_data, lat_data, var_data, clevs, ttl=None, cmap=None, clab=None, var_lims=None,
                  normalize=None, cbar_ticks=None, extend=None):
    """
    Create a filled contour plot with user-defined levels and colors
    :param fig: figure object
    :param ax: plotting axis object
    :param lon_data: longitude data
    :param lat_data: latitude data
    :param var_data: variable data
    :param clevs: list of colorbar level demarcations
    :param ttl: optional plot title
    :param cmap: optional colormap, default = jet
    :param clab: optional colorbar label
    :param var_lims: optional, [minimum, maximum] values for plotting (for fixed colorbar)
    :param normalize: optional, object that normalizes the colorbar level demarcations
    :param cbar_ticks: optional, specify colorbar ticks
    :param extend: optional extend the colorbar, default is 'neither'
    :returns fig, ax objects
    """
    ttl = ttl or None
    cmap = cmap or 'jet'
    clab = clab or None
    var_lims = var_lims or None
    normalize = normalize or None
    cbar_ticks = cbar_ticks or None
    extend = extend or 'neither'

    plt.subplots_adjust(right=0.88)
    if ttl:
        plt.title(ttl, fontsize=17)
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size='5%', pad=0.1, axes_class=plt.Axes)
    fig.add_axes(cax)

    if normalize:
        #norm = mpl.colors.BoundaryNorm(clevs, 15)
        norm = mpl.colors.BoundaryNorm(clevs, len(clevs)-1)
        cs = ax.contourf(lon_data, lat_data, var_data, clevs, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
                         alpha=.9, extend=extend)
    else:
        cs = ax.contourf(lon_data, lat_data, var_data, clevs, vmin=var_lims[0], vmax=var_lims[1], cmap=cmap,
                         transform=ccrs.PlateCarree(), alpha=.9, extend=extend)
        #cs = ax.contourf(lon_data, lat_data, var_data, clevs, vmin=var_min, vmax=var_max, cmap=cmap, alpha=.9)

    if cbar_ticks:
        cb = plt.colorbar(cs, cax=cax, ticks=cbar_ticks)
    else:
        cb = plt.colorbar(cs, cax=cax)

    if clab:
        cb.set_label(label=clab, fontsize=14)

    return fig, ax


def plot_pcolormesh(fig, ax, lon_data, lat_data, var_data, ttl=None, cmap=None, clab=None, var_lims=None,
                    norm_clevs=None, extend=None):
    """
    Create a pseudocolor plot
    :param fig: figure object
    :param ax: plotting axis object
    :param lon_data: longitude data
    :param lat_data: latitude data
    :param var_data: variable data
    :param ttl: optional plot title
    :param cmap: optional color map, default is 'jet'
    :param clab: optional colorbar label
    :param var_lims: optional, [minimum, maximum] values for plotting (for fixed colorbar)
    :param norm_clevs: optional normalized levels
    :param extend: optional extend the colorbar, default is 'neither'
    """
    ttl = ttl or None
    cmap = cmap or 'jet'
    clab = clab or None
    var_lims = var_lims or None
    norm_clevs = norm_clevs or None
    extend = extend or 'neither'

    plt.subplots_adjust(right=0.88)
    if ttl:
        plt.title(ttl, fontsize=17)
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size='5%', pad=0.1, axes_class=plt.Axes)
    fig.add_axes(cax)


    if var_lims:
        h = ax.pcolormesh(lon_data, lat_data, var_data, vmin=var_lims[0], vmax=var_lims[1],  cmap=cmap,
                          transform=ccrs.PlateCarree())
    elif norm_clevs:
        h = ax.pcolormesh(lon_data, lat_data, var_data, cmap=cmap, norm=norm_clevs,
                          transform=ccrs.PlateCarree())
    else:
        h = ax.pcolormesh(lon_data, lat_data, var_data, cmap=cmap, transform=ccrs.PlateCarree())

    cb = plt.colorbar(h, cax=cax, extend=extend)
    if clab:
        cb.set_label(label=clab, fontsize=14)


def plot_pcolormesh_panel(fig, ax, lon_data, lat_data, var_data, panel_title=None, cmap=None, clab=None, var_lims=None,
                          norm_clevs=None, extend=None, title_pad=None):
    """
    Create a pseudocolor plot for panel plots (e.g. cloud fraction)
    :param fig: figure object
    :param ax: plotting axis object
    :param lon_data: longitude data
    :param lat_data: latitude data
    :param var_data: variable data
    :param panel_title: optional plot title
    :param cmap: optional color map, default is 'jet'
    :param clab: optional colorbar label, if None colorbar is not added
    :param var_lims: optional [minimum, maximum] values for plotting (for fixed colorbar)
    :param norm_clevs: optional normalized levels
    :param extend: optional extend the colorbar, default is 'neither'
    """
    panel_title = panel_title or None
    cmap = cmap or 'jet'
    var_lims = var_lims or None
    clab = clab or None
    norm_clevs = norm_clevs or None
    extend = extend or 'neither'
    title_pad = title_pad or 7

    if panel_title:
        ax.set_title(panel_title, fontsize=15, pad=title_pad)
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size='5%', pad=0.1, axes_class=plt.Axes)

    if var_lims:
        h = ax.pcolormesh(lon_data, lat_data, var_data, vmin=var_lims[0], vmax=var_lims[1],  cmap=cmap,
                          transform=ccrs.PlateCarree())
    elif norm_clevs:
        h = ax.pcolormesh(lon_data, lat_data, var_data, cmap=cmap, norm=norm_clevs,
                          transform=ccrs.PlateCarree())
    else:
        h = ax.pcolormesh(lon_data, lat_data, var_data, cmap=cmap, transform=ccrs.PlateCarree())

    if clab:
        fig.add_axes(cax)
        cb = plt.colorbar(h, cax=cax, extend=extend)
        cb.set_label(label=clab, fontsize=14)
        cb.ax.tick_params(labelsize=12)
