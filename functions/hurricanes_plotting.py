import warnings
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from cartopy.io.shapereader import Reader

# Suppresing warnings for a "pretty output."
warnings.simplefilter("ignore")

try:
    from urllib.request import urlopen, urlretrieve
except Exception:
    from urllib import urlopen, urlretrieve


proj = dict(
    map=ccrs.Mercator(),  # the projection that you want the map to be in
    data=ccrs.PlateCarree()  # the projection that the data is.
    )


def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
    """
    From ImportanceOfBeingErnest
    https://stackoverflow.com/a/47232942/2643708
    :param nc: number of categories (colors)
    :param nsc: number of subcategories (shades for each color)
    :param cmap: matplotlib colormap
    :param continuous:
    :return:
    """
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0,1,nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv,nsc).reshape(nsc,3)
        arhsv[:,1] = np.linspace(chsv[1],0.25,nsc)
        arhsv[:,2] = np.linspace(chsv[2],1,nsc)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i*nsc:(i+1)*nsc,:] = rgb
    cmap = matplotlib.colors.ListedColormap(cols)
    return cmap


# decimal degrees to degree-minute-second converter
def dd2dms(vals):
    n = np.empty(np.shape(vals))
    n[:] = False
    n[vals < 0] = True
    vals[n == True] = -vals[n == True]
    d = np.floor(vals)
    rem = vals - d
    rem = rem * 60
    m = np.floor(rem)
    rem -= m
    s = np.round(rem * 60)
    d[n == True] = -d[n == True]
    return d, m, s


def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s}"


# function to define major and minor tick locations and major tick labels
def get_ticks(bounds, dirs, otherbounds):
    dirs = dirs.lower()
    l0 = np.float(bounds[0])
    l1 = np.float(bounds[1])
    r = np.max([l1 - l0, np.float(otherbounds[1]) - np.float(otherbounds[0])])
    if r <= 1.5:
        # <1.5 degrees: 15' major ticks, 5' minor ticks
        minor_int = 1.0 / 12.0
        major_int = 1.0 / 4.0
    elif r <= 3.0:
        # <3 degrees: 30' major ticks, 10' minor ticks
        minor_int = 1.0 / 6.0
        major_int = 0.5
    elif r <= 7.0:
        # <7 degrees: 1d major ticks, 15' minor ticks
        minor_int = 0.25
        major_int = np.float(1)
    elif r <= 15:
        # <15 degrees: 2d major ticks, 30' minor ticks
        minor_int = 0.5
        major_int = np.float(2)
    elif r <= 30:
        # <30 degrees: 3d major ticks, 1d minor ticks
        minor_int = np.float(1)
        major_int = np.float(3)
    else:
        # >=30 degrees: 5d major ticks, 1d minor ticks
        minor_int = np.float(1)
        major_int = np.float(5)

    minor_ticks = np.arange(np.ceil(l0 / minor_int) * minor_int, np.ceil(l1 / minor_int) * minor_int + minor_int,
                            minor_int)
    minor_ticks = minor_ticks[minor_ticks <= l1]
    major_ticks = np.arange(np.ceil(l0 / major_int) * major_int, np.ceil(l1 / major_int) * major_int + major_int,
                            major_int)
    major_ticks = major_ticks[major_ticks <= l1]

    if major_int < 1:
        d, m, s = dd2dms(np.array(major_ticks))
        if dirs == 'we' or dirs == 'ew' or dirs == 'lon' or dirs == 'long' or dirs == 'longitude':
            n = 'W' * sum(d < 0)
            p = 'E' * sum(d >= 0)
            dir = n + p
            major_tick_labels = [str(np.abs(int(d[i]))) + u"\N{DEGREE SIGN}" + str(int(m[i])) + "'" + dir[i] for i in
                                 range(len(d))]
        elif dirs == 'sn' or dirs == 'ns' or dirs == 'lat' or dirs == 'latitude':
            n = 'S' * sum(d < 0)
            p = 'N' * sum(d >= 0)
            dir = n + p
            major_tick_labels = [str(np.abs(int(d[i]))) + u"\N{DEGREE SIGN}" + str(int(m[i])) + "'" + dir[i] for i in
                                 range(len(d))]
        else:
            major_tick_labels = [str(int(d[i])) + u"\N{DEGREE SIGN}" + str(int(m[i])) + "'" for i in range(len(d))]
    else:
        d = major_ticks
        if dirs == 'we' or dirs == 'ew' or dirs == 'lon' or dirs == 'long' or dirs == 'longitude':
            n = 'W' * sum(d < 0)
            p = 'E' * sum(d >= 0)
            dir = n + p
            major_tick_labels = [str(np.abs(int(d[i]))) + u"\N{DEGREE SIGN}" + dir[i] for i in range(len(d))]
        elif dirs == 'sn' or dirs == 'ns' or dirs == 'lat' or dirs == 'latitude':
            n = 'S' * sum(d < 0)
            p = 'N' * sum(d >= 0)
            dir = n + p
            major_tick_labels = [str(np.abs(int(d[i]))) + u"\N{DEGREE SIGN}" + dir[i] for i in range(len(d))]
        else:
            major_tick_labels = [str(int(d[i])) + u"\N{DEGREE SIGN}" for i in range(len(d))]

    return minor_ticks, major_ticks, major_tick_labels


def map_add_bathymetry(ax, lon, lat, elevation, levels=(-1000), zorder=5,
                       transform=proj['data']):
    # lon = ds.variables['longitude'][:]
    # lat = ds.variables['latitude'][:]
    # elevation = ds.variables['elevation'][:]
    lons, lats = np.meshgrid(lon, lat)
    h = ax.contour(lons, lats, elevation, levels,
                   inewidths=.75, alpha=.5, colors='k',
                   transform=transform,
                   zorder=zorder)
    ax.clabel(h, levels, inline=True, fontsize=6, fmt=fmt)
    return ax


def map_add_eez(ax, zorder=1):
    eez = 'data/eez/eez_boundaries_v11.shp'
    shape_feature = cfeature.ShapelyFeature(
        Reader(eez).geometries(),
        proj['data'],
        edgecolor='black',
        facecolor='none'
        )
    ax.add_feature(shape_feature, zorder=zorder)


def map_add_boem_outlines(ax, shpfile, edgecolor=None, zorder=None, alpha=None):
    edgecolor = edgecolor or 'black'
    zorder = zorder or 1
    alpha = alpha or 1

    shape_feature = cfeature.ShapelyFeature(
        Reader(shpfile).geometries(),
        proj['data'],
        edgecolor=edgecolor,
        facecolor='none'
        )
    ax.add_feature(shape_feature, zorder=zorder, alpha=alpha)


def map_add_features(ax, extent, edgecolor="black", landcolor="tan", zorder=0, zoom_coastline=True, add_ocean_color=True):
    """_summary_
    Args:
        ax (_type_): _description_
        extent (_type_): _description_
        edgecolor (str, optional): _description_. Defaults to "black".
        landcolor (str, optional): _description_. Defaults to "tan".
        zorder (int, optional): _description_. Defaults to 0.
    """

    state_lines = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none'
    )

    if zoom_coastline:
        LAND = cfeature.GSHHSFeature(scale='full')
    else:
        LAND = cfeature.NaturalEarthFeature('physical', 'land', '10m')

    # Axes properties and features
    ax.set_extent(extent)

    if add_ocean_color:
        ax.set_facecolor(cfeature.COLORS['water'])

    ax.add_feature(LAND,
                   edgecolor=edgecolor,
                   facecolor=landcolor,
                   zorder=zorder+10)
    ax.add_feature(cfeature.RIVERS, zorder=zorder+10.2)
    ax.add_feature(cfeature.LAKES, zorder=zorder+10.2, alpha=0.5)
    ax.add_feature(state_lines, edgecolor=edgecolor, zorder=zorder+10.3)
    ax.add_feature(cfeature.BORDERS, linestyle='--', zorder=zorder+10.3)


def map_add_legend(ax):
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def map_add_ticks(ax, extent, fontsize=13, transform=proj['data'], left_label=True, bottom_label=True):
    xl = [extent[0], extent[1]]
    yl = [extent[2], extent[3]]

    tick0x, tick1, ticklab = get_ticks(xl, 'we', yl)
    ax.set_xticks(tick0x, minor=True, crs=transform)
    ax.set_xticks(tick1, crs=transform)
    ax.set_xticklabels(ticklab, fontsize=fontsize)

    # get and add latitude ticks/labels
    tick0y, tick1, ticklab = get_ticks(yl, 'sn', xl)
    ax.set_yticks(tick0y, minor=True, crs=transform)
    ax.set_yticks(tick1, crs=transform)
    ax.set_yticklabels(ticklab, fontsize=fontsize)

    # gl = ax.gridlines(draw_labels=False, linewidth=.5, color='gray', alpha=0.75, linestyle='--', crs=ccrs.PlateCarree())
    # gl.xlocator = mticker.FixedLocator(tick0x)
    # gl.ylocator = mticker.FixedLocator(tick0y)

    ax.tick_params(which='major',
                   direction='out',
                   bottom=True, top=True,
                   labelbottom=bottom_label, labeltop=False,
                   left=True, right=True,
                   labelleft=left_label, labelright=False,
                   length=5, width=2)

    ax.tick_params(which='minor',
                   direction='out',
                   bottom=True, top=True,
                   labelbottom=bottom_label, labeltop=False,
                   left=True, right=True,
                   labelleft=left_label, labelright=False,
                   width=1)

    # if grid:
        # ax.grid(color='k', linestyle='--', zorder=zorder)
    return ax


def map_create(extent, proj=proj['map'], labelsize=14, ticks=True, labels=False, features=True, edgecolor="black",
               landcolor="tan", ax=None, figsize=(11, 8), fig_init=False, zoom_coastline=True, add_ocean_color=False,
               left_label=True, bottom_label=True):
    """Create a cartopy map within a certain extent.
    Args:
        extent (tuple or list): Extent (x0, x1, y0, y1) of the map in the given coordinate system.
        proj (cartopy.crs class, optional): Define a projected coordinate system with flat topology and Euclidean distance.
            Defaults to ccrs.Mercator().
        features (bool, optional): Add preferred map settings.
            Defaults to True.
        ax (_type_, optional): Pass matplotlib axis to function. Not necessary if plotting to subplot.
            Defaults to None.
        figsize (tuple, optional): (width, height) of the figure. Defaults to (11, 8).
    Returns:
        _type_: _description_
    """
    # If a matplotlib axis is not passed, create a new cartopy/mpl figure
    if not ax:
        fig_init = True
        fig, ax = plt.subplots(
            figsize=figsize,  # 12, 9
            subplot_kw=dict(projection=proj)
        )

    # Make the map pretty
    if features:
        fargs = {
            "edgecolor": edgecolor,
            "landcolor": landcolor,
            "zoom_coastline": zoom_coastline,
            "add_ocean_color": add_ocean_color
            }
        map_add_features(ax, extent, **fargs)

    # # Add bathymetry
    # if bathy:
    #     bargs = {
    #         "isobaths": isobaths,
    #         "zorder": 1.5
    #     }
    #     map_add_bathymetry(ax, bathy, proj, **bargs)

    # Add ticks
    if ticks:
        map_add_ticks(ax, extent, left_label=left_label, bottom_label=bottom_label)

    if labels:
        # Set labels
        ax.set_xlabel('Longitude', fontsize=labelsize, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=labelsize, fontweight='bold')

    # If we generate a figure in this function, we have to return the figure
    # and axis to the calling function.
    if fig_init:
        return fig, ax
