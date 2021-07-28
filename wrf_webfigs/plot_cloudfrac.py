#!/usr/bin/env python

"""
Author: Lori Garzio on 8/6/2020
Last modified: 7/28/2021
Creates 4-panel hourly plots of cloud fraction from RU-WRF 4.1 at low, medium, high levels, and Total Cloud Fraction.
The plots are used to populate RUCOOL's RU-WRF webpage:
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


def plt_cloudfrac(nc, model, figname, lease_areas=None):
    """
    Create a pcolor surface map of cloud cover
    :param nc: netcdf file
    :param model: the model version that is being plotted, e.g. 3km or 9km
    :param figname: full file path to save directory and save filename
    :param lease_areas: optional dictionary containing lat/lon coordinates for wind energy lease area polygon
    """
    lease_areas = lease_areas or None

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
            max_cloudfrac_sub, ax_lims, xticks, yticks = cf.subset_grid(max_cloudfrac, model)
        else:
            # create a new file path
            new_fname = 'bight_{}_{}'.format(splitter[0], '_'.join(splitter[1:len(splitter)]))
            save_fig = '/{}/{}'.format(os.path.join(*figname.split('/')[0:-1]), new_fname)

            # subset just NY Bight
            max_cloudfrac_sub, ax_lims, xticks, yticks = cf.subset_grid(max_cloudfrac, 'bight')

        fig, axs, lat, lon = set_map_panel(max_cloudfrac_sub)
        fig.suptitle(color_label, fontsize=17, y=.94)

        d = max_cloudfrac_sub.values * 100
        vlims = [0, 100]

        # plot Total Cloud Fraction in the bottom right panel
        ax4 = axs[1, 1]

        # add text to the bottom of the plot
        targs = dict()
        targs['add_y'] = .08
        cf.add_text(ax4, nc.SIMULATION_START_DATE, nc.time_coverage_start, model, **targs)

        kwargs = dict()
        kwargs['panel_title'] = 'Total Cloud Fraction'
        kwargs['cmap'] = 'gray'
        kwargs['var_lims'] = vlims
        kwargs['clab'] = color_label
        pf.plot_pcolormesh_panel(fig, ax4, lon, lat, d, **kwargs)

        kwargs = dict()
        kwargs['ecolor'] = 'red'
        kwargs['xticks'] = xticks
        kwargs['yticks'] = yticks
        kwargs['left_labs'] = 'remove'
        cf.add_map_features(ax4, ax_lims, **kwargs)

        if lease_areas:
            pf.add_lease_area_polygon(ax4, lease_areas, 'magenta')

        # plot each level
        ax1 = axs[0, 0]
        ax2 = axs[0, 1]
        ax3 = axs[1, 0]
        axes_ind = [ax1, ax2, ax3]
        for i, level in enumerate(var['low_mid_high'].values):
            ds = var.sel(low_mid_high=level)
            if pt == 'full_grid':
                # subset the entire grid
                ds_sub, ax_lims, xticks, yticks = cf.subset_grid(ds, model)
            else:
                # subset just NY Bight
                ds_sub, ax_lims, xticks, yticks = cf.subset_grid(ds, 'bight')

            d = ds_sub.values * 100

            # keyword arguments for plotting function
            pcargs = dict()
            pcargs['cmap'] = 'gray'
            pcargs['var_lims'] = vlims

            # keyword arguments for add_map_features
            mapargs = dict()
            mapargs['ecolor'] = 'red'
            mapargs['xticks'] = xticks
            mapargs['yticks'] = yticks

            if level == 300:
                pcargs['panel_title'] = 'Low Level ({} m)'.format(level)
                mapargs['bottom_labs'] = 'remove'
            elif level == 2000:
                pcargs['panel_title'] = 'Mid Level ({} m)'.format(level)
                pcargs['clab'] = color_label
                mapargs['bottom_labs'] = 'remove'
                mapargs['left_labs'] = 'remove'
            elif level == 6000:
                pcargs['panel_title'] = 'High Level ({} m)'.format(level)

            pf.plot_pcolormesh_panel(fig, axes_ind[i], lon, lat, d, **pcargs)
            cf.add_map_features(axes_ind[i], ax_lims, **mapargs)
            if lease_areas:
                pf.add_lease_area_polygon(axes_ind[i], lease_areas, 'magenta')

        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.02, hspace=0.01)
        plt.savefig(save_fig, dpi=200)
        plt.close()


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

    kwargs = dict()
    # kwargs['lease_areas'] = cf.extract_lease_areas()

    for i, f in enumerate(files):
        fname = f.split('/')[-1].split('.')[0]
        splitter = fname.split('/')[-1].split('_')
        ncfile = xr.open_dataset(f, mask_and_scale=False)
        sfile = cf.save_filepath(save_dir, 'cloud', splitter)
        plt_cloudfrac(ncfile, model_ver, sfile, **kwargs)

    print('')
    print('Script run time: {} minutes'.format(round(((time.time() - start_time) / 60), 2)))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Plot cloud cover',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-wrf_dir',
                            dest='wrf_dir',
                            default='/home/coolgroup/ru-wrf/real-time/v4.1_parallel/processed/modlevs/3km/20200101',
                            type=str,
                            help='Full directory path to subset WRF native model level netCDF files.')

    arg_parser.add_argument('-save_dir',
                            dest='save_dir',
                            default='/home/coolgroup/ru-wrf/real-time/v4.1_parallel/scripts/webfigs/3km',
                            type=str,
                            help='Full directory path to save output plots.')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
