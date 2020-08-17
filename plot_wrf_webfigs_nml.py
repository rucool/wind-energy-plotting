#!/usr/bin/env python

"""
Author: Lori Garzio on 8/17/2020
Last modified: 8/17/2020
This is a wrapper script that imports tools to plot RU-WRF 4.1 data using the subset native model level .nc files.
The plots are used to populate RUCOOL's RU-WRF webpage:
https://rucool.marine.rutgers.edu/data/meteorological-modeling/ruwrf-mesoscale-meteorological-model-forecast/
"""

import argparse
import sys
import wrf_webfigs

arg_parser = argparse.ArgumentParser(description='Plot RU-WRF figs',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

arg_parser.add_argument('-wrf_dir',
                        dest='wrf_dir',
                        default='/home/coolgroup/ru-wrf/real-time/v4.1_parallel/processed/modlevs/3km/20200101',
                        type=str,
                        help='Full directory path to subset WRF netCDF files.')

arg_parser.add_argument('-save_dir',
                        dest='save_dir',
                        default='/home/coolgroup/ru-wrf/real-time/v4.1_parallel/scripts/webfigs/3km',
                        type=str,
                        help='Full directory path to save output plots.')

parsed_args = arg_parser.parse_args()

print('Plotting cloud cover')
wrf_webfigs.plot_cloudfrac.main(parsed_args)

sys.exit()
