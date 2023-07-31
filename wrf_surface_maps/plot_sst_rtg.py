# Author: James Kim
# date: 07/31/23

#this code will plot the RTG SST input used in WRF model Runs (Made for December 2018 backdate)


import argparse
import numpy as np
import pandas as pd
import os
import glob
import sys
import xarray as xr
import yaml
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cmocean as cmo
import functions.common as cf
import functions.plotting as pf
import functions.hurricanes_plotting as hp
plt.rcParams.update({'font.size': 12})  # all font sizes are 12 unless otherwise specified


def subset_grid(ext, dataset, lon_name, lat_name):
    if len(np.shape(dataset[lon_name])) == 1:
        lonx, laty = np.meshgrid(dataset[lon_name], dataset[lat_name])
    else:
        lonx = dataset[lon_name]
        laty = dataset[lat_name]

    if dataset.name == 'TMP_P0_L1_GLL0':  # RTG dataset
        lonx[lonx > 180] = lonx[lonx > 180] - 360  # convert longitude from 0 to 360 to -180 to 180

    lon_idx = np.logical_and(lonx > ext[0], lonx < ext[1])
    lat_idx = np.logical_and(laty > ext[2], laty < ext[3])

    # find i and j indices of lon/lat in boundaries
    ind = np.where(np.logical_and(lat_idx, lon_idx))

    # subset data from min i,j corner to max i,j corner
    # there will be some points outside of defined boundaries because grid is not rectangular
    data_sub = np.squeeze(dataset)[range(np.min(ind[0]), np.max(ind[0]) + 1), range(np.min(ind[1]), np.max(ind[1]) + 1)]
    lon = data_sub[lon_name]
    lat = data_sub[lat_name]

    return data_sub, lon, lat



