from scipy import interpolate
from glob import glob
from os.path import join

import xarray as xr
import numpy as np

import os
import time

TOP_BOUNDARY_PRESSURE = 50 # mb


def valid_raob(xar):
    """Greater than N observations with all valid QC values
    """
    return (xar.time.values.size >= 1000 and 
        all(np.unique(xar.qc_pres.values) == [0]) 
        and all(np.unique(xar.qc_tdry.values) == [0]) 
        and all(np.unique(xar.qc_dp.values) == [0]))


def read_valid_profiles(vol):
    files = np.array(glob(join(vol, '*/sgpsondewnpnC1.b1.*.cdf')))

    profiles = []
    valid_profiles = []

    s = time.time()
    for i, f in enumerate(files):
        xar = xr.open_dataset(f)
        if valid_raob(xar):
            profile = np.concatenate((xar.pres.values.reshape(-1,1),
                                      xar.tdry.values.reshape(-1,1),
                                      xar.dp.values.reshape(-1,1)), axis=1)
            profiles.append(profile)
            valid_profiles.append(f)
        xar.close()

    e = time.time() - s
    print(f'time: {e:.3f}, avg: {e/files.size:.3f} seconds')
    return profiles, valid_profiles

def valid_profile_bounries(profiles):
    valid_boundary_indicies = []
    for i, profile in enumerate(profiles):
        if profile[:, 0][-1] <= top_boundary: # just above the min EL
            valid_boundary_indicies.append(i)
    pfs = np.array(profiles)
    return pfs[valid_profile_boundries]


def interpolate_to_sigma_intervals(sigma, y, sigma_n):
    f = interpolate.interp1d(sigma, y)
    return f(sigma_n)


def presure_to_sigma(p):
    """𝜎 = (𝑝−𝑝𝑇)/(𝑝𝑆−𝑝𝑇)"""
    return (p - p[-1])/(p[0] - p[-1])


def sigma_to_pressure(sigma, p):
    """𝑝 = 𝑝𝑇 + 𝜎(𝑝𝑆−𝑝𝑇)"""
    return p[-1] + sigma*(p[0] - p[-1])
