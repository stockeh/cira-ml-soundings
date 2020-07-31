import concurrent.futures
import sys
import time as cpytime
from os import makedirs
from os.path import exists, join

import numpy as np
import pandas as pd
import xarray as xr
from scipy import interpolate

from soundings.preprocessing import goesimager, rtmaloader


class DataHolder(object):

    def __init__(self, sonde):

        self.sonde_time = pd.Timestamp(
            sonde['time'].values[0], unit='s', tz='UTC')
        self.sonde_lon = sonde['lon'].values[0]
        self.sonde_lat = sonde['lat'].values[0]
        self.sonde_file = None
        self.sonde_pres = None
        self.sonde_tdry = None
        self.sonde_dp = None
        self.sonde_alt = None
        self.sonde_site_id = None

        self.goes_files = None
        self.goes_patches = None
        self.goes_patch_lons = None
        self.goes_patch_lats = None

        self.rtma_files = None
        self.rtma_patches = None
        self.rtma_patch_lons = None
        self.rtma_patch_lats = None

    def save(self, processed_dir):
        patch_ds = xr.Dataset(data_vars={'sonde_rel_time': (self.sonde_time),
                                         'sonde_file': (self.sonde_file),
                                         'sonde_pres': (('sonde_profile_dims'), self.sonde_pres),
                                         'sonde_tdry': (('sonde_profile_dims'), self.sonde_tdry),
                                         'sonde_dp': (('sonde_profile_dims'), self.sonde_dp),
                                         'sonde_alt': (('sonde_profile_dims'), self.sonde_alt),
                                         'goes_files': (('band'), self.goes_files),
                                         'goes_abi': (('band', 'goes_y', 'goes_x'), self.goes_patches),
                                         'goes_lon': (('goes_y', 'goes_x'), self.goes_patch_lons),
                                         'goes_lat': (('goes_y', 'goes_x'), self.goes_patch_lats),
                                         'rtma_files': (('rtma_type'), self.rtma_files),
                                         'rtma_values': (('rtma_type', 'rtma_y', 'rtma_x'), self.rtma_patches),
                                         'rtma_lon': (('rtma_y', 'rtma_x'), self.rtma_patch_lons),
                                         'rtma_lat': (('rtma_y', 'rtma_x'), self.rtma_patch_lats)
                                         },
                              coords={'goes_y': np.arange(GOES_CONFIG['patch_y_length_pixels']),
                                      'goes_x': np.arange(GOES_CONFIG['patch_x_length_pixels']),
                                      'band': GOES_CONFIG['bands'],
                                      'rtma_y': np.arange(RTMA_CONFIG['patch_y_length_pixels']),
                                      'rtma_x': np.arange(RTMA_CONFIG['patch_x_length_pixels']),
                                      'rtma_type': RTMA_CONFIG['rtma_type'],
                                      'sonde_profile_dims': np.arange(SONDE_CONFIG['sonde_profile_dims'])})

        patch_ds['sonde_pres'].attrs['units'] = 'hectopascals'
        patch_ds['sonde_tdry'].attrs['units'] = 'celsius'
        patch_ds['sonde_dp'].attrs['units'] = 'celsius'
        patch_ds['sonde_alt'].attrs['units'] = 'meters'
        patch_ds['goes_abi'].attrs['units'] = 'rad' if GOES_CONFIG['bt'] == False else 'bt'
        patch_ds['rtma_values'].attrs['units'] = 'LPI: something, LTI: something, LRI: something'

        out_file = join(
            processed_dir, f"{self.sonde_site_id}_{self.sonde_time.strftime('%Y_%m_%d_%H%M')}.nc")
        print(out_file)
        if not exists(processed_dir):
            makedirs(processed_dir)
        patch_ds.to_netcdf(out_file, engine='netcdf4')


def interpolate_to_height_intervals(alt, y, altitude_intervals):
    f = interpolate.interp1d(alt, y)
    return f(altitude_intervals)


def set_nwp_profile(time, lon, lat, dataset):

    pass


def set_radiosonde_profile(sonde, path, dataset):
    """
    Read NetCDF formatted radiosonde for a specific launch
    Inputs:

    """
    p = sonde.pres.values
    t = sonde.tdry.values
    td = sonde.dp.values
    alt = sonde.alt.values

    alt_s = alt[0]

    # remove duplicate values at surface level
    start_indx = 0
    for i in range(1, len(alt)):
        if alt[i] == alt_s:
            start_indx = i
        else:
            break

    altitude_intervals = np.linspace(
        alt_s, SONDE_CONFIG['alt_el'], SONDE_CONFIG['sonde_profile_dims'])

    dataset.sonde_pres = interpolate_to_height_intervals(
        alt[start_indx:], p[start_indx:], altitude_intervals)
    dataset.sonde_tdry = interpolate_to_height_intervals(
        alt[start_indx:], t[start_indx:], altitude_intervals)
    dataset.sonde_dp = interpolate_to_height_intervals(
        alt[start_indx:], td[start_indx:], altitude_intervals)
    dataset.sonde_alt = altitude_intervals

    dataset.sonde_file = path
    dataset.sonde_site_id = sonde.site_id


def set_rtma_data(time, lon, lat, dataset):
    try:
        rtma_timestep = rtmaloader.RTMALoader(RTMA_CONFIG['path'], time, RTMA_CONFIG['rtma_type'],
                                              time_range_minutes=RTMA_CONFIG['time_range_minutes'])
    except FileNotFoundError as fnfe:  # likely missing a file for all bands
        raise fnfe

    try:
        patches, patch_lons, \
            patch_lats = rtma_timestep.extract_image_patch(lon, lat, RTMA_CONFIG['patch_x_length_pixels'],
                                                           RTMA_CONFIG['patch_y_length_pixels'])
        dataset.rtma_patches = patches[0]
        dataset.rtma_patch_lons = patch_lons
        dataset.rtma_patch_lats = patch_lats
        dataset.rtma_files = np.array(rtma_timestep.rtma_files)

    except ValueError as ve:  # likely invalid lon/lat
        raise ve


def set_goes_data(time, lon, lat, dataset):
    try:
        goes16_abi_timestep = goesimager.GOES16ABI(GOES_CONFIG['abi_path'], time, GOES_CONFIG['bands'],
                                                   time_range_minutes=GOES_CONFIG['time_range_minutes'])
    except FileNotFoundError as fnfe:  # likely missing a file for all bands
        raise fnfe

    try:
        patches, patch_lons, \
            patch_lats = goes16_abi_timestep.extract_image_patch(lon, lat, GOES_CONFIG['patch_x_length_pixels'],
                                                                 GOES_CONFIG['patch_y_length_pixels'], bt=GOES_CONFIG['bt'])
        dataset.goes_patches = patches[0]
        dataset.goes_patch_lons = patch_lons
        dataset.goes_patch_lats = patch_lats
        dataset.goes_files = np.array(goes16_abi_timestep.channel_files)

    except ValueError as ve:  # likely invalid lon/lat
        raise ve


def extract_all_information(root_path):

    start_t = cpytime.time()

    with open(root_path + 'raobs/profiles-alt-files-processed.txt') as fp:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            path = fp.readline().rstrip('\n')
            while path:
                if '.20190912.' not in path:
                    path = fp.readline().rstrip('\n')
                    continue

                sonde = xr.open_dataset(root_path + path[38:])
                dataset = DataHolder(sonde)

                futures = []

                futures.append(pool.submit(set_radiosonde_profile,
                                           sonde, path, dataset))
                futures.append(pool.submit(set_goes_data, dataset.sonde_time, dataset.sonde_lon,
                                           dataset.sonde_lat, dataset))
                futures.append(pool.submit(set_rtma_data, dataset.sonde_time, dataset.sonde_lon,
                                           dataset.sonde_lat, dataset))
                futures.append(pool.submit(set_nwp_profile, dataset.sonde_time, dataset.sonde_lon,
                                           dataset.sonde_lat, dataset))
                try:
                    for future in concurrent.futures.as_completed(futures, timeout=5):
                        try:
                            _ = future.result()
                        except Exception as e:
                            raise e
                    dataset.save(root_path + 'processed')
                except Exception as e:
                    print('ERROR:', e)

                sonde.close()
                del dataset

                path = fp.readline().rstrip('\n')

    print(f"runtime: {cpytime.time()-start_t}")


def set_configs(sonde_config, goes_config, rtma_config, nwp_config=None):
    global SONDE_CONFIG, GOES_CONFIG, RTMA_CONFIG, NWP_CONFIG
    # TODO: REPLACE WITH MAIN
    SONDE_CONFIG = sonde_config
    GOES_CONFIG = goes_config
    RTMA_CONFIG = rtma_config
    NWP_CONFIG = nwp_config
