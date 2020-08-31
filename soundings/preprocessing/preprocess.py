import argparse
import concurrent.futures
import sys
import time as cpytime
import os
from os.path import exists, join
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy import interpolate

from soundings.preprocessing import goesimager, rtmaloader, raploader


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

        self.nwp_file = None
        self.nwp_lon = None
        self.nwp_lat = None
        self.nwp_pres = None
        self.nwp_tdry = None
        self.nwp_spfm = None
        self.nwp_alt = None
        
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
                                         'sonde_pres': (('profile_dims'), self.sonde_pres),
                                         'sonde_tdry': (('profile_dims'), self.sonde_tdry),
                                         'sonde_dp': (('profile_dims'), self.sonde_dp),
                                         'sonde_alt': (('profile_dims'), self.sonde_alt),
                                         'nwp_file': (self.nwp_file),
                                         'nwp_lon': (self.nwp_file),
                                         'nwp_lat': (self.nwp_file),
                                         'nwp_pres': (('nwp_dims'), self.nwp_pres),
                                         'nwp_tdry': (('nwp_dims'), self.nwp_tdry),
                                         'nwp_spfm': (('nwp_dims'), self.nwp_spfm),
                                         'nwp_alt': (('nwp_dims'), self.nwp_alt),
                                         'goes_files': (('band'), self.goes_files),
                                         'goes_abi': (('band', 'goes_y', 'goes_x'), self.goes_patches),
                                         'goes_lon': (('goes_y', 'goes_x'), self.goes_patch_lons),
                                         'goes_lat': (('goes_y', 'goes_x'), self.goes_patch_lats),
                                         'rtma_files': (('rtma_type'), self.rtma_files),
                                         'rtma_values': (('rtma_type', 'rtma_y', 'rtma_x'), self.rtma_patches),
                                         'rtma_lon': (('rtma_y', 'rtma_x'), self.rtma_patch_lons),
                                         'rtma_lat': (('rtma_y', 'rtma_x'), self.rtma_patch_lats)
                                         },
                              coords={'goes_y': np.arange(config['goes']['patch_y_length_pixels']),
                                      'goes_x': np.arange(config['goes']['patch_x_length_pixels']),
                                      'band': config['goes']['bands'],
                                      'rtma_y': np.arange(config['rtma']['patch_y_length_pixels']),
                                      'rtma_x': np.arange(config['rtma']['patch_x_length_pixels']),
                                      'rtma_type': config['rtma']['rtma_type'],
                                      'profile_dims': np.arange(config['raob']['profile_dims']),
                                      'nwp_dims': np.arange(config['nwp']['nwp_dims'])})

        patch_ds['sonde_pres'].attrs['units'] = 'hectopascals'
        patch_ds['sonde_tdry'].attrs['units'] = 'celsius'
        patch_ds['sonde_dp'].attrs['units'] = 'celsius'
        patch_ds['sonde_alt'].attrs['units'] = 'meters'
        patch_ds['goes_abi'].attrs['units'] = 'rad' if config['goes']['bt'] == False else 'bt'
        patch_ds['rtma_values'].attrs['units'] = 'LPI: something, LTI: something, LRI: something'

        out_file = join(
            processed_dir, f"{self.sonde_site_id}_{self.sonde_time.strftime('%Y_%m_%d_%H%M')}.nc")
        print(out_file)
        if not exists(processed_dir):
            os.makedirs(processed_dir)
        try:
            os.remove(out_file)
        except OSError:
            pass
        patch_ds.to_netcdf(out_file, engine='netcdf4')


def interpolate_to_height_intervals(alt, y, altitude_intervals):
    # alititude does not always increase mononically, 
    # however, assume_sorted if True, x has to be an array of 
    # monotonically increasing values... 
    f = interpolate.interp1d(alt, y, assume_sorted=True)
    return f(altitude_intervals)


def set_nwp_profile(time, lon, lat, dataset):
    try:
        rap_timestep = raploader.RAPLoader(config['nwp']['path'], time, 
                                              time_range_minutes=config['nwp']['time_range_minutes'])
    except FileNotFoundError as fnfe:  # likely missing a file for all bands
        raise fnfe
    pres, temp, spec, height, rap_lon, rap_lat = rap_timestep.extract_rap_profile(lon, lat, config['nwp']['wgrib2'])

    dataset.nwp_file = rap_timestep.rap_file
    dataset.nwp_lon = rap_lon
    dataset.nwp_lat = rap_lat
    dataset.nwp_pres = pres
    dataset.nwp_tdry = temp
    dataset.nwp_spfm = spec
    dataset.nwp_alt = height


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
        alt_s, config['raob']['alt_el'], config['raob']['profile_dims'])
    np.set_printoptions(threshold=sys.maxsize)
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
        rtma_timestep = rtmaloader.RTMALoader(config['rtma']['path'], time, config['rtma']['rtma_type'],
                                              time_range_minutes=config['rtma']['time_range_minutes'])
    except FileNotFoundError as fnfe:  # likely missing a file for all bands
        raise fnfe

    try:
        patches, patch_lons, \
            patch_lats = rtma_timestep.extract_image_patch(lon, lat, config['rtma']['patch_x_length_pixels'],
                                                           config['rtma']['patch_y_length_pixels'])
        dataset.rtma_patches = patches[0]
        dataset.rtma_patch_lons = patch_lons
        dataset.rtma_patch_lats = patch_lats
        dataset.rtma_files = np.array(rtma_timestep.rtma_files)

    except ValueError as ve:  # likely invalid lon/lat
        raise ve
        
    rtma_timestep.close()

def set_goes_data(time, lon, lat, dataset):
    try:
        goes16_abi_timestep = goesimager.GOES16ABI(config['goes']['path'], time, config['goes']['bands'],
                                                   time_range_minutes=config['goes']['time_range_minutes'])
    except FileNotFoundError as fnfe:  # likely missing a file for all bands
        raise fnfe

    try:
        patches, patch_lons, \
            patch_lats = goes16_abi_timestep.extract_image_patch(lon, lat, config['goes']['patch_x_length_pixels'],
                                                                 config['goes']['patch_y_length_pixels'], 
                                                                 bt=config['goes']['bt'])
        dataset.goes_patches = patches[0]
        dataset.goes_patch_lons = patch_lons
        dataset.goes_patch_lats = patch_lats
        dataset.goes_files = np.array(goes16_abi_timestep.channel_files)

    except ValueError as ve:  # likely invalid lon/lat
        raise ve
        
    goes16_abi_timestep.close()


def extract_all_information():

    start_t = cpytime.time()
    
    already_processed = glob(join(config['output_path'], '*'))
    
    with open(config['raob']['valid_files_path']) as fp:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            path = fp.readline().rstrip('\n')
            while path:
                if '.20200202' not in path:
                    path = fp.readline().rstrip('\n')
                    continue
        
                # arm-sgp / year / file.cdf
                sonde = xr.open_dataset(
                    join(config['raob']['path'], *path.split('/')[-3:]))
                dataset = DataHolder(sonde)
                
                if f"{config['output_path']}/sgp_{dataset.sonde_time.strftime('%Y_%m_%d_%H%M')}.nc" in already_processed:
                    path = fp.readline().rstrip('\n')
                    continue

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
                    for future in concurrent.futures.as_completed(futures, timeout=20):
                        try:
                            _ = future.result()
                        except Exception as e:
                            raise e
                    dataset.save(config['output_path'])
                except Exception as e:
                    print(f"ERROR: {path.split('/')[-1]}, {e}")

                sonde.close()
                del dataset

                path = fp.readline().rstrip('\n')

    print(f"runtime: {cpytime.time()-start_t}")


def main(config_path):
    global config
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)
    extract_all_information()


if __name__ == "__main__":
    """
    Usage: python -m soundings.preprocessing.preprocess -c ./soundings/preprocessing/config.yaml
    """
    parser = argparse.ArgumentParser(description='data preprocessing')
    parser.add_argument('-c', '--config', metavar='path', type=str,
                        required=True, help='the path to config file')
    args = parser.parse_args()
    main(config_path=args.config)
