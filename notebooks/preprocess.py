import xarray as xr
import time as cpytime
import numpy as np
import pandas as pd


def valide_radiosonde(time, lon, lat):
    top = 49.3457868  # north lat
    left = -124.7844079  # west long
    right = -66.9513812  # east long
    bottom = 24.7433195  # south lat
    year = int(time.strftime('%Y'))
    return bottom <= lat <= top \
        and left <= lon <= right \
        and 2017 <= year <= 2019


def extract_all_information(root_path, goes_config):

    start_t = cpytime.time()

    for sonde in [root_path + 'radiosonde/US_25Jun2019.cdf']:

        sonde = xr.open_dataset(radiosonde_path, decode_times=False)
        times = sonde['relTime'].values
        lons = sonde['staLon'].values
        lats = sonde['staLat'].values

        # TODO: extract radiosonde information
        # sonde.close()

        is_valid = np.ones((times.size), dtype=bool)

        goes_patches = np.zeros((times.size, goes_config['bands'].size, goes_config['patch_y_length_pixels'],
                                 goes_config['patch_x_length_pixels']), dtype=np.float32)
        goes_patch_lons = np.zeros(
            (times.size, goes_config['patch_y_length_pixels'], goes_config['patch_x_length_pixels']), dtype=np.float32)
        goes_patch_lats = np.zeros(
            (times.size, goes_config['patch_y_length_pixels'], goes_config['patch_x_length_pixels']), dtype=np.float32)

        for t, time in enumerate(times):

            time = pd.Timestamp(time, unit='s', tz='UTC')

            if not valide_radiosonde(time, lons[t], lats[t]):
                print(t, f'Central Lon ({lons[t]:.3f}) and Lat ({lats[t]:.3f}) '
                      f'does not exist in GOES-16 projection.')
                is_valid[t] = False
                continue
