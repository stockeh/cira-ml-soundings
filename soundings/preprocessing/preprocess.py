import time as cpytime
from os import makedirs
from os.path import exists, join

import numpy as np
import pandas as pd
import xarray as xr
from scipy import interpolate

from soundings.preprocessing import goesimager as im


class DataHolder(object):

    def __init__(self, sonde):
        self.n_samples = sonde['relTime'].values.size
        self.sonde_times = sonde['relTime'].values
        self.sonde_lons = sonde['staLon'].values
        self.sonde_lats = sonde['staLat'].values

        self.sonde_profile_P = np.zeros(
            (self.n_samples, SONDE_CONFIG['sonde_profile_dims']), dtype=np.float32)
        self.sonde_profile_T = np.zeros(
            (self.n_samples, SONDE_CONFIG['sonde_profile_dims']), dtype=np.float32)
        self.sonde_profile_Td = np.zeros(
            (self.n_samples, SONDE_CONFIG['sonde_profile_dims']), dtype=np.float32)
        self.sonde_profile_Ws = np.zeros(
            (self.n_samples, SONDE_CONFIG['sonde_profile_dims']), dtype=np.float32)
        self.sonde_profile_Wd = np.zeros(
            (self.n_samples, SONDE_CONFIG['sonde_profile_dims']), dtype=np.float32)

        self.is_valid = np.ones((self.n_samples), dtype=bool)

        self.goes_patches = np.zeros(
            (self.n_samples, GOES_CONFIG['bands'].size, GOES_CONFIG['patch_y_length_pixels'],
             GOES_CONFIG['patch_x_length_pixels']), dtype=np.float32)
        self.goes_patch_lons = np.zeros(
            (self.n_samples, GOES_CONFIG['patch_y_length_pixels'],
             GOES_CONFIG['patch_x_length_pixels']), dtype=np.float32)
        self.goes_patch_lats = np.zeros(
            (self.n_samples, GOES_CONFIG['patch_y_length_pixels'],
             GOES_CONFIG['patch_x_length_pixels']), dtype=np.float32)

    def save(self, processed_dir):
        x_coords = np.arange(GOES_CONFIG['patch_x_length_pixels'])
        y_coords = np.arange(GOES_CONFIG['patch_y_length_pixels'])
        sonde_profile_coords = np.arange(SONDE_CONFIG['sonde_profile_dims'])
        valid_patches = np.where(self.is_valid)[0]
        patch_num = np.arange(valid_patches.shape[0])

        patch_ds = xr.Dataset(data_vars={'goes_abi': (('sample', 'band', 'y', 'x'), self.goes_patches[valid_patches]),
                                         'goes_lon': (('sample', 'y', 'x'), self.goes_patch_lons[valid_patches]),
                                         'goes_lat': (('sample', 'y', 'x'), self.goes_patch_lats[valid_patches]),
                                         'sonde_rel_time': (('sample', ), self.sonde_times[valid_patches]),
                                         'sonde_profile_P': (('sample', 'sonde_profile_dims'), self.sonde_profile_P[valid_patches]),
                                         'sonde_profile_T': (('sample', 'sonde_profile_dims'), self.sonde_profile_T[valid_patches]),
                                         'sonde_profile_Td': (('sample', 'sonde_profile_dims'), self.sonde_profile_Td[valid_patches]),
                                         'sonde_profile_Ws': (('sample', 'sonde_profile_dims'), self.sonde_profile_Ws[valid_patches]),
                                         'sonde_profile_Wd': (('sample', 'sonde_profile_dims'), self.sonde_profile_Wd[valid_patches])
                                         },
                              coords={'sample': patch_num, 'y': y_coords, 'x': x_coords, 'band': GOES_CONFIG['bands'],
                                      'sonde_profile_dims': sonde_profile_coords})
        patch_ds['sonde_profile_P'].attrs['units'] = 'hectopascals'
        patch_ds['sonde_profile_T'].attrs['units'] = 'celsius'
        patch_ds['sonde_profile_Td'].attrs['units'] = 'celsius'
        patch_ds['sonde_profile_Ws'].attrs['units'] = 'knots'
        patch_ds['sonde_profile_Wd'].attrs['units'] = 'degrees'
        patch_ds['goes_abi'].attrs['units'] = 'rad' if GOES_CONFIG['bt'] == False else 'bt'
        out_file = join(processed_dir, 'abi_patches_{0}.nc'.format('TEST2'))

        if not exists(processed_dir):
            makedirs(processed_dir)
        patch_ds.to_netcdf(out_file,
                           engine='netcdf4',
                           encoding={'goes_abi': {'zlib': True}, 'sonde_rel_time': {'zlib': True}, 'goes_lon': {'zlib': True},
                                     'goes_lat': {'zlib': True}, 'sonde_profile_P': {'zlib': True},
                                     'sonde_profile_T': {'zlib': True}, 'sonde_profile_Td': {'zlib': True},
                                     'sonde_profile_Ws': {'zlib': True}, 'sonde_profile_Wd': {'zlib': True}})


def set_radiosonde_profile(sonde, t, dataset):
    """
    Read NetCDF formatted radiosonde for a specific launch
    Inputs:

    """

    numMand = sonde['numMand'][t].values
    numSigT = sonde['numSigT'][t].values
    numSigW = sonde['numSigW'][t].values
    numTrop = sonde['numTrop'][t].values
    numMwnd = sonde['numMwnd'][t].values

    mand = np.vstack((sonde['prMan'][t].values[:numMand], sonde['tpMan'][t].values[:numMand],
                      sonde['tdMan'][t].values[:numMand], sonde['wsMan'][t].values[:numMand],
                      sonde['wdMan'][t].values[:numMand])).T
    sigT = np.vstack((sonde['prSigT'][t].values[:numSigT], sonde['tpSigT'][t].values[:numSigT],
                      sonde['tdSigT'][t].values[:numSigT], sonde['wsSigT'][t].values[:numSigT],
                      sonde['wdSigT'][t].values[:numSigT])).T
    sigW = np.vstack((sonde['prSigW'][t].values[:numSigW], sonde['tpSigW'][t].values[:numSigW],
                      sonde['tdSigW'][t].values[:numSigW], sonde['wsSigW'][t].values[:numSigW],
                      sonde['wdSigW'][t].values[:numSigW])).T
    trop = np.vstack((sonde['prTrop'][t].values[:numTrop], sonde['tpTrop'][t].values[:numTrop],
                      sonde['tdTrop'][t].values[:numTrop], sonde['wsTrop'][t].values[:numTrop],
                      sonde['wdTrop'][t].values[:numTrop])).T
    mwnd = np.vstack((sonde['prMaxW'][t].values[:numMwnd], sonde['tpMaxW'][t].values[:numMwnd],
                      sonde['tdMaxW'][t].values[:numMwnd], sonde['wsMaxW'][t].values[:numMwnd],
                      sonde['wdMaxW'][t].values[:numMwnd])).T

    # P, T, Dp, Ws, Wd
    og_profile = np.concatenate((mand, sigT, sigW, trop, mwnd))
    og_profile = og_profile[og_profile[:, 0].argsort()][::-1]

    dims = og_profile.shape[1]

    # interpolate nan values
    for i in range(1, dims):
        y = og_profile[:, i]
        nans, x = np.isnan(y), lambda z: z.nonzero()[0]
        f = interpolate.interp1d(x(~nans), y[~nans], kind='linear', bounds_error=False,
                                 fill_value=(y[~nans][0], y[~nans][-1]))
        y[nans] = f(x(nans))

    # convert Dew Point Depression to Dew Point Temperature
    og_profile[:, 2] = -(og_profile[:, 2] - og_profile[:, 1]) - 273.15
    og_profile[:, 1] -= 273.15

    profile = np.zeros((SONDE_CONFIG['sonde_profile_dims'], dims))

    # interpolate all values to sonde_profile_dims
    for i in range(dims):
        y = og_profile[:, i]
        profile[:, i] = interpolate_profile(
            y, SONDE_CONFIG['sonde_profile_dims'])

    dataset.sonde_profile_P[t] = profile[:, 0]
    dataset.sonde_profile_T[t] = profile[:, 1]
    dataset.sonde_profile_Td[t] = profile[:, 2]
    dataset.sonde_profile_Ws[t] = profile[:, 3]
    dataset.sonde_profile_Wd[t] = profile[:, 4]


def set_goes_data(time, t, lon, lat, goes16_cache, dataset):
    try:
        cached_goes16 = goes16_cache.get_goes(time)
        goes16_abi_timestep = im.GOES16ABI(GOES_CONFIG['abi_path'], time, GOES_CONFIG['bands'],
                                           time_range_minutes=GOES_CONFIG['time_range_minutes']) \
            if cached_goes16 is None else cached_goes16

    except FileNotFoundError as fnfe:  # likely missing a file for all bands
        print(t, fnfe)
        dataset.is_valid[t] = False
        return

    if cached_goes16 is None:
        goes16_cache.put_goes(goes16_abi_timestep)

    try:
        dataset.goes_patches[t], \
            dataset.goes_patch_lons[t], \
            dataset.goes_patch_lats[t] = goes16_abi_timestep.extract_image_patch(lon, lat, GOES_CONFIG['patch_x_length_pixels'],
                                                                                 GOES_CONFIG['patch_y_length_pixels'], bt=GOES_CONFIG['bt'])
    except ValueError as ve:  # likely invalid lon/lat
        print(t, ve)
        dataset.is_valid[t] = False


def valid_radiosonde(time, lon, lat):
    top = 49.3457868  # north lat
    left = -124.7844079  # west long
    right = -66.9513812  # east long
    bottom = 24.7433195  # south lat
    year = int(time.strftime('%Y'))
    return bottom <= lat <= top \
        and left <= lon <= right \
        and 2017 <= year <= 2019


def extract_all_information(root_path):

    start_t = cpytime.time()

    for sonde_path in [root_path + 'radiosonde/US_25Jun2019.cdf']:

        sonde = xr.open_dataset(sonde_path, decode_times=False)

        dataset = DataHolder(sonde)
        goes16_cache = im.GOES16ABICache(
            time_range_minutes=int(GOES_CONFIG['time_range_minutes']//1.5), cache_size=10)

        for t, time in enumerate(dataset.sonde_times):

            time = pd.Timestamp(time, unit='s', tz='UTC')
            lon, lat = dataset.sonde_lons[t], dataset.sonde_lats[t]

            if not valid_radiosonde(time, lon, lat):
                print(
                    t, f'Invalid sounding time: {time} lon: {lon:.3f} lat: {lat:.3f}')
                dataset.is_valid[t] = False
                continue

            set_goes_data(time, t, lon, lat, goes16_cache, dataset)
            set_radiosonde_profile(sonde, t, dataset)
            # set_rtma_data
            # set_nwp_profile

        sonde.close()
        goes16_cache.clear()
        dataset.save(root_path + 'processed')
        del dataset

    print(f"runtime: {cpytime.time()-start_t}")


def interpolate_profile(y, sonde_profile_dims=2000):
    f = interpolate.interp1d(np.arange(len(y)), y, kind='linear')
    return f(np.linspace(0, len(y) - 1, sonde_profile_dims))


def set_configs(sonde_config, goes_config, rtma_config=None, nwp_config=None):
    global SONDE_CONFIG, GOES_CONFIG, RTMA_CONFIG, NWP_CONFIG
    # TODO: REPLACE WITH MAIN
    SONDE_CONFIG = sonde_config
    GOES_CONFIG = goes_config
    RTMA_CONFIG = rtma_config
    NWP_CONFIG = nwp_config
