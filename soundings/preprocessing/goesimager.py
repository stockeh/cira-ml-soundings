import concurrent.futures
import time as cpytime
from datetime import datetime
from glob import glob
from os import makedirs
from os.path import exists, join

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Proj


class GOES16ABICache(object):

    def __init__(self, time_range_minutes=3, cache_size=5):
        self.time_range_minutes = time_range_minutes
        self.goes_collection = []
        self.cache_size = cache_size

    def get_goes(self, time):
        if not isinstance(time, pd.Timestamp):
            time = pd.Timestamp(time, unit='s', tz='UTC')
        for i, goes in enumerate(self.goes_collection):
            if np.abs(time - goes.date) <= pd.Timedelta(minutes=self.time_range_minutes):
                return goes
        return None

    def put_goes(self, goes):
        if len(self.goes_collection) > self.cache_size:
            rem_goes = self.goes_collection[0]
            self.goes_collection = self.goes_collection[1:]
            rem_goes.close()
            del rem_goes
        self.goes_collection.append(goes)

    def clear(self):
        for rem_goes in self.goes_collection:
            rem_goes.close()
            del rem_goes
        self.goes_collection = []


class GOES16ABI(object):
    """Handles data I/O and map projections for GOES-16 Advanced Baseline Imager data.

    :params
    ---
    path : str
        Path to top level of GOES-16 ABI directory
    date : class:`datetime.datetime`
        Date of interest
    bands : class:`numpy.ndarray`
        GOES-16 hyperspectral bands to load
    time_range_minutes : int  
        interval in number of minutes to search for file that matches input time
    goes16_ds : dict  
        Dictionary of of :class:`xarray.Dataset` objects. Datasets for each channel

    """

    def __init__(self, path, date, bands=np.array([8, 9, 10, 11, 13, 14, 15, 16]), time_range_minutes=5):
        if not exists(path):
            raise FileNotFoundError(f'Path: {path} does NOT exist.')
        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date, unit='s', tz='UTC')
        if not isinstance(bands, np.ndarray):
            bands = np.array(bands)
        self.date = date
        self.path = path
        self.bands = bands
        self.time_range_minutes = time_range_minutes

        self.goes16_ds = dict()
        self.channel_files = []
        for band in bands:
            self.channel_files.append(self._goes16_abi_filename(band))
            self.goes16_ds[band] = xr.open_dataset(self.channel_files[-1])

        self.proj = self._goes16_projection()
        self.x = None
        self.y = None
        self.x_g = None
        self.y_g = None
        self.lon = None
        self.lat = None
        self._sat_coordinates()
        self._lon_lat_coords()

    @staticmethod
    def _abi_file_dates(files, file_date='e'):
        """
        Extract the file creation dates from a list of GOES-16 files.
        Date format: Year (%Y), Day of Year (%j), Hour (%H), Minute (%M), Second (%s), Tenth of a second
        See `AWS <https://docs.opendata.aws/noaa-goes16/cics-readme.html>`_ for more details

        :params
        ---
        files : list  
            list of GOES-16 filenames.
        file_date : str  
            date in filename to extract. Valid options are
            's' (start), 'e' (end), and 'c' (creation, default).

        :returns
        ---
        dates : class:`pandas.DatetimeIndex`
         dates for each file
        """
        if file_date not in ['c', 's', 'e']:
            file_date = 'c'
        date_index = {"c": -1, "s": -3, "e": -2}
        channel_dates = pd.DatetimeIndex(
            [datetime.strptime(c_file[:-3].split("/")[-1].split("_")[date_index[file_date]][1:-1],
                               "%Y%j%H%M%S") for c_file in files], tz='UTC')
        return channel_dates

    def _goes16_abi_filename(self, channel):
        """
        Given a path to a dataset of GOES-16 files, find the netCDF file that matches the expected
        date and channel, or band number.

        The GOES-16 path should point to a directory containing a series of directories named by
        valid date in %Y%m%d format. Each directory should contain Level 1 CONUS sector files.

        :params
        ---
        channel : int
            GOES-16 ABI `channel <https://www.goes-r.gov/mission/ABI-bands-quick-info.html>`_.

        :returns
        ---
        filename : str 
            full path to requested GOES-16 file
        """
        channel_files = np.array(sorted(glob(join(self.path, self.date.strftime('%Y'), 'l1b', self.date.strftime('%Y_%m_%d_%j'),
                                                  f"OR_ABI-L1b-RadC-M*C{channel:02d}_G16_*.nc"))))
        # print("channel_files", channel_files)
        channel_dates = self._abi_file_dates(channel_files)
        # print("channel_dates", channel_dates)
        date_diffs = np.abs(channel_dates - self.date)
        # print("Date_diffs", date_diffs)
        file_index = np.where(date_diffs <= pd.Timedelta(
            minutes=self.time_range_minutes))[0]
        # print("File_index", file_index)
        if len(file_index) == 0:
            diff = (date_diffs.total_seconds().values.min() /
                    60) if len(date_diffs) != 0 else float('inf')
            raise FileNotFoundError('No GOES-16 files within {0:d} minutes of '.format(self.time_range_minutes) + self.date.strftime(
                "%Y-%m-%d %H:%M:%S" + ". Nearest file is within {0:.3f} minutes".format(diff)))
        else:
            filename = channel_files[np.argmin(date_diffs)]
        return filename

    def _goes16_projection(self):
        """
        Create a Pyproj projection object with the projection information from a GOES-16 file.
        The geostationary map projection is described in the
        `PROJ <https://proj4.org/operations/projections/geos.html>`_ documentation.

        :returns
        ---
        projection : Pyroj.Proj
        """
        goes16_ds = self.goes16_ds[self.bands.min()]
        proj_dict = dict(proj="geos",
                         h=goes16_ds["goes_imager_projection"].attrs["perspective_point_height"],
                         lon_0=goes16_ds["goes_imager_projection"].attrs["longitude_of_projection_origin"],
                         sweep=goes16_ds["goes_imager_projection"].attrs["sweep_angle_axis"])
        return Proj(projparams=proj_dict)

    def _sat_coordinates(self):
        """
        Calculate the geostationary projection x and y coordinates in m for each pixel in the image.
        """
        goes16_ds = self.goes16_ds[self.bands.min()]
        sat_height = goes16_ds["goes_imager_projection"].attrs["perspective_point_height"]
        self.x = goes16_ds["x"].values * sat_height
        self.y = goes16_ds["y"].values * sat_height
        self.x_g, self.y_g = np.meshgrid(self.x, self.y)

    def _parallel_lon_lat_coords(self, cols, s):
        """
        Allow for sections of the longitude and latitude to be calculated in parallel.
        """
        self.lon[:, s:s+cols], self.lat[:, s:s+cols] = self.proj(
            self.x_g[:, s:s+cols], self.y_g[:, s:s+cols], inverse=True)

    def _lon_lat_coords(self):
        """
        Calculate longitude and latitude coordinates for each point in the GOES-16 image.
        """
        threads = 10
        cols = self.x_g.shape[1] // threads
        self.lon = np.zeros(self.x_g.shape)
        self.lat = np.zeros(self.x_g.shape)
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for section in [cols * i for i in range(threads)]:
                executor.submit(self._parallel_lon_lat_coords, cols, section)
        self.lon[self.lon > 1e10] = np.nan
        self.lat[self.lat > 1e10] = np.nan

    def extract_image_patch(self, center_lon, center_lat, x_size_pixels, y_size_pixels, bt=True):
        """
        Extract a subset of a satellite image around a given location.

        :params
        ---
        center_lon : float 
            longitude of the center pixel of the image
        center_lat : float
            latitude of the center pixel of the image
        x_size_pixels : int
            number of pixels in the west-east direction
        y_size_pixels : int
            number of pixels in the south-north direction
        bt : bool
            Convert to brightness temperature during extraction

        :returns
        ---
        patch : ndarray
        lons : ndarray
        lats : ndarray
        """
        center_x, center_y = self.proj(center_lon, center_lat)
        center_row = np.argmin(np.abs(self.y - center_y))
        center_col = np.argmin(np.abs(self.x - center_x))
        row_slice = slice(int(center_row - y_size_pixels // 2),
                          int(center_row + y_size_pixels // 2))
        col_slice = slice(int(center_col - x_size_pixels // 2),
                          int(center_col + x_size_pixels // 2))
        patch = np.zeros((1, self.bands.size, y_size_pixels,
                          x_size_pixels), dtype=np.float32)
        for b, band in enumerate(self.bands):
            """
            Converting Spectral Radiance to BT (bands 7-16)
            Planck Function constants are used for the conversion
            between radiance (mW/(m2·sr·cm- 1)) and BT (K)
            <https://www.star.nesdis.noaa.gov/goesr/documents/ATBDs/Baseline/ATBD_GOES-R_ABI_CMI_KPP_v3.0_July2012.pdf>
            (fk2 / np.log((fk1 / rad_fnc) + 1.0) - bc1) / bc2
            """
            try:
                if bt and 7 <= band and band <= 16:
                    patch[0, b, :, :] = (self.goes16_ds[band]["planck_fk2"].values /
                                         np.log(self.goes16_ds[band]["planck_fk1"].values / self.goes16_ds[band]["Rad"][row_slice, col_slice].values + 1) -
                                         self.goes16_ds[band]["planck_bc1"].values) / self.goes16_ds[band]["planck_bc2"].values
                else:
                    patch[0, b, :, :] = self.goes16_ds[band]["Rad"][row_slice,
                                                                    col_slice].values
            except ValueError as ve:
                raise ve

        lons = self.lon[row_slice, col_slice]
        lats = self.lat[row_slice, col_slice]

        return patch, lons, lats

    def close(self):
        for band in self.bands:
            self.goes16_ds[band].close()
            del self.goes16_ds[band]


def valide_lon_lat(lon, lat):
    """
    Bounds for longitude + latitude 
    """
    top = 49.3457868  # north lat
    left = -124.7844079  # west long
    right = -66.9513812  # east long
    bottom = 24.7433195  # south lat
    return bottom <= lat <= top and left <= lon <= right


def extract_abi_patches(radiosonde_path, abi_path, patch_path, bands=np.array([8, 9, 10, 11, 13, 14, 15, 16]),
                        patch_x_length_pixels=28, patch_y_length_pixels=28,
                        time_range_minutes=5, bt=False):
    """
    NOTE: Only works when sounding are of the form US_25Jun2019.cdf with all launches in US for June 25, 2019.

    For a given set of gridded GLM counts, sample from the grids at each time step and extract ABI
    patches centered on the lightning grid cell.

    :params
    ---
    radiosonde_path : str
        Path to the radiosonde data
    abi_path : str
        path to GOES-16 ABI input data
    patch_path : str
        Path to GOES-16 output patches
    bands : class:`numpy.ndarray`, int
        timeArray of band numbers
    patch_x_length_pixels : int
        Size of patch in x direction in pixels
    patch_y_length_pixels : int
        Size of patch in y direction in pixels
    time_range_minutes : int
        Minutes before or after time in which GOES16 files are valid.
    bt : bool
        Calculate brightness temperature instead of radiance

    :returns
    ---

    """
    start_t = cpytime.time()
    sonde = xr.open_dataset(radiosonde_path, decode_times=False)
    times = sonde['relTime'].values
    lons = sonde['staLon'].values
    lats = sonde['staLat'].values

    # TODO: extract radiosonde information

    sonde.close()
    del sonde

    patches = np.zeros((times.size, bands.size, patch_y_length_pixels,
                        patch_x_length_pixels), dtype=np.float32)
    patch_lons = np.zeros(
        (times.size, patch_y_length_pixels, patch_x_length_pixels), dtype=np.float32)
    patch_lats = np.zeros(
        (times.size, patch_y_length_pixels, patch_x_length_pixels), dtype=np.float32)
    is_valid = np.ones((times.size), dtype=bool)

    goes16_cache = GOES16ABICache(
        time_range_minutes=int(time_range_minutes//1.5), cache_size=10)
    goes16_abi_timestep = None

    for t, time in enumerate(times):

        if not valide_lon_lat(lons[t], lats[t]):
            print(t, f'Central Lon ({lons[t]:.3f}) and Lat ({lats[t]:.3f}) '
                     f'does not exist in GOES-16 projection.')
            is_valid[t] = False
            continue

        try:
            cached_goes16 = goes16_cache.get_goes(time)
            goes16_abi_timestep = GOES16ABI(abi_path, time, bands, time_range_minutes=time_range_minutes) \
                if cached_goes16 is None else cached_goes16

        except FileNotFoundError as fnfe:  # likely missing a file for all bands
            print(t, fnfe)
            is_valid[t] = False
            continue

        if cached_goes16 is None:
            goes16_cache.put_goes(goes16_abi_timestep)

        try:
            patches[t], \
                patch_lons[t], \
                patch_lats[t] = goes16_abi_timestep.extract_image_patch(lons[t], lats[t], patch_x_length_pixels,
                                                                        patch_y_length_pixels, bt=bt)
        except ValueError as ve:  # likely invalid lon/lat
            print(t, ve)
            is_valid[t] = False

    goes16_cache.clear()

    x_coords = np.arange(patch_x_length_pixels)
    y_coords = np.arange(patch_y_length_pixels)
    valid_patches = np.where(is_valid)[0]
    patch_num = np.arange(valid_patches.shape[0])

    patch_ds = xr.Dataset(data_vars={"abi": (("samples", "band", "y", "x"), patches[valid_patches]),
                                     "time": (("samples", ), times[valid_patches]),
                                     "lon": (("samples", "y", "x"), patch_lons[valid_patches]),
                                     "lat": (("samples", "y", "x"), patch_lats[valid_patches])},
                          coords={"samples": patch_num,
                                  "y": y_coords, "x": x_coords, "bands": bands})

    out_file = join(patch_path, "abi_patches_{0}.nc".format('TEST'))

    if not exists(patch_path):
        makedirs(patch_path)
    patch_ds.to_netcdf(out_file,
                       engine="netcdf4",
                       encoding={"abi": {"zlib": True}, "time": {"zlib": True}, "lon": {"zlib": True},
                                 "lat": {"zlib": True}})
    print(f"runtime: {cpytime.time()-start_t}")
    return 0
