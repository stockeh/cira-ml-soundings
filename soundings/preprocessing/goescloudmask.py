import concurrent.futures
import time as cpytime
from datetime import datetime
from glob import glob
from os import makedirs
from os.path import exists, join

from tqdm import tqdm

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Proj


class GOES16ABICache(object):

    def __init__(self, time_range_minutes=20, cache_size=5):
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


class GOES16BCM(object):
    """Handles data I/O and map projections for GOES-16 Advanced Baseline Imager data.

    :params
    ---
    path : str
        Path to top level of GOES-16 ABI directory
    date : class:`datetime.datetime`
        Date of interest
    time_range_minutes : int  
        interval in number of minutes to search for file that matches input time
    goes16_ds : dict  
        Dictionary of of :class:`xarray.Dataset` objects. Datasets for each channel

    """

    def __init__(self, path, date, time_range_minutes=60):
        if not exists(path):
            raise FileNotFoundError(f'Path: {path} does NOT exist.')
        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date, unit='s', tz='UTC')
        self.date = date
        self.path = path # /mnt/hilburnnas1/goes16/ABI/RadC/ACM/
        self.time_range_minutes = time_range_minutes

        self.file = self._goes16_bcm_filename()
        try:
            self.goes16_ds = xr.open_dataset(self.file, decode_times=False)
        except Exception as e:
            print('Unable to read NetCDF:', self.file, e)
            raise e
            
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

    def _goes16_bcm_filename(self):
        """
        Given a path to a dataset of GOES-16 files, find the netCDF file that matches the date

        The GOES-16 path should point to a directory containing a series of directories named by
        valid date in %Y%m%d format. Each directory should contain Level 1 CONUS sector files.

        :params
        ---
        :returns
        ---
        filename : str 
            full path to requested GOES-16 file
        """
        
        # /mnt/hilburnnas1/goes16/ABI/RadC/ACM/
        # 2017/0419/OR_ABI-L2-ACMC-M3_G16_s20171091202221_e20171091204594_c20171091205164.nc'
        daily_files = np.array(sorted(glob(join(self.path, self.date.strftime('%Y'), 
                                                self.date.strftime('%m%d'), 'OR_ABI-L2-*.nc'))))
        # print("daily_files", daily_files)
        channel_dates = self._abi_file_dates(daily_files)
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
            filename = daily_files[np.argmin(date_diffs)]
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
        proj_dict = dict(proj="geos",
                         h=self.goes16_ds["goes_imager_projection"].attrs["perspective_point_height"],
                         lon_0=self.goes16_ds["goes_imager_projection"].attrs["longitude_of_projection_origin"],
                         sweep=self.goes16_ds["goes_imager_projection"].attrs["sweep_angle_axis"])
        return Proj(projparams=proj_dict)

    def _sat_coordinates(self):
        """
        Calculate the geostationary projection x and y coordinates in m for each pixel in the image.
        """
        sat_height = self.goes16_ds["goes_imager_projection"].attrs["perspective_point_height"]
        self.x = self.goes16_ds["x"].values * sat_height
        self.y = self.goes16_ds["y"].values * sat_height
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

        :returns
        ---
        patch : ndarray
        lons : ndarray
        lats : ndarray
        """
        center_x, center_y = self.proj(center_lon, center_lat)
        center_row = np.argmin(np.abs(self.y - center_y))
        center_col = np.argmin(np.abs(self.x - center_x))
        # Removed integer division // to allow for odd sizes images
        row_slice = slice(int(center_row - y_size_pixels / 2),
                          int(center_row + y_size_pixels / 2))
        col_slice = slice(int(center_col - x_size_pixels / 2),
                          int(center_col + x_size_pixels / 2))
        patch = np.zeros((1, 1, y_size_pixels, x_size_pixels), dtype=np.float32)
        
        try:
            patch[0, 0, :, :] = self.goes16_ds["BCM"][row_slice, col_slice].values
        except ValueError as ve:
            raise ValueError(f'lon {center_lon} and lat {center_lat} out of range')

        lons = self.lon[row_slice, col_slice]
        lats = self.lat[row_slice, col_slice]

        return patch, lons, lats

    def close(self):
        self.goes16_ds.close()
        del self.goes16_ds


def valide_lon_lat(lon, lat):
    """
    Bounds for longitude + latitude 
    """
    top = 49.3457868  # north lat
    left = -124.7844079  # west long
    right = -66.9513812  # east long
    bottom = 24.7433195  # south lat
    return bottom <= lat <= top and left <= lon <= right


def extract_bcm_patches(sonde_files, bcm_path, patch_x_length_pixels=128, patch_y_length_pixels=128,
                        time_range_minutes=60):
    """
    :params
    ---
    sonde_files : str
        files with lat lon soundings
    bcm_path : str
        path to GOES-16 BCM input data
    patch_x_length_pixels : int
        Size of patch in x direction in pixels
    patch_y_length_pixels : int
        Size of patch in y direction in pixels
    time_range_minutes : int
        Minutes before or after time in which GOES16 files are valid.

    :returns
    ---

    """
    start_t = cpytime.time()
    patches = np.zeros((sonde_files.size, 1, patch_y_length_pixels, patch_x_length_pixels), dtype=np.float32)
    
    goes16_cache = GOES16ABICache(
        time_range_minutes=int(time_range_minutes//1.5), cache_size=10)
    goes16_abi_timestep = None
    
    for t, s in tqdm(enumerate(sonde_files)):
        content = s.split('_')
        time = pd.to_datetime(content[1], utc=True)
        lon = float(content[-2])
        lat = float(content[-1])

        try:
            cached_goes16 = goes16_cache.get_goes(time)
            goes16_abi_timestep = GOES16BCM(bcm_path, time, time_range_minutes=time_range_minutes) \
                if cached_goes16 is None else cached_goes16

        except Exception as e:  # likely missing a file for all bands
            # print(t, e)
            patches[t] = -1
            continue

        if cached_goes16 is None:
            goes16_cache.put_goes(goes16_abi_timestep)

        try:
            patches[t],_,_ = goes16_abi_timestep.extract_image_patch(lon, lat, patch_x_length_pixels,
                                                                     patch_y_length_pixels)
        except ValueError as ve:  # likely invalid lon/lat
            # print(t, ve)
            patches[t] = -1
            
    goes16_cache.clear()
    
    print(f"runtime: {cpytime.time()-start_t}")
    return patches
