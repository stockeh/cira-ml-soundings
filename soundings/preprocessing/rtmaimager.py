import concurrent.futures
import time as cpytime
from datetime import datetime
from glob import glob
from os import makedirs
from os.path import exists, join

import numpy as np
import pandas as pd
import pygrib
from pyproj import Proj


class GOES16ABI(object):
    """Handles data I/O and map projections for Real-Time Mesoscale Analysis data.

    :params
    ---
    path : str
        Path to top level of RTMA directory
    date : class:`datetime.datetime`
        Date of interest
    time_range_minutes : int  
        interval in number of minutes to search for file that matches input time
    """

    def __init__(self, path, date, time_range_minutes=5):
        if not exists(path):
            raise FileNotFoundError(f'Path: {path} does NOT exist.')
        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date, unit='s', tz='UTC')
        self.date = date
        self.path = path
        self.time_range_minutes = time_range_minutes

        self.rtma_ds = dict()
        self.channel_files = []
        self.data_index = 1  # TODO: Check which index is always the data
        self.rtma_types = np.array(['LPI', 'LTI', 'LRI'])
        for rtma_type in self.rtma_types:
            self.channel_files.append(self._goes16_abi_filename(rtma_type))
            self.rtma_ds[rtma_type] = pygrib.open(self.channel_files[-1])

        self.proj = Proj(self.rtma_ds[self.rtma_types[0]]
                         [self.data_index].projparams)
        self.x = None
        self.y = None
        self.lon = None
        self.lat = None
        self._set_coordinates()
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
            Date in filename to extract. Valid options are
            's' (start), 'e' (end), and 'c' (creation, default).

        :returns
        ---
        dates : class:`pandas.DatetimeIndex`
        Dates for each file
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
        channel_files = np.array(sorted(glob(join(self.path, self.date.strftime('%Y'), self.date.strftime('%Y_%m_%d_%j'),
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

    def _set_coordinates(self):
        """
        Calculate the projection x and y coordinates in m for each pixel in the image.

        Referenced from here: https://ftp.emc.ncep.noaa.gov/mmb/bblake/plot_allvars.py
        """
        rtma_ds = self.rtma_ds[self.rtma_types[0]][self.data_index]
        lat1 = rtma_ds['latitudeOfFirstGridPointInDegrees']
        lon1 = rtma_ds['longitudeOfFirstGridPointInDegrees']

        try:
            nx = rtma_ds['Nx']
            ny = rtma_ds['Ny']
        except:
            nx = rtma_ds['Ni']
            ny = rtma_ds['Nj']

        llcrnrx, llcrnry = self.proj(lon1, lat1)

        dx = rtma_ds['DxInMetres']
        dy = rtma_ds['DyInMetres']

        llcrnrx = llcrnrx - (dx/2.)
        llcrnry = llcrnry - (dy/2.)
        x = llcrnrx + dx*np.arange(nx)
        y = (llcrnry + dy*np.arange(ny))[::-1]  # flip vertically

        self.x, self.y = np.meshgrid(x, y)

    def _parallel_lon_lat_coords(self, cols, s):
        """
        Allow for sections of the longitude and latitude to be calculated in parallel.
        """
        self.lon[:, s:s+cols], self.lat[:, s:s+cols] = self.proj(
            self.x[:, s:s+cols], self.y[:, s:s+cols], inverse=True)

    def _lon_lat_coords(self):
        """
        Calculate longitude and latitude coordinates for each point in the RTMA data.
        """
        threads = 10
        cols = self.x.shape[1] // threads
        self.lon = np.zeros(self.x.shape)
        self.lat = np.zeros(self.x.shape)
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for section in [cols * i for i in range(threads)]:
                executor.submit(self._parallel_lon_lat_coords, cols, section)
        self.lon[self.lon > 1e10] = np.nan
        self.lat[self.lat > 1e10] = np.nan

    def extract_image_patch(self, center_lon, center_lat, x_size_pixels, y_size_pixels, bt=True):
        """
        Extract a subset of an image around a given location.

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
        row_slice = slice(int(center_row - y_size_pixels // 2),
                          int(center_row + y_size_pixels // 2))
        col_slice = slice(int(center_col - x_size_pixels // 2),
                          int(center_col + x_size_pixels // 2))
        patch = np.zeros((1, self.rtma_types.size, y_size_pixels,
                          x_size_pixels), dtype=np.float32)
        for b, rtma_type in enumerate(self.rtma_types):
            try:
                # flip vertically
                values = self.rtma_ds[rtma_type][self.data_index][::-1]
                patch[0, b, :, :] = values[row_slice, col_slice].values
            except ValueError as ve:
                raise ve

        lons = self.lon[row_slice, col_slice]
        lats = self.lat[row_slice, col_slice]

        return patch, lons, lats

    def close(self):
        for rtma_type in self.rtma_types:
            self.rtma_ds[rtma_type].close()
            del self.rtma_ds[rtma_type]
