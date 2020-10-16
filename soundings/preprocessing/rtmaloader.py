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

import sys


class RTMALoader(object):
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

    def __init__(self, path, date, rtma_types=np.array(['LPI']), time_range_minutes=60):
        if not exists(path):
            raise FileNotFoundError(f'Path: {path} does NOT exist.')
        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date, unit='s', tz='UTC')
        if not isinstance(rtma_types, np.ndarray):
            rtma_types = np.array(rtma_types)
        self.date = date
        self.path = path
        self.time_range_minutes = time_range_minutes
        self.rtma_types = rtma_types  # , 'LTI', 'LRI'

        self.rtma_ds = dict()
        self.rtma_files = []
        self.analysis_index = []

        for rtma_type in self.rtma_types:
            self.rtma_files.append(self._rtma_filename(rtma_type))
            self.rtma_ds[rtma_type] = pygrib.open(self.rtma_files[-1])
            for i in range(1, self.rtma_ds[rtma_type].messages + 1):
                if self.rtma_ds[rtma_type][i]['typeOfGeneratingProcess'] == 0:
                    self.analysis_index.append(i)
                    break

        self.proj = Proj(self.rtma_ds[self.rtma_types[0]]
                         [self.analysis_index[0]].projparams)
        self.x = None
        self.y = None
        self.lon = None
        self.lat = None
        self._set_coordinates()
        self._lon_lat_coords()

    @staticmethod
    def _rtma_file_dates(files):
        """
        Extract the file creation dates from a list of RTMA files.
        Date format: Year (%Y), Month (%m), Hour (%H), Minute (%M), Second (%s),

        :params
        ---
        files : list
            list of RTMA filenames.

        :returns
        ---
        dates : class:`pandas.DatetimeIndex`
            dates for each file
        """
        return pd.DatetimeIndex(
            [datetime.strptime(c_file.split("/")[-1].split("_")[-1], "%Y%m%d%H%M")
             for c_file in files], tz='UTC')

    def _rtma_filename(self, rtma_type):
        """
        Given a path to a dataset of RTMA files, find the grib file that matches the expected
        date and rtma type.

        The RTMA path should point to a directory containing a series of directories named by
        valid date in %Y%m%d format.

        :params
        ---
        rtma_type : str
            RTMA data type, e.g., pressure, temperature, dewpoint as ['LPI', 'LTI', 'LRI']

        :returns
        ---
        filename : str
            full path to requested RTMA file
        """
        rtma_files = np.array(sorted(glob(join(self.path, rtma_type.lower(), self.date.strftime('%Y'),
                                               f"*_{rtma_type}_{self.date.strftime('%Y%m')}*", "*"))))
        # print("rtma_files", rtma_files)
        rtma_dates = self._rtma_file_dates(rtma_files)
        # print("rtma_dates", rtma_dates)
        date_diffs = np.abs(rtma_dates - self.date)
        # print("Date_diffs", date_diffs)
        file_index = np.where(date_diffs <= pd.Timedelta(
            minutes=self.time_range_minutes))[0]
        # print("File_index", file_index, len(file_index))
        if len(file_index) == 0:
            diff = (date_diffs.total_seconds().values.min() /
                    60) if len(date_diffs) != 0 else float('inf')
            raise FileNotFoundError(
                f"No {rtma_type} RTMA files within {self.time_range_minutes} minutes of "
                f"{self.date.strftime('%Y-%m-%d %H:%M:%S')}. Nearest file is within {diff:.3f} minutes")
        else:
            filename = rtma_files[np.argmin(date_diffs)]
        return filename

    def _set_coordinates(self):
        """
        Calculate the projection x and y coordinates in m for each pixel in the image.

        Referenced from here: https://ftp.emc.ncep.noaa.gov/mmb/bblake/plot_allvars.py
        """
        rtma_ds = self.rtma_ds[self.rtma_types[0]][self.analysis_index[0]]
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
        # flip vertically
        y = (llcrnry + dy*np.arange(ny))[::-1]

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

    def extract_image_patch(self, center_lon, center_lat, x_size_pixels, y_size_pixels):
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
        center_row = np.unravel_index(
            np.argmin(np.abs(self.y - center_y)), self.y.shape)[0]
        center_col = np.unravel_index(
            np.argmin(np.abs(self.x - center_x)), self.x.shape)[1]
        
        # Removed integer division // to allow for odd sizes images
        row_slice = slice(int(center_row - y_size_pixels / 2),
                          int(center_row + y_size_pixels / 2))
        col_slice = slice(int(center_col - x_size_pixels / 2),
                          int(center_col + x_size_pixels / 2))
        
        patch = np.zeros((1, self.rtma_types.size, y_size_pixels,
                          x_size_pixels), dtype=np.float32)
        for t, rtma_type in enumerate(self.rtma_types):
            try:
                # flip vertically
                values = self.rtma_ds[rtma_type][self.analysis_index[t]].values[::-1]
                patch[0, t, :, :] = values[row_slice, col_slice]
            except ValueError as ve:
                raise ValueError(f'lon {center_lon} and lat {center_lat} out of range')

        lons = self.lon[row_slice, col_slice]
        lats = self.lat[row_slice, col_slice]

        return patch, lons, lats

    def close(self):
        for rtma_type in self.rtma_types:
            self.rtma_ds[rtma_type].close()
            del self.rtma_ds[rtma_type]
