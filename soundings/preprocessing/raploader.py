from datetime import datetime
from glob import glob
from os.path import exists, join

import pandas as pd
import numpy as np
import pygrib
import subprocess

class RAPLoader(object):
    """Handles data I/O and map projections for the Rapid Refresh data using `wgrib2`.

    :params
    ---
    path : str
        Path to top level of Rap directory
    date : class:`datetime.datetime`
        Date of interest
    time_range_minutes : int
        interval in number of minutes to search for file that matches input time
    """

    def __init__(self, path, date, time_range_minutes=120):
        if not exists(path):
            raise FileNotFoundError(f'Path: {path} does NOT exist.')
        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date, tz='UTC')
            
        self.date = date
        self.path = path
        self.time_range_minutes = time_range_minutes

        self.rap_file = self._rap_filename()
        
        
    @staticmethod
    def _rap_file_dates(files):
        """
        Extract the file creation dates from a list of RAP files.
        Date format: Year no Century (%y), Day of Year (%j), Hour (%h),

        :params
        ---
        files : list
            list of RAP filenames.

        :returns
        ---
        dates : class:`pandas.DatetimeIndex`
            dates for each file
        """
        return pd.DatetimeIndex( # remove trailing 0's
            [datetime.strptime(c_file.split("/")[-1][:-6], "%y%j%H")
             for c_file in files], tz='UTC')

    def _rap_filename(self):
        """
        Given a path to a dataset of RAP files, find the grib file that matches the expected
        date.

        The RAP path should point to a directory containing a series of directories named by
        valid date in %Y%m%d format.

        :returns
        ---
        filename : str
            full path to requested RAP file
        """
        rap_files = np.array(sorted(glob(join(self.path, self.date.strftime('%Y%m%d'),'*'))))
        # print("rap_files", rap_files)
        rap_dates = self._rap_file_dates(rap_files)
        # print("rap_dates", rap_dates)
        date_diffs = np.abs(rap_dates - self.date)
        # print("Date_diffs", date_diffs)
        file_index = np.where(date_diffs <= pd.Timedelta(
            minutes=self.time_range_minutes))[0]
        # print("File_index", file_index, len(file_index))
        if len(file_index) == 0:
            diff = (date_diffs.total_seconds().values.min() /
                    60) if len(date_diffs) != 0 else float('inf')
            raise FileNotFoundError(
                f"No RAP files within {self.time_range_minutes} minutes of "
                f"{self.date.strftime('%Y-%m-%d %H:%M:%S')}. Nearest file is within {diff:.3f} minutes")
        else:
            filename = rap_files[np.argmin(date_diffs)]
        return filename
    
    def extract_rap_profile(self, center_lon, center_lat, wgrib2):
        if not isinstance(center_lon, str):
            center_lon = str(center_lon)
        if not isinstance(center_lat, str):
            center_lat = str(center_lat)
        values = subprocess.check_output([wgrib2, self.rap_file,
                                          "-match",":[0-9]* hybrid level*", "-s",
                                          "-lon", center_lon, center_lat]).decode('utf-8').split('\n')[:-1]

        pres = np.zeros(50)
        temp = np.zeros(50)
        spec = np.zeros(50)
        height = np.zeros(50)

        lon = float(values[0][values[0].find('lon=') + len('lon='):values[0].find('lat=') - 1])
        lat = float(values[0][values[0].find('lat=') + len('lat='):values[0].find('val=') - 1])

        for line in values:
            items = line.split(':')
            index = int(items[4].split('hybrid level')[0]) - 1
            value = float(items[-1].split('val=')[-1])
            if items[3] == 'PRES':
                pres[index] = value
            if items[3] == 'HGT':
                height[index] = value
            if items[3] == 'TMP':
                temp[index] = value
            if items[3] == 'SPFH':
                spec[index] = value

        return pres, temp, spec, height, lon, lat