import numpy as np
import warnings
import pygrib

from netCDF4 import Dataset

########################################
#
# convert_rad_tb
# convert_angle_geodetic
# lat_lon_point
# read_goes
# read_rtma
# read_nwp
# read_sonde
#
########################################


def convert_rad_tb(g16nc):
    """
    Converting Spectral Radiance to BT (bands 7-16)
    Planck Function constants are used for the conversion 
    between radiance (mW/(m2·sr·cm- 1)) and BT (K) 
    Equations from https://www.star.nesdis.noaa.gov/goesr/documents/ATBDs/Baseline/ATBD_GOES-R_ABI_CMI_KPP_v3.0_July2012.pdf
    """
    rad_fnc = g16nc.variables['Rad'][:, :]
    # Radiances units: mW m-2 sr-1 (cm-1)-1
    fk1 = g16nc.variables['planck_fk1'][0]
    fk2 = g16nc.variables['planck_fk2'][0]
    bc1 = g16nc.variables['planck_bc1'][0]
    bc2 = g16nc.variables['planck_bc2'][0]

    data = (fk2 / np.log((fk1 / rad_fnc) + 1.0) - bc1) / bc2
    # TODO: check if missing values should be filled or not
    # data = data.filled(-999.)

    return data


def convert_angle_geodetic(g16nc, proj_info):
    """
    Navigating from N/S Elevation Angle (y) and E/W Scanning Angle (x)
    to Geodetic Latitude (φ) and Longitude (λ)
    Equations from https://www.goes-r.gov/users/docs/PUG-L1b-vol3.pdf
    """
    # GOES-R projection info and retrieving relevant constants
    lon_origin = proj_info.longitude_of_projection_origin
    H = proj_info.perspective_point_height+proj_info.semi_major_axis
    r_eq = proj_info.semi_major_axis
    r_pol = proj_info.semi_minor_axis

    # grid info
    lat_rad_1d = g16nc.variables['x'][:]
    lon_rad_1d = g16nc.variables['y'][:]

    # create meshgrid filled with radian angles
    lat_rad, lon_rad = np.meshgrid(lat_rad_1d, lon_rad_1d)

    # lat/lon calc routine from satellite radian angle vectors

    lambda_0 = (lon_origin*np.pi)/180.0

    a_var = np.sin(lat_rad)**2 + (np.cos(lat_rad)**2 * (np.cos(lon_rad)
                                                        ** 2 + (((r_eq**2)/(r_pol**2))*np.sin(lon_rad)**2)))
    b_var = -2.0*H*np.cos(lat_rad)*np.cos(lon_rad)
    c_var = (H**2.0)-(r_eq**2.0)

    # np.any((b_var**2)-(4.0*a_var*c_var) == 0)
    # error due to zero values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_s = (- b_var - np.sqrt((b_var**2)-(4.0*a_var*c_var)))/(2.0*a_var)

    s_x = r_s*np.cos(lat_rad)*np.cos(lon_rad)
    s_y = - r_s*np.sin(lat_rad)
    s_z = r_s*np.cos(lat_rad)*np.sin(lon_rad)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat = (180.0/np.pi) * np.arctan(((r_eq**2)/(r_pol**2))
                                        * (s_z / np.sqrt((H-s_x)**2 + s_y**2)))
    lon = (180.0/np.pi)*(lambda_0 - np.arctan(s_y/(H-s_x)))

    return lat_rad_1d, lon_rad_1d, lat, lon


def lat_lon_point(D, lat=40.5853, lon=-105.0844):
    """
    Extract the pixels (x, y) corresponding latitude/longitude of GOES/NWP/RTMA
    Output:
        x, y: closest indicies corresponding to the lat/lon provided
    """
    # Compute the abosulte difference between the grid lat/lon and the point
    abslat = np.abs(D['lat']-lat)
    abslon = np.abs(D['lon']-lon)

    # Element-wise maxima
    c = np.maximum(abslon, abslat)

    # The index of the minimum maxima (which is the nearest lat/lon)
    x, y = np.where(c == np.min(c))

    return x[0], y[0]


def read_goes(fin):
    """
    Read a single netCDF GOES file.
    Input:
        fin: file input
    Output:
        Dictionary of relevant information
    """
    data = None
    g16nc = Dataset(fin, 'r')

    try:
        band_id = g16nc.variables['band_id'][:]
        band_wavelength = g16nc.variables['band_wavelength']
        band_wavelength_units = band_wavelength.units
        scan_start = g16nc.time_coverage_start
    except:
        band_id = -1
        band_wavelength = -1
        band_wavelength_units = ''
        scan_start = ''

    if (7 <= band_id[0] and band_id[0] <= 16):
        data = convert_rad_tb(g16nc)

    proj_info = g16nc.variables['goes_imager_projection']

    lat_rad_1d, lon_rad_1d, lat, lon = convert_angle_geodetic(g16nc, proj_info)

    central_lon = proj_info.longitude_of_projection_origin

    h = proj_info.perspective_point_height
    x = lat_rad_1d * h
    y = lon_rad_1d * h

    g16nc.close()
    return {'data': data,
            'band_id': band_id,
            'scan_start': scan_start,
            'central_lon': central_lon,
            'h': h, 'x': x, 'y': y,
            'lat': lat, 'lon': lon}


def read_rtma(fin):
    grbs = pygrib.open(fin)

    for i, grb in enumerate(grbs[:]):
        # Actual data index, but not always the first value
        if grb.typeOfGeneratingProcess == 0:
            idx = i + 1
            break

    value, lat, lon = grbs[idx].data()
    level = grbs[idx].level
    level_units = grbs[idx].typeOfLevel
    validDATE = grbs[idx].validDate
    name = grbs[idx].parameterName

    grbs.close()
    return {'data': value, 'lat': lat, 'lon': lon,
            'level': level, 'level_units': level_units,
            'valid': validDATE, 'name': name}


def read_nwp(fin):
    grbs = pygrib.open(fin)
    Ds = []
    lat, lon, validDATE = None, None, None
    for i, grb in enumerate(grbs[:]):

        name = grb.parameterName

        if name in ['Temperature', 'Dew point temperature']:
            level = grb.level
            if level >= 100 and level <= 1000:
                if not any((lat, lon, validDATE)):
                    value, lat, lon = grb.data()
                    validDATE = grb.validDate
                else:
                    value, _, _ = grb.data()
                level_units = grb.typeOfLevel

                Ds.append({'data': value, 'lat': lat, 'lon': lon,
                           'level': level, 'level_units': level_units,
                           'valid': validDATE, 'name': name})

    grbs.close()
    return Ds


def read_sonde(fin):
    """
    Read in a .dat FSL Rawinsonde data file/
    input:
        fin: file input name
    output:
        [{'metadata' {...}, 'data': {...}}, ...]
    """
    cols = {254: (('HOUR', 7, 14), ('DAY', 14, 21), ('MONTH', 21, 30), ('YEAR', 30, 38)),
            1: (('WBAN', 7, 14), ('WMO', 14, 21), ('LAT', 21, 29),
                ('LON', 29, 36), ('ELEV', 36, 42), ('RTIME', 42, 49)),
            2: (('HYDRO', 7, 14), ('MXWD', 14, 21), ('TROPL', 21, 28),
                ('LINES', 28, 35), ('TINDEX', 35, 42), ('SOURCE', 42, 49)),
            3: (('STAID', 17, 21), ('SONDE', 37, 42), ('WSUNITS', 42, 49))}

    dflt = (('PRESSURE', 7, 14), ('HEIGHT', 14, 21), ('TEMP', 21, 28),
            ('DEWPT', 28, 35), ('WINDDIR', 35, 42), ('WINDSPD', 42, 49))

    experiments = []

    with open(fin, 'r') as f:
        def build_dict(LINTYP, dictionary):
            for name, beg, end in cols.get(LINTYP, dflt):
                value = line[beg:end].strip()
                # remove missing values and those below 100 hPa
                # convert tenths to whole units
                if name in ['TEMP', 'DEWPT']:
                    value = float(value) / 10.
                    if value == 9999.9:
                        return False
                elif name in ['PRESSURE']:
                    value = float(value) / 10.
                    if value < 100:
                        return False
                elif name in ['WINDSPD', 'WINDDIR']:
                    value = float(value)
                    if value == 99999:
                        value = 0
                elif name in ['LAT', 'LON']:
                    value = float(value[:-1]) * \
                        (1 if value[-1] in ['N', 'E'] else -1)

                dictionary[name] = value
            return True

        for i, line in enumerate(f):
            LINTYP = int(line[:7])

            if LINTYP == 254:
                if i != 0:
                    experiments.append({'metadata': metadata, 'data': data})
                metadata = {}
                data = []
                build_dict(LINTYP, metadata)

            elif LINTYP in [1, 2, 3]:
                build_dict(LINTYP, metadata)

            else:  # 4,5,6,7,8,9
                dictionary = {}
                dictionary['LINTYP'] = LINTYP
                if build_dict(LINTYP, dictionary):
                    data.append(dictionary)

        experiments.append({'metadata': metadata, 'data': data})

    return experiments
