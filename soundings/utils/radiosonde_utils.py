import metpy.units

PRESSURE_COLUMN_KEY = 'pressures_mb'
TEMPERATURE_COLUMN_KEY = 'temperatures_deg_c'
DEWPOINT_COLUMN_KEY = 'dewpoints_deg_c'
WIND_SPEED_COLUMN_KEY = 'wind_speed_kt'
WIND_DIR_COLUMN_KEY = 'wind_dir_deg'

PREDICTED_TEMPERATURE_COLUMN_KEY = 'predicted_temperatures_deg_c'
PREDICTED_DEWPOINT_COLUMN_KEY = 'predicted_dewpoints_deg_c'


def convert_metpy_winds(sounding_dict):
    """Convert windspeed and wind direction to MetPy units

    :params
    ---
    sounding_dict : dict
        WIND_SPEED_COLUMN_KEY: numpy array of wind speeds.
        WIND_DIR_COLUMN_KEY: numpy array of wind dir.

    :returns
    ---
    wind_speed : array
    wind_dir : array
    """

    wind_speed = sounding_dict[WIND_SPEED_COLUMN_KEY] * \
        metpy.units.units.knots
    wind_dir = sounding_dict[WIND_DIR_COLUMN_KEY] * \
        metpy.units.units.degrees

    return wind_speed, wind_dir


def convert_metpy_pressure(sounding_dict):
    return sounding_dict[PRESSURE_COLUMN_KEY] * \
        metpy.units.units.hPa


def convert_metpy_temperature(sounding_dict):
    return sounding_dict[TEMPERATURE_COLUMN_KEY] * \
        metpy.units.units.degC


def convert_metpy_dewpoint(sounding_dict):
    return sounding_dict[DEWPOINT_COLUMN_KEY] * \
        metpy.units.units.degC


def convert_metpy_profile(sounding_dict):
    """Convert pressure, temperature, and dewpoint to MetPy units

    :params
    ---
    sounding_dict : dict
        Dictionary with the following keys:
        PRESSURE_COLUMN_KEY: numpy array of pressures (hPa).
        TEMPERATURE_COLUMN_KEY: numpy array of temperatures.
        DEWPOINT_COLUMN_KEY: numpy array of dewpoints.

    :returns
    ---
    pressure : array
    temperature : array
    dewpoint : array
    """

    pressure = convert_metpy_pressure(sounding_dict)
    temperature = convert_metpy_temperature(sounding_dict)
    dewpoint = convert_metpy_dewpoint(sounding_dict)

    return pressure, temperature, dewpoint
