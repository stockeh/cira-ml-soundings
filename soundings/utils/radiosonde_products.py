import metpy.calc
import metpy.units
import numpy as np

# pylint: disable=import-error
from soundings.utils import radiosonde_utils

"""
1) Dewpoint temperature ≤ temperature, 
2) temperature does not decrease faster than the adiabatic lapse rate (g/cp) above 10 m, 
3) At the tropopause the temperature is nearly constant with height. 

4) z(0) - z(10m) super adiabatic lapse rate, the lower levels are hard to capture with NWP

Additionally, two derived parameters will be included: 

1) the convective inhibition (CIN), and 
2) the convective available potential energy (CAPE). 

We will start with these two values as features related to the lowest 2 km that we want to get correct.
"""


def dewpt_temp_regularization(rho, temperature, dewpoint):
    """Penalize by a the sum of `rho` for all dewpoint > temperature 

    TODO: is this redunant to the fact that MSE will enforce 
        T and Td errors to be minimized?
    """
    np.aran
    return sum(rho for _ in np.where(dewpoint > temperature)[0])


def assert_dewpt_lte_temp(temperature, dewpoint):
    return all(dewpoint <= temperature)


def surface_based_cape_cin(sounding_dict):
    """Calculate surface-based CAPE and CIN.

    :params
    ---
    sounding_dict : dict
        Dictionary with the following keys.
        radiosonde_utils.PRESSURE_COLUMN_KEY: numpy array of pressures (hPa).
        radiosonde_utils.TEMPERATURE_COLUMN_KEY: numpy array of temperatures.
        radiosonde_utils.DEWPOINT_COLUMN_KEY: numpy array of dewpoints.

    :returns
    ---
    cape : float
        Surface based Convective Available Potential Energy (CAPE).  
    cin : float
        Surface based Convective Inhibition (CIN).
    """
    pressure, temperature, dewpoint = radiosonde_utils.convert_units(
        sounding_dict)

    cape, cin = metpy.calc.thermo.surface_based_cape_cin(
        pressure, temperature, dewpoint)

    return cape.m, cin.m


def most_unstable_cape_cin(sounding_dict):
    """Calculate most unstable CAPE/CIN.

    :params
    ---
    sounding_dict : dict
        Dictionary with the following keys.
        radiosonde_utils.PRESSURE_COLUMN_KEY: numpy array of pressures (hPa).
        radiosonde_utils.TEMPERATURE_COLUMN_KEY: numpy array of temperatures.
        radiosonde_utils.DEWPOINT_COLUMN_KEY: numpy array of dewpoints.

    :returns
    ---
    cape : float
        Surface based Convective Available Potential Energy (CAPE).  
    cin : float
        Surface based Convective Inhibition (CIN).
    """
    pressure, temperature, dewpoint = radiosonde_utils.convert_units(
        sounding_dict)

    cape, cin = metpy.calc.thermo.most_unstable_cape_cin(
        pressure, temperature, dewpoint)

    return cape.m, cin.m


def el(sounding_dict):
    pressure, temperature, dewpoint = radiosonde_utils.convert_units(
        sounding_dict)

    el_pressure, el_temperature = metpy.calc.thermo.el(
        pressure, temperature, dewpoint)

    return el_pressure.m, el_temperature.m
