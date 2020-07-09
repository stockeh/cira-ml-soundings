"""Plotting methods for atmospheric sounding"""

import os
import tempfile

import matplotlib
import matplotlib.pyplot as pyplot
import metpy.plots
import metpy.units
import metpy.calc
import numpy

from soundings.utils import radiosonde_utils

# matplotlib.use('agg')


MAIN_LINE_COLOUR_KEY = 'main_line_colour'
PREDICTED_LINE_COLOUR_KEY = 'predicted_line_colour'
MAIN_LINE_WIDTH_KEY = 'main_line_width'
DRY_ADIABAT_COLOUR_KEY = 'dry_adiabat_colour'
MOIST_ADIABAT_COLOUR_KEY = 'moist_adiabat_colour'
ISOHUME_COLOUR_KEY = 'isohume_colour'
CONTOUR_LINE_WIDTH_KEY = 'contour_line_width'
GRID_LINE_COLOUR_KEY = 'grid_line_colour'
GRID_LINE_WIDTH_KEY = 'grid_line_width'
FIGURE_WIDTH_KEY = 'figure_width_inches'
FIGURE_HEIGHT_KEY = 'figure_height_inches'

DEFAULT_OPTION_DICT = {
    MAIN_LINE_COLOUR_KEY: numpy.array([0, 0, 0], dtype=float),
    PREDICTED_LINE_COLOUR_KEY: numpy.array([44, 114, 230], dtype=float) / 255,
    MAIN_LINE_WIDTH_KEY: 3,
    DRY_ADIABAT_COLOUR_KEY: numpy.array([217, 95, 2], dtype=float) / 255,
    MOIST_ADIABAT_COLOUR_KEY: numpy.array([117, 112, 179], dtype=float) / 255,
    ISOHUME_COLOUR_KEY: numpy.array([27, 158, 119], dtype=float) / 255,
    CONTOUR_LINE_WIDTH_KEY: 1,
    GRID_LINE_COLOUR_KEY: numpy.array([152, 152, 152], dtype=float) / 255,
    GRID_LINE_WIDTH_KEY: 2,
    FIGURE_WIDTH_KEY: 8,  # 15,
    FIGURE_HEIGHT_KEY: 8  # 15
}

DEFAULT_FONT_SIZE = 12  # 30
TITLE_FONT_SIZE = 12  # 25

# pyplot.rc('font', size=DEFAULT_FONT_SIZE)
# pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
# pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
# pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
# pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
# pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
# pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

DOTS_PER_INCH = 300


def colour_from_numpy_to_tuple(input_colour):
    """Converts colour from numpy array to tuple (if necessary).
    :param input_colour: Colour (possibly length-3 or length-4 numpy array).
    :return: output_colour: Colour (possibly length-3 or length-4 tuple).
    """

    if not isinstance(input_colour, numpy.ndarray):
        return input_colour

    return tuple(input_colour.tolist())


def _plot_attributes(skewt_object, option_dict, font_size, title_string):
    dry_adiabat_colour = option_dict[DRY_ADIABAT_COLOUR_KEY]
    moist_adiabat_colour = option_dict[MOIST_ADIABAT_COLOUR_KEY]
    isohume_colour = option_dict[ISOHUME_COLOUR_KEY]
    contour_line_width = option_dict[CONTOUR_LINE_WIDTH_KEY]
    grid_line_colour = option_dict[GRID_LINE_COLOUR_KEY]
    grid_line_width = option_dict[GRID_LINE_WIDTH_KEY]

    axes_object = skewt_object.ax

    axes_object.grid(
        color=colour_from_numpy_to_tuple(grid_line_colour),
        linewidth=grid_line_width, linestyle='dashed'
    )

    skewt_object.plot_dry_adiabats(
        color=colour_from_numpy_to_tuple(dry_adiabat_colour),
        linewidth=contour_line_width, linestyle='solid', alpha=1.
    )
    skewt_object.plot_moist_adiabats(
        color=colour_from_numpy_to_tuple(moist_adiabat_colour),
        linewidth=contour_line_width, linestyle='solid', alpha=1.
    )
    skewt_object.plot_mixing_lines(
        color=colour_from_numpy_to_tuple(isohume_colour),
        linewidth=contour_line_width, linestyle='solid', alpha=1.
    )

    axes_object.set_ylim(1000, 100)
    axes_object.set_xlim(-40, 50)
    axes_object.set_xlabel('')
    axes_object.set_ylabel('')

    tick_values_deg_c = numpy.linspace(-40, 50, num=10)
    axes_object.set_xticks(tick_values_deg_c)

    x_tick_labels = [
        '{0:d}'.format(int(numpy.round(x))) for x in axes_object.get_xticks()
    ]
    axes_object.set_xticklabels(x_tick_labels, fontsize=font_size)

    y_tick_labels = [
        '{0:d}'.format(int(numpy.round(y))) for y in axes_object.get_yticks()
    ]
    axes_object.set_yticklabels(y_tick_labels, fontsize=font_size)

    if title_string is not None:
        pyplot.title(title_string, fontsize=font_size)


def _init_skewT(option_dict):
    figure_width_inches = option_dict[FIGURE_WIDTH_KEY]
    figure_height_inches = option_dict[FIGURE_HEIGHT_KEY]

    figure_object = pyplot.figure(
        figsize=(figure_width_inches, figure_height_inches)
    )
    skewt_object = metpy.plots.SkewT(figure_object, rotation=45)

    return figure_object, skewt_object


def plot_sounding(
        sounding_dict, font_size=DEFAULT_FONT_SIZE, title_string=None,
        option_dict=None):
    """Plots atmospheric sounding.

    H = number of vertical levels in sounding
    :params
    ---
        sounding_dict : dict
            The following keys: pressures_mb, temperatures_deg_c, dewpoints_deg_c
            wind_speed_kt, wind_dir_deg
        font_size : int
        title_string : str
        option_dict : dict

    :return
    ---
        figure_object : Figure handle
        skewt_object : SkewT handle (skewt_object.ax)
    """
    if option_dict is None:
        orig_option_dict = {}
    else:
        orig_option_dict = option_dict.copy()

    option_dict = DEFAULT_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    main_line_colour = option_dict[MAIN_LINE_COLOUR_KEY]
    main_line_width = option_dict[MAIN_LINE_WIDTH_KEY]

    figure_object, skewt_object = _init_skewT(option_dict)

    pressure = radiosonde_utils.convert_metpy_pressure(sounding_dict)
    try:
        temperature = radiosonde_utils.convert_metpy_temperature(sounding_dict)
        skewt_object.plot(
            pressure, temperature,
            color=colour_from_numpy_to_tuple(main_line_colour),
            linewidth=main_line_width, linestyle='solid'
        )
    except KeyError:
        pass

    try:
        dewpoint = radiosonde_utils.convert_metpy_dewpoint(sounding_dict)
        skewt_object.plot(
            pressure, dewpoint,
            color=colour_from_numpy_to_tuple(main_line_colour),
            linewidth=main_line_width, linestyle='dashed'
        )
    except KeyError:
        pass

    try:
        wind_speed, wind_dir = radiosonde_utils.convert_metpy_winds(
            sounding_dict)
        u_winds_kt, v_winds_kt = metpy.calc.wind_components(
            wind_speed, wind_dir)
        skewt_object.plot_barbs(pressure, u_winds_kt, v_winds_kt)
    except KeyError:
        pass

    _plot_attributes(skewt_object, option_dict, font_size, title_string)

    # pyplot.savefig('OKAY', dpi=DOTS_PER_INCH)
    # pyplot.close()

    return figure_object, skewt_object


def plot_predicted_sounding(sounding_dict, font_size=DEFAULT_FONT_SIZE,
                            title_string=None, option_dict=None):
    if option_dict is None:
        orig_option_dict = {}
    else:
        orig_option_dict = option_dict.copy()

    option_dict = DEFAULT_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    predicted_line_colour = option_dict[PREDICTED_LINE_COLOUR_KEY]
    main_line_width = option_dict[MAIN_LINE_WIDTH_KEY] / 2.0

    figure_object, skewt_object = plot_sounding(
        sounding_dict, font_size, title_string, option_dict)

    pressure = radiosonde_utils.convert_metpy_pressure(sounding_dict)
    predicted_temperatures_deg_c = sounding_dict[radiosonde_utils.PREDICTED_TEMPERATURE_COLUMN_KEY] * \
        metpy.units.units.degC

    skewt_object.plot(
        pressure, predicted_temperatures_deg_c,
        color=colour_from_numpy_to_tuple(predicted_line_colour),
        linewidth=main_line_width, linestyle='solid'
    )

    pyplot.legend(('T', 'Y'), fontsize=font_size)

    return figure_object, skewt_object
