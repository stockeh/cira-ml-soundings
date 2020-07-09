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


def plot_sounding(
        sounding_dict_for_metpy, font_size=DEFAULT_FONT_SIZE, title_string=None,
        option_dict=None):
    """Plots atmospheric sounding.

    H = number of vertical levels in sounding

    :param sounding_dict_for_metpy: Dictionary with the following keys.
    sounding_dict_for_metpy['pressures_mb']: length-H numpy array of pressures
        (millibars).
    sounding_dict_for_metpy['temperatures_deg_c']: length-H numpy array of
        temperatures.
    sounding_dict_for_metpy['dewpoints_deg_c']: length-H numpy array of
        dewpoints.
    sounding_dict_for_metpy['wind_speed_kt']: length-H numpy array of wind
        components (nautical miles per hour, or "knots").
    sounding_dict_for_metpy['wind_dir_deg']: length-H numpy array of directional
        wind in degrees.

    :param font_size: Font size.
    :param title_string: Title.
    :param option_dict: Dictionary with the following keys.
    option_dict['main_line_colour']: Colour for temperature and dewpoint lines
        (in any format accepted by matplotlib).
    option_dict['main_line_width']: Width for temperature and dewpoint lines.
    option_dict['dry_adiabat_colour']: Colour for dry adiabats.
    option_dict['moist_adiabat_colour']: Colour for moist adiabats.
    option_dict['isohume_colour']: Colour for isohumes (lines of constant mixing
        ratio).
    option_dict['contour_line_width']: Width for adiabats and isohumes.
    option_dict['grid_line_colour']: Colour for grid lines (temperature and
        pressure contours).
    option_dict['grid_line_width']: Width for grid lines.
    option_dict['figure_width_inches']: Figure width.
    option_dict['figure_height_inches']: Figure height.

    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """
    if option_dict is None:
        orig_option_dict = {}
    else:
        orig_option_dict = option_dict.copy()

    option_dict = DEFAULT_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    main_line_colour = option_dict[MAIN_LINE_COLOUR_KEY]
    main_line_width = option_dict[MAIN_LINE_WIDTH_KEY]
    dry_adiabat_colour = option_dict[DRY_ADIABAT_COLOUR_KEY]
    moist_adiabat_colour = option_dict[MOIST_ADIABAT_COLOUR_KEY]
    isohume_colour = option_dict[ISOHUME_COLOUR_KEY]
    contour_line_width = option_dict[CONTOUR_LINE_WIDTH_KEY]
    grid_line_colour = option_dict[GRID_LINE_COLOUR_KEY]
    grid_line_width = option_dict[GRID_LINE_WIDTH_KEY]
    figure_width_inches = option_dict[FIGURE_WIDTH_KEY]
    figure_height_inches = option_dict[FIGURE_HEIGHT_KEY]

    figure_object = pyplot.figure(
        figsize=(figure_width_inches, figure_height_inches)
    )
    skewt_object = metpy.plots.SkewT(figure_object, rotation=45)

    pressures_mb = sounding_dict_for_metpy[
        radiosonde_utils.PRESSURE_COLUMN_KEY] * metpy.units.units.hPa
    temperatures_deg_c = sounding_dict_for_metpy[
        radiosonde_utils.TEMPERATURE_COLUMN_KEY] * metpy.units.units.degC
    dewpoints_deg_c = sounding_dict_for_metpy[
        radiosonde_utils.DEWPOINT_COLUMN_KEY] * metpy.units.units.degC

    skewt_object.plot(
        pressures_mb, temperatures_deg_c,
        color=colour_from_numpy_to_tuple(main_line_colour),
        linewidth=main_line_width, linestyle='solid'
    )

    skewt_object.plot(
        pressures_mb, dewpoints_deg_c,
        color=colour_from_numpy_to_tuple(main_line_colour),
        linewidth=main_line_width, linestyle='dashed'
    )

    try:
        wind_speed = sounding_dict_for_metpy[
            radiosonde_utils.WIND_SPEED_COLUMN_KEY] * metpy.units.units.knots
        wind_dir = sounding_dict_for_metpy[
            radiosonde_utils.WIND_DIR_COLUMN_KEY] * metpy.units.units.degrees
        u_winds_kt, v_winds_kt = metpy.calc.wind_components(
            wind_speed, wind_dir)
        skewt_object.plot_barbs(pressures_mb, u_winds_kt, v_winds_kt)
    except KeyError:
        pass

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

    # TODO(thunderhoser): Shouldn't need this hack.
    axes_object.set_xlim(-40, 50)

    if title_string is not None:
        pyplot.title(title_string, fontsize=font_size)

    # pyplot.savefig('OKAY', dpi=DOTS_PER_INCH)
    # pyplot.close()

    return figure_object, axes_object
