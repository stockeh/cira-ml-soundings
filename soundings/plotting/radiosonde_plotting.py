"""Plotting methods for atmospheric sounding"""

import os
import tempfile

import matplotlib
import matplotlib.pyplot as pyplot
import metpy.plots
import metpy.units
import metpy.calc
import numpy
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

from soundings.utils import radiosonde_utils

# matplotlib.use('agg')


MAIN_LINE_COLOUR_KEY = 'main_line_colour'
PREDICTED_LINE_COLOUR_KEY = 'predicted_line_colour'
NWP_LINE_COLOUR_KEY = 'nwp_line_colour'
MAIN_LINE_WIDTH_KEY = 'main_line_width'
DRY_ADIABAT_COLOUR_KEY = 'dry_adiabat_colour'
MOIST_ADIABAT_COLOUR_KEY = 'moist_adiabat_colour'
ISOHUME_COLOUR_KEY = 'isohume_colour'
CONTOUR_LINE_WIDTH_KEY = 'contour_line_width'
GRID_LINE_COLOUR_KEY = 'grid_line_colour'
GRID_LINE_WIDTH_KEY = 'grid_line_width'
FIGURE_WIDTH_KEY = 'figure_width_inches'
FIGURE_HEIGHT_KEY = 'figure_height_inches'
DEFAULT_FONT_SIZE = 'default_font_size'
TITLE_FONT_SIZE = 'title_font_size'
DOTS_PER_INCH = 'dots_per_inch'

DEFAULT_OPTION_DICT = {
    MAIN_LINE_COLOUR_KEY: numpy.array([0, 0, 0], dtype=float),
    PREDICTED_LINE_COLOUR_KEY: numpy.array([44, 114, 230], dtype=float) / 255,
    NWP_LINE_COLOUR_KEY: numpy.array([162, 20, 47], dtype=float) / 255,
    MAIN_LINE_WIDTH_KEY: 4,
    DRY_ADIABAT_COLOUR_KEY: numpy.array([217, 95, 2], dtype=float) / 255,
    MOIST_ADIABAT_COLOUR_KEY: numpy.array([117, 112, 179], dtype=float) / 255,
    ISOHUME_COLOUR_KEY: numpy.array([27, 158, 119], dtype=float) / 255,
    CONTOUR_LINE_WIDTH_KEY: 1,
    GRID_LINE_COLOUR_KEY: numpy.array([152, 152, 152], dtype=float) / 255,
    GRID_LINE_WIDTH_KEY: 2,
    FIGURE_WIDTH_KEY: 8,
    FIGURE_HEIGHT_KEY: 8,
    DEFAULT_FONT_SIZE: 12,
    TITLE_FONT_SIZE: 12,
    DOTS_PER_INCH: 300
}


def _init_save_img(option_dict):    
    option_dict[DEFAULT_FONT_SIZE] = 30
    option_dict[TITLE_FONT_SIZE] = 25
    option_dict[FIGURE_WIDTH_KEY] = 15
    option_dict[FIGURE_HEIGHT_KEY] = 15
    option_dict[MAIN_LINE_WIDTH_KEY] = 5

def colour_from_numpy_to_tuple(input_colour):
    """Converts colour from numpy array to tuple (if necessary).
    :param input_colour: Colour (possibly length-3 or length-4 numpy array).
    :return: output_colour: Colour (possibly length-3 or length-4 tuple).
    """

    if not isinstance(input_colour, numpy.ndarray):
        return input_colour

    return tuple(input_colour.tolist())


def _plot_attributes(skewt_object, option_dict, title_string):
    
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
    
    axes_object.set_xticklabels(labels=x_tick_labels)

    y_tick_labels = [
        '{0:d}'.format(int(numpy.round(y))) for y in axes_object.get_yticks()
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # UserWarning: FixedFormatter should only be used 
        # together with FixedLocator
        axes_object.set_yticklabels(labels=y_tick_labels)
    # TODO: Shouldn't need this hack...
    axes_object.set_xlim(-40, 50)

    if title_string is not None:
        pyplot.title(title_string)


def _init_skewT(option_dict):

    pyplot.rc('font', size=option_dict[DEFAULT_FONT_SIZE])
    pyplot.rc('axes', titlesize=option_dict[TITLE_FONT_SIZE])
    pyplot.rc('axes', labelsize=option_dict[DEFAULT_FONT_SIZE])
    pyplot.rc('xtick', labelsize=option_dict[DEFAULT_FONT_SIZE])
    pyplot.rc('ytick', labelsize=option_dict[DEFAULT_FONT_SIZE])
    pyplot.rc('legend', fontsize=option_dict[DEFAULT_FONT_SIZE])
    pyplot.rc('figure', titlesize=option_dict[TITLE_FONT_SIZE])

    figure_object = pyplot.figure(
        figsize=(option_dict[FIGURE_WIDTH_KEY], option_dict[FIGURE_HEIGHT_KEY])
    )
    skewt_object = metpy.plots.SkewT(figure_object, rotation=45)

    return figure_object, skewt_object


def plot_sounding(sounding_dict, title_string=None, option_dict=None, file_name=None):
    """Plots atmospheric sounding.

    H = number of vertical levels in sounding
    :params
    ---
        sounding_dict : dict
            The following keys: pressures_mb, temperatures_deg_c, dewpoints_deg_c
            wind_speed_kt, wind_dir_deg
        title_string : str
        option_dict : dict
        file_name : str
            Default `None` does not save profile to disk
    :return
    ---
        figure_object : Figure handle
        skewt_object : SkewT handle (skewt_object.ax)
    """
    if option_dict is None:
        option_dict = {}
        orig_option_dict = DEFAULT_OPTION_DICT.copy()
    else:
        orig_option_dict = option_dict.copy()

    option_dict.update(orig_option_dict)

    if file_name:
        _init_save_img(option_dict)
     
    main_line_width = option_dict[MAIN_LINE_WIDTH_KEY]

    figure_object, skewt_object = _init_skewT(option_dict)

    pressure = radiosonde_utils.convert_metpy_pressure(sounding_dict)
    try:
        temperature = radiosonde_utils.convert_metpy_temperature(sounding_dict)
        skewt_object.plot(
            pressure, temperature,
            color=colour_from_numpy_to_tuple(option_dict[MAIN_LINE_COLOUR_KEY]),
            linewidth=main_line_width, linestyle='solid', label='RAOB'
        )
    except KeyError:
        pass
    try:
        dewpoint = radiosonde_utils.convert_metpy_dewpoint(sounding_dict)
        skewt_object.plot(
            pressure, dewpoint,
            color=colour_from_numpy_to_tuple(option_dict[MAIN_LINE_COLOUR_KEY]),
            linewidth=main_line_width, linestyle='solid'
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

    _plot_attributes(skewt_object, option_dict, title_string)

    if file_name:
        pyplot.savefig(file_name, dpi=option_dict[DOTS_PER_INCH])
        pyplot.show()
        pyplot.close()

    return figure_object, skewt_object


def plot_predicted_sounding(sounding_dict, title_string=None, option_dict=None, file_name=None):
    """Plots atmospheric sounding ground truth and prediction.

    H = number of vertical levels in sounding
    :params
    ---
        sounding_dict : dict
            The following keys: pressures_mb, temperatures_deg_c, dewpoints_deg_c
            wind_speed_kt, wind_dir_deg
        title_string : str
        option_dict : dict
        file_name : str
            Default `None` does not save profile to disk
    :return
    ---
        figure_object : Figure handle
        skewt_object : SkewT handle (skewt_object.ax)
    """
    if option_dict is None:
        option_dict = {}
        orig_option_dict = DEFAULT_OPTION_DICT.copy()
    else:
        orig_option_dict = option_dict.copy()

    option_dict.update(orig_option_dict)

    if file_name:
        _init_save_img(option_dict)

    figure_object, skewt_object = plot_sounding(
        sounding_dict, title_string, option_dict)

    pressure = radiosonde_utils.convert_metpy_pressure(sounding_dict)
    predicted_temperatures_deg_c = sounding_dict[radiosonde_utils.PREDICTED_TEMPERATURE_COLUMN_KEY] * \
        metpy.units.units.degC
    
    predicted_line_colour = option_dict[PREDICTED_LINE_COLOUR_KEY]
    main_line_width = option_dict[MAIN_LINE_WIDTH_KEY] / 1.85
    skewt_object.plot(
        pressure, predicted_temperatures_deg_c,
        color=colour_from_numpy_to_tuple(predicted_line_colour),
        linewidth=main_line_width, linestyle='solid', label='ML'
    )

    pyplot.legend(fontsize=option_dict[DEFAULT_FONT_SIZE])
    
    if file_name:
        pyplot.savefig(file_name, dpi=option_dict[DOTS_PER_INCH])
        pyplot.show()
        pyplot.close()

    return figure_object, skewt_object


def plot_nwp_ml_sounding(sounding_dict, title_string=None, option_dict=None, file_name=None):
    """Plots radiosonde, nwp profile, and ML prediction.

    H = number of vertical levels in sounding
    :params
    ---
        sounding_dict : dict
            The following keys: 
            RAOB: pressures_mb, temperatures_deg_c, dewpoints_deg_c
            NWP: nwp_temperatures_deg_c, nwp_dewpoints_deg_c
            ML: predicted_temperature_deg_c, predicted_dewpoints_deg_c
        title_string : str
        option_dict : dict
        file_name : str
            Default `None` does not save profile to disk
    :return
    ---
        figure_object : Figure handle
        skewt_object : SkewT handle (skewt_object.ax)
    """
    if option_dict is None:
        option_dict = {}
        orig_option_dict = DEFAULT_OPTION_DICT.copy()
    else:
        orig_option_dict = option_dict.copy()

    option_dict.update(orig_option_dict)

    if file_name:
        _init_save_img(option_dict)

    figure_object, skewt_object = plot_sounding(
        sounding_dict, title_string, option_dict)
    
    pressure = radiosonde_utils.convert_metpy_pressure(sounding_dict)
    
    nwp_line_colour = option_dict[NWP_LINE_COLOUR_KEY]
    predicted_line_colour = option_dict[PREDICTED_LINE_COLOUR_KEY]
    main_line_width = option_dict[MAIN_LINE_WIDTH_KEY] / 1.4

    try:
        nwp_temperatures_deg_c = sounding_dict[radiosonde_utils.NWP_TEMPERATURE_COLUMN_KEY] * \
            metpy.units.units.degC

        skewt_object.plot(
            pressure, nwp_temperatures_deg_c,
            color=colour_from_numpy_to_tuple(nwp_line_colour),
            linewidth=main_line_width, linestyle='solid', label='RAP'
        )
    except KeyError:
        pass
    
    try: 
        predicted_temperatures_deg_c = sounding_dict[radiosonde_utils.PREDICTED_TEMPERATURE_COLUMN_KEY] * \
            metpy.units.units.degC

        skewt_object.plot(
            pressure, predicted_temperatures_deg_c,
            color=colour_from_numpy_to_tuple(predicted_line_colour),
            linewidth=main_line_width, linestyle='solid', label='ML'
        )
    except KeyError:
        pass
    
    pyplot.legend(fontsize=option_dict[DEFAULT_FONT_SIZE])
    
    try:
        nwp_dewpoint_deg_c = sounding_dict[radiosonde_utils.NWP_DEWPOINT_COLUMN_KEY] * \
            metpy.units.units.degC

        skewt_object.plot(
            pressure, nwp_dewpoint_deg_c,
            color=colour_from_numpy_to_tuple(nwp_line_colour),
            linewidth=main_line_width, linestyle='solid'
        )
    except KeyError:
        pass
    
    try: 
        predicted_dewpoint_deg_c = sounding_dict[radiosonde_utils.PREDICTED_DEWPOINT_COLUMN_KEY] * \
            metpy.units.units.degC

        skewt_object.plot(
            pressure, predicted_dewpoint_deg_c,
            color=colour_from_numpy_to_tuple(predicted_line_colour),
            linewidth=main_line_width, linestyle='solid'
        )
    except KeyError:
        pass
    
    if file_name:
        pyplot.savefig(file_name, dpi=option_dict[DOTS_PER_INCH])
        pyplot.show()
        pyplot.close()

    return figure_object, skewt_object


def plot_monthly_error(monthly_err, months, title_string=None, option_dict=None, file_name=None):
    """Plots the monthly errors

    :params
    ---
        monthly_err : list
            RMSE values of N samples
        months : np.array
        title_string: str
        option_dict : dict
        file_name : str
            Default `None` does not save profile to disk
    :return
    ---
        fig : figure handle
        ax : figure axis
    """
    if option_dict is None:
        option_dict = {}
        orig_option_dict = DEFAULT_OPTION_DICT.copy()
    else:
        orig_option_dict = option_dict.copy()

    option_dict.update(orig_option_dict)
    
    if file_name:
        option_dict[DEFAULT_FONT_SIZE] = 25
        option_dict[TITLE_FONT_SIZE] = 20
        option_dict[FIGURE_WIDTH_KEY] = 15
        option_dict[FIGURE_HEIGHT_KEY] = 12
        markersize = 12
        markeredgewidth = 2
    else:
        option_dict[FIGURE_HEIGHT_KEY] = 6
        option_dict[MAIN_LINE_WIDTH_KEY] = 2
        option_dict[GRID_LINE_WIDTH_KEY] = 1
        markersize = 8
        markeredgewidth = 1
        
    pyplot.rc('font', size=option_dict[DEFAULT_FONT_SIZE])
    pyplot.rc('axes', titlesize=option_dict[TITLE_FONT_SIZE])
    pyplot.rc('axes', labelsize=option_dict[DEFAULT_FONT_SIZE])
    pyplot.rc('xtick', labelsize=option_dict[DEFAULT_FONT_SIZE])
    pyplot.rc('ytick', labelsize=option_dict[DEFAULT_FONT_SIZE])
    pyplot.rc('legend', fontsize=option_dict[DEFAULT_FONT_SIZE])
    pyplot.rc('figure', titlesize=option_dict[TITLE_FONT_SIZE])

    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(option_dict[FIGURE_WIDTH_KEY], 
                                                         option_dict[FIGURE_HEIGHT_KEY]))
    
    ax.boxplot(monthly_err, 
               flierprops=dict(markersize=markersize, markeredgewidth=markeredgewidth), 
               boxprops=dict(facecolor='white', linewidth=option_dict[MAIN_LINE_WIDTH_KEY], ), 
               medianprops=dict(linewidth=option_dict[MAIN_LINE_WIDTH_KEY], color='firebrick'),
               whiskerprops=dict(linewidth=option_dict[MAIN_LINE_WIDTH_KEY]),
               capprops=dict(linewidth=option_dict[MAIN_LINE_WIDTH_KEY]),patch_artist=True)

    ax.yaxis.grid(True, linewidth=option_dict[GRID_LINE_WIDTH_KEY])
    ax.set_xticks(numpy.unique(months))
    
    if title_string:
        ax.set_title(title_string)
    
    ax.set_ylabel('Vertical RMSE');
    
    # TODO: varaible months index into list 
    pyplot.setp(ax, xticks=numpy.unique(months),
                xticklabels=['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                             'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])
    
    if file_name:
        pyplot.savefig(file_name, dpi=option_dict[DOTS_PER_INCH], bbox_inches='tight')
    else:
        fig.tight_layout() 
        
    pyplot.show()
    return fig, ax
