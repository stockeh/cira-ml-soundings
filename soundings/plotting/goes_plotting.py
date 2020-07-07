import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from soundings.preprocessing import goesimager as goes16im
from soundings.utilities import plotting_utils


def plot_goes_region(D, projection='geos', extent=None):
    """
    Display a single GOES image for one channel.
    Input:
        D: Data dictionary 
        projection: project using Geostationary, Mercator or Lambert Conformal
        extent: region to focus on as using Plate Carree lat/lon coordinates
    """
    geos = ccrs.Geostationary(central_longitude=D['central_lon'],
                              satellite_height=D['h'])

    fig = plt.figure(figsize=(12, 6))

    try:
        proj = [geos,
                ccrs.Mercator(),
                ccrs.LambertConformal()][[
                    'geos', 'mercator', 'lambert'].index(projection)]
    except:
        raise Exception(
            "not a valid projection")

    # create axis with projection
    ax = fig.add_subplot(111, projection=proj)

    im = ax.imshow(D['data'], origin='upper', cmap=plotting_utils.PLOT_CMAP,
                   extent=(D['x'].min(), D['x'].max(),
                           D['y'].min(), D['y'].max()),
                   transform=geos, interpolation='none')

    ax.coastlines(resolution='50m', color='black', linewidth=0.25)
    ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.25)
    ax.set_title(f"GOES-16 {D['band_id']}", loc='left')
    ax.set_title(f"{D['scan_start']}", loc='right', fontweight='bold')

    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    fig.colorbar(im)

    plt.show()


def plot_goes_patch(vol):
    date = 1561417260.0
    goes16 = goes16im.GOES16ABI(vol + 'goes', date)

    center_lon, center_lat = -107.9277, 22.9908
    x_size_pixels, y_size_pixels = 500, 500
    patch, lons, lats = goes16.extract_image_patch(
        center_lon, center_lat, x_size_pixels, y_size_pixels, bt=True)

    index = 0
    vmin = patch[index, :, :, :].min()  # set min/max plotting range
    vmax = patch[index, :, :, :].max()

    fig, axs = plt.subplots(2, 4, figsize=(9, 7))
    [axi.set_axis_off() for axi in axs.ravel()]
    r = 0
    for i in range(8):
        if i >= 4:
            r = 1
        axs[r, i % 4].imshow(patch[index, i, :, :],
                             vmin=vmin, vmax=vmax, cmap=plotting_utils.PLOT_CMAP)
        plt.suptitle('Exmaple Patch ', fontsize=20)
        axs[r, i % 4].set_title(f'ch, {str(goes16.bands[i])}')
    fig.tight_layout()
