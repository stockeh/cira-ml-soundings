import numpy as np
import xarray as xr

from glob import glob
from os.path import join
from scipy import interpolate

from IPython.display import display
from ipywidgets import FloatProgress

# TODO: Should this be done in preprocessing?

def interpolate_to_height_intervals(alt, y, altitude_intervals):
    f = interpolate.interp1d(alt, y)
    return f(altitude_intervals)


def load_samples(processed_vol):
    """
    Load the processed data files to np arrays.
    
    :params
    ---
    processed_vol : str
        Path to the /processed directory
    :returns
    ---
    raob, rap, goes, rtma, sonde_files
        Respective datafiles are in p, t, td, alt values
    """
    files = sorted(glob(join(processed_vol, '*')))
    print(f'total of {len(files)} samples!')
    
    fp = FloatProgress(min=0, max=(len(files)))
    display(fp)
    
    sonde_files = []
    raob = []
    rap = []
    goes = []
    rtma = []
    
    s = time.time()
    for i, f in enumerate(files):
        xar = xr.open_dataset(f)
        raob_profile = np.concatenate((xar.sonde_pres.values.reshape(-1,1),
                                       xar.sonde_tdry.values.reshape(-1,1),
                                       xar.sonde_dp.values.reshape(-1,1),
                                       xar.sonde_alt.values.reshape(-1,1)), axis=1)
        raob.append(raob_profile)
        
        alt = xar.nwp_alt.values
        altitude_intervals = np.linspace(alt[0], 18_000, 256)
        
        p = xar.nwp_pres.values
        t = xar.nwp_tdry.values-272.15 # convert to deg C
        q = xar.nwp_spfm.values
        
        pres = interpolate_to_height_intervals(alt, p/100., altitude_intervals)  # convert Pa to hPa
        tdry = interpolate_to_height_intervals(alt, t, altitude_intervals)
        
        epsilon = 0.622
        A = 17.625
        B = 243.04 # deg C
        C = 610.94 # Pa
        
        # vapor pressure
        e = p*q / (epsilon + (1 - epsilon)*q)
        
        if e[0] == 0: # replace first value with eps if zero
            e[0] = np.finfo(float).eps
        if e.all() == 0: # forward fill values where zero exist
            prev = np.arange(len(e))
            prev[e == 0] = 0
            prev = np.maximum.accumulate(prev)
            e = e[prev]
        # dewpoint temperature 
        td = B * np.log(e/C) / (A - np.log(e/C))

        td = interpolate_to_height_intervals(alt, td, altitude_intervals)
        
        rap_profile = np.concatenate((pres.reshape(-1,1),
                                      tdry.reshape(-1,1),
                                      td.reshape(-1,1),
                                      altitude_intervals.reshape(-1,1)), axis=1)
        rap.append(rap_profile)
        goes.append(xar.goes_abi.values)
        rtma.append(xar.rtma_values.values)
        sonde_files.append(str(xar.sonde_file.values))
        xar.close()
        fp.value += 1

    e = time.time() - s
    print(f'time: {e:.3f}, avg: {e/len(files):.3f} seconds')
    
    return (np.array(raob), np.array(rap), np.array(goes).transpose(0, 2, 3, 1),
            np.array(rtma).transpose(0, 2, 3, 1), sonde_files)