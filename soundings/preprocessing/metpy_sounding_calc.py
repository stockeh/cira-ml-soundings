import numpy as np
import metpy.calc
import metpy.units
import time
import argparse
import sys

from tqdm import tqdm

from soundings.experiments import results as results_calc
from soundings.preprocessing import dataloader as dl

def save_results(dataset, product, results):
    f = f'/mnt/data1/stock/mlsoundings/{dataset}_{product}.npy'
    np.save(f, results)
    

def retrieve_relative_humidity(r):
    relative_humidity = np.zeros((r.shape[0],r.shape[1]))
    for i in tqdm(range(len(r))):
        temperature = r[i, :, dl.TEMPERATURE] * metpy.units.units.degC
        dewpoint = r[i, :, dl.DEWPOINT] * metpy.units.units.degC
        for level in range(temperature.shape[0]):
            t = temperature[level]
            td = dewpoint[level]
            try:
                rh = metpy.calc.relative_humidity_from_dewpoint(t, td).magnitude
            except Exception as e:
                rh = np.nan
            relative_humidity[i, level] = rh
            
    return relative_humidity
    
    
def retrieve_parcel_profile(r):
    parcels = np.zeros((len(r), r.shape[1]))
    for i in tqdm(range(len(r))):
        pressure = r[i, :, dl.PRESSURE] * metpy.units.units.hPa
        temperature = r[i, 0, dl.TEMPERATURE] * metpy.units.units.degC
        dewpoint = r[i, 0, dl.DEWPOINT] * metpy.units.units.degC

        try:
            p = metpy.calc.parcel_profile(pressure, temperature, dewpoint).to(metpy.units.units.degC).magnitude
        except Exception as e:
            p = np.nan
            
        parcels[i] = p
        
    return parcels

    
def retrieve_cape_cin(r: np.ndarray):
    cape_cin = np.zeros((len(r), 2))

    for i in tqdm(range(len(r))):
        pressure = r[i, :, dl.PRESSURE]
        temperature = r[i, :, dl.TEMPERATURE]
        dewpoint = r[i, :, dl.DEWPOINT]
        
        try:
            cape, cin = results_calc.compute_cape_cin(pressure, temperature, dewpoint)
        except Exception as e:
            cape, cin = np.nan, np.nan
            
        cape_cin[i, 0] = cape
        cape_cin[i, 1] = cin

    return cape_cin


def main(dataset, product):

    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(suppress=True)

    start_t = time.time()
    
    f = '/mnt/data1/stock/mlsoundings/preprocessed.npz'
    container = np.load(f)
    r = container[dataset]
    container.close()
    
    if product == 'cape_cin':
        results = retrieve_cape_cin(r)
    elif product == 'parcel_profile':
        results = retrieve_parcel_profile(r)
    elif product == 'relative_humidity':
        results = retrieve_relative_humidity(r)
    else:
        print('ERROR: Invalid product requested!')
        sys.exit(1)
    
    save_results(dataset, product, results)
    
    print(f"Finished! Runtime: {time.time()-start_t}")

    
if __name__ == "__main__":
    """
    Usage: python -u -m soundings.preprocessing.metpy_sounding_calc -d rap -p cape_cin
    """
    parser = argparse.ArgumentParser(description='preprocessing configuration')
    parser.add_argument('-d', '--dataset', metavar='dataset', type=str,
                        required=True, help='the dataset to compute products on')
    parser.add_argument('-p', '--product', metavar='product', type=str,
                    required=True, help='the product to compute [cape_cin, parcel_profile, relative_humidity]')
    args = parser.parse_args()
    main(dataset=args.dataset, product=args.product)
