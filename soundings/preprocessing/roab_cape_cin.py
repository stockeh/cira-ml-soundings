import numpy as np
import time
import sys

from tqdm import tqdm

from soundings.experiments import results as results_calc
from soundings.preprocessing import dataloader as dl

def save_results(cape_cin):
    f = '/mnt/data1/stock/mlsoundings/raob_cape_cin.npy'
    np.save(f, cape_cin)
    

def compute_values(raob: np.ndarray):
    cape_cin = np.zeros((len(raob), 2))

    for i in tqdm(range(len(raob))):
        pressure = raob[i, :, dl.PRESSURE]
        temperature = raob[i, :, dl.TEMPERATURE]
        dewpoint = raob[i, :, dl.DEWPOINT]
        
        try:
            cape, cin = results_calc.compute_cape_cin(pressure, temperature, dewpoint)
        except Exception as e:
            cape, cin = np.nan, np.nan
            
        cape_cin[i, 0] = cape
        cape_cin[i, 1] = cin

    return cape_cin


def main():

    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(suppress=True)

    start_t = time.time()
    
    f = '/mnt/data1/stock/mlsoundings/preprocessed.npz'
    container = np.load(f)
    raob = container['raob']
    container.close()
    
    cape_cin = compute_values(raob)
    save_results(cape_cin)
    
    print(f"Finished! Runtime: {time.time()-start_t}")

    
if __name__ == "__main__":
    """
    Usage: python -u -m soundings.preprocessing.roab_cape_cin
    """
    main()