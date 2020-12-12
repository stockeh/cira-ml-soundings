import numpy as np
from soundings.preprocessing import goescloudmask

def main():
    f = '/mnt/data1/stock/mlsoundings/preprocessed_alley.npz'
    bcm_path = '/mnt/hilburnnas1/goes16/ABI/RadC/ACM/'

    container = np.load(f)
    sonde_files = container['sonde_files']
    container.close()
    patches = goescloudmask.extract_bcm_patches(sonde_files, bcm_path,
                patch_x_length_pixels=128, patch_y_length_pixels=128,
                time_range_minutes=60)
    
if __name__ == "__main__":
    """python -u -m soundings.preprocessing.scripts.bcm_from_processed_sondes"""
    main()
