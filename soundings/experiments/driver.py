import numpy as np
import pandas as pd
import argparse
import itertools
import time
import yaml
import sys
import os

from soundings.preprocessing import dataloader as dl
from soundings.deep_learning import mlutilities as ml
from soundings.deep_learning import tf_neuralnetwork as nn
from soundings.experiments import experiment_interface as ei

def save_results(config, results, driver):
    results.to_csv(config['results_file'], index=False)
    print('INFO: Results Saved!')

          
def experiment(config, network_name, data):
    Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = data
    if network_name in ['NeuralNetwork']:
        driver = ei.NeuralNetworkDriver()
    elif network_name in ['Convolutional']:
        pass
    else:
        raise ValueError(f'{network_name} not a valid type.')
    
    experiments = driver.get_experiemnts(config)
    results = driver.run_experiments(config, experiments, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)
    
    return results, driver


def load_data(config, network_name):
    container = np.load(config['data']['saved_f'])
    raob = container['raob']
    rap  = container['rap']
    goes = container['goes']
    rtma = container['rtma']
    sonde_files = container['sonde_files']
    print(f'INFO: total data shape -- {raob.shape}, {rap.shape}, {goes.shape}, {rtma.shape}')
    
    if network_name not in ['MultiNeuralNetwork', 'MultiSkipNetwork']:
        Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = ml.partition(rap, raob, (0.75, 0.10, 0.15),
                                                                shuffle=True, seed=1234)
    else:
        raise ValueError('Need to support GOES & RTMA partitioning')
        
    return (Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)


def main(config_path):
    global config

    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(suppress=True)

    start_t = time.time()
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)
            
    network_name = config['model']['network']
    try:   
        data = load_data(config, network_name)
        results, driver = experiment(config, network_name, data)
        print('INFO: finished experiments')
        save_results(config, results, driver)
    except Exception as e:
        print('ERROR:', e)
        sys.exit(1)
        
    print(f"Finished! Runtime: {time.time()-start_t}")

    
if __name__ == "__main__":
    """
    Usage: python -u -m soundings.experiments.driver -c ./soundings/experiments/__config__.yaml
    """
    parser = argparse.ArgumentParser(description='experimental configuration')
    parser.add_argument('-c', '--config', metavar='path', type=str,
                        required=True, help='the path to config file')
    args = parser.parse_args()
    main(config_path=args.config)