import tensorflow as tf
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

from soundings.experiments.neuralnetwork_driver import NeuralNetworkDriver
from soundings.experiments.cnn_skip_neuralnetwork_driver import CNNSkipNetworkDriver

def save_results(config: dict, results, driver):
    results.to_csv(config['results_file'], index=False)
    print('INFO: Results Saved!')

          
def experiment(config: dict, network_name: str, data: tuple):
    if network_name in ['NeuralNetwork']:
        driver = NeuralNetworkDriver()
    elif network_name in ['MultiConvolutionalNeuralNetwork', 'SkipNeuralNetwork']:
        driver = CNNSkipNetworkDriver()
    else:
        raise ValueError(f'{network_name} not a valid type.')
    
    network_experiments, data_experiments = driver.get_experiemnts(config)
    results = driver.run_experiments(config, network_experiments, data_experiments, data)
    
    return results, driver


def load_data(config: dict, network_name: str) -> tuple:
    """Assumes all NaNs are removed prior"""
    container = np.load(config['data']['saved_f'])
    raob = container['raob']
    rap  = container['rap']
    goes = container['goes']
    rtma = container['rtma']
    sonde_files = container['sonde_files'] # is this needed now?
    
    print(f'INFO: total data shape -- {raob.shape}, {rap.shape}, {goes.shape}, {rtma.shape}')

    train_i, val_i, test_i = ml.standard_partition_indicies(rap, percentages=(0.75,0.10,0.15),
                                                            shuffle=True, seed=1234)

    RAPtrain,  RAPval , RAPtest  = rap[train_i], rap[val_i], rap[test_i]
    RTMAtrain, RTMAval, RTMAtest = rtma[train_i], rtma[val_i], rtma[test_i]
    GOEStrain, GOESval, GOEStest = goes[train_i], goes[val_i], goes[test_i]
    RAOBtrain, RAOBval, RAOBtest = raob[train_i], raob[val_i], raob[test_i]
    
    print(f'INFO: partitioned data shape -- train: {RAPtrain.shape[0]}, val: {RAPval.shape[0]}, test: {RAPtest.shape[0]}')
    return (RAPtrain, RAPval, RAPtest, RTMAtrain, RTMAval, RTMAtest,
            GOEStrain, GOESval, GOEStest, RAOBtrain, RAOBval, RAOBtest)


def main(config_path: str):
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
    Usage: python -u -m soundings.experiments.driver -c ./soundings/experiments/configs/__config__.yaml
    """

    gpus = tf.config.get_visible_devices('GPU')
    for device in gpus:
        print(device)
        tf.config.experimental.set_memory_growth(device, True)

    parser = argparse.ArgumentParser(description='experimental configuration')
    parser.add_argument('-c', '--config', metavar='path', type=str,
                        required=True, help='the path to config file')
    args = parser.parse_args()
    main(config_path=args.config)
