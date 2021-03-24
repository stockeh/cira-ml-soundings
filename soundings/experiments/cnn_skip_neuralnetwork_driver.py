import time
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from ast import literal_eval

from soundings.experiments.experiment_interface import ExperimentInterface
from soundings.experiments import results as results_calc
from soundings.deep_learning import tf_neuralnetwork as nn
from soundings.deep_learning import mlutilities as ml

class CNNSkipNetworkDriver(ExperimentInterface):
    
    def get_experiemnts(self, config: dict) -> (list, list):
        # network config
        
        n_hiddens_list           = config['model']['n_hiddens_list']
        n_units_in_conv_layers   = config['model']['n_units_in_conv_layers']
        kernels_size_and_strides = config['model']['kernels_size_and_strides']
        rap_activations   = config['model']['rap_activations']
        dense_activations = config['model']['dense_activations']
        optimizers        = config['model']['optimizers']
        learning_rates    = config['model']['learning_rates']
        losses = config['model']['losses']
        epochs = config['model']['epochs']
        batch_sizes = config['model']['batch_sizes']
        dropout     = config['model']['dropout']
        batchnorm   = config['model']['batchnorm']
        regularization = config['model']['regularization']
        
        network_experiments = list(itertools.product(n_hiddens_list, n_units_in_conv_layers, kernels_size_and_strides,
                                                     rap_activations, dense_activations, optimizers, learning_rates,
                                                     losses, epochs, batch_sizes, dropout, batchnorm, regularization))
        
        # data config
        rap_input_dims  = config['data']['rap']['input_dims']
        rap_output_dims = config['data']['rap']['output_dims'] # same for RAOB output
        rtma_input_channels = config['data']['rtma']['input_channels']
        goes_input_channels = config['data']['goes']['input_channels']
        
        data_experiments = list(itertools.product(rap_input_dims, rap_output_dims, 
                                                  rtma_input_channels, goes_input_channels))
        assert(len(rap_output_dims[0]) == 2), f'output MUST contain temperature (0) and dewpoint (1) profile.'
        
        return network_experiments, data_experiments

    
    def organize_data(self, data, rap_input_dims, rap_output_dims, rtma_input_channels, goes_input_channels):
        print('INFO: data organization -', rap_input_dims, rap_output_dims,
              rtma_input_channels, goes_input_channels)
        
        (RAPtrain , RAPval,  RAPtest,
         RTMAtrain, RTMAval, RTMAtest,
         GOEStrain, GOESval, GOEStest,
         RAOBtrain, RAOBval, RAOBtest) = data
            
        Xti, Xvi, Xei = [], [], []
        if rtma_input_channels:
            Xti.append(RTMAtrain[:,:,:,rtma_input_channels])
            Xvi.append(RTMAval[:,:,:,rtma_input_channels])
            Xei.append(RTMAtest[:,:,:,rtma_input_channels])
        if goes_input_channels:
            Xti.append(GOEStrain[:,:,:,goes_input_channels])
            Xvi.append(GOESval[:,:,:,goes_input_channels])
            Xei.append(GOEStest[:,:,:,goes_input_channels])
        
        # compute mean for each channel and keep dims, i.e. (N, 1, 1, D)
        im_dim = (1,2)
        
        # train
        Xtr = RAPtrain[:,:,rap_input_dims]
        Xti = np.expand_dims(np.concatenate(Xti, axis=3).mean(axis=im_dim), axis=im_dim) if len(Xti) else None
        Tt  = RAOBtrain[:,:,rap_output_dims].reshape(RAOBtrain.shape[0],-1)
        # validation
        Xvr = RAPval[:,:,rap_input_dims]
        Xvi = np.expand_dims(np.concatenate(Xvi, axis=3).mean(axis=im_dim), axis=im_dim) if len(Xvi) else None
        Tv  = RAOBval[:,:,rap_output_dims].reshape(RAOBval.shape[0],-1)
        # test
        Xer = RAPtest[:,:,rap_input_dims]
        Xei = np.expand_dims(np.concatenate(Xei, axis=3).mean(axis=im_dim), axis=im_dim) if len(Xei) else None
        Te  = RAOBtest[:,:,rap_output_dims].reshape(RAOBtest.shape[0],-1)

        Xti_print = Xti.shape if Xti is not None else 'None'
        Xvi_print = Xvi.shape if Xvi is not None else 'None'
        Xei_print = Xei.shape if Xei is not None else 'None'
        
        print('INFO: data dimensions -', Xtr.shape, Xti_print, Tt.shape, 
              Xvr.shape, Xvi_print, Tv.shape, 
              Xer.shape, Xei_print, Te.shape)
        return Xtr, Xti, Tt, Xvr, Xvi, Tv, Xer, Xei, Te
    
    
    def run_experiments(self, config: dict, network_experiments: list,
                        data_experiments: list, data: tuple) -> pd.DataFrame:
        
        (RAPtrain, RAPval, RAPtest,
         RTMAtrain, RTMAval, RTMAtest,
         GOEStrain, GOESval, GOEStest,
         RAOBtrain, RAOBval, RAOBtest) = data
        
        repeat = config['model']['repeat_factor']
        results = []
        i = 0
        
        for _ in range(repeat):
            for rap_input_dims, rap_output_dims, rtma_input_channels, goes_input_channels in data_experiments:
                Xtr, Xti, Tt, Xvr, Xvi, Tv, Xer, Xei, Te = self.organize_data(data, rap_input_dims, rap_output_dims,
                                                                              rtma_input_channels, goes_input_channels)
                n_rap_inputs = Xtr.shape[1:] # (256, 4)
                n_im_inputs  = Xti.shape[1:] if Xti is not None else None # e.g., (3, 3, D) or (1, 1, D)
                n_network_outputs = Tt.shape[1]
                for _, (n_hiddens_list, n_units_in_conv_layers, kernels_size_and_stride,
                        rap_activation, dense_activation, optim, lr, loss,
                        n_epochs,batch_size, dropout, batchnorm, regularization) in enumerate(network_experiments):
                    print(f'INFO: trial -- {i + 1}/{len(data_experiments) * len(network_experiments) * repeat}', 
                          n_hiddens_list, n_units_in_conv_layers, kernels_size_and_stride,
                          rap_activation, dense_activation, optim, lr, loss,
                          n_epochs, batch_size, dropout, batchnorm, regularization)
                    if config['model']['network'] == 'MultiConvolutionalNeuralNetwork':
                        network = nn.MultiConvolutionalNeuralNetwork
                    elif config['model']['network'] == 'SkipNeuralNetwork':
                        network = nn.SkipNeuralNetwork
                    else:
                        raise ValueException(f"invalid network spec, {config['model']['network']}")
                    nnet = network(n_rap_inputs, n_im_inputs, n_hiddens_list, n_units_in_conv_layers,
                                                              [literal_eval(kernels_size_and_stride)]* \
                                                                  len(n_units_in_conv_layers), # all same size
                                                              n_network_outputs, rap_activation=rap_activation,
                                                              dense_activation=dense_activation, batchnorm=batchnorm, 
                                                              dropout=dropout, regularization=regularization)
                    nnet.model.summary()
                    nnet.train(Xtr, Xti, Tt, n_epochs, batch_size, method=optim, verbose=False,
                               learning_rate=lr, validation=(Xvr, Xvi, Tv), loss_f=loss)
                    print(f'INFO: finished training model in {nnet.training_time:.3f} seconds. Benchmarking now.')

                    r = {# data params
                         'rap_input_dims': rap_input_dims, 'rap_output_dims': rap_output_dims,
                         'rtma_input_channels': rtma_input_channels, 'goes_input_channels': goes_input_channels,
                         # model params
                         'n_rap_inputs': n_rap_inputs, 'n_im_inputs': n_im_inputs, 'hiddens': n_hiddens_list, 
                         'n_network_outputs': n_network_outputs, 'n_units_in_conv_layers': n_units_in_conv_layers,
                         'kernels_size_and_stride': kernels_size_and_stride, 'rap_activation': rap_activation,
                         'dense_activation': dense_activation, 'optim': optim, 'lr': lr, 'loss_f': loss,
                         'n_epochs': n_epochs, 'batch_size': batch_size, 'batchnorm': batchnorm, 'dropout': dropout,
                         'regularization': regularization}

                    # metrics from nnet.model.history
                    for key, val in nnet.history.items():
                        r[key] = val

                    TEMP, DEWPT = 0, 1
                    sets = ['train', 'val', 'test']

                    for j, (Xr, Xi, T, RAP, RAOB) in enumerate([(Xtr, Xti, Tt, RAPtrain, RAOBtrain),
                                                                (Xvr, Xvi, Tv, RAPval  , RAOBval),
                                                                (Xer, Xei, Te, RAPtest , RAOBtest)]):

                        X = {'rap': Xr, 'im': Xi} if Xi is not None else {'rap': Xr}
                        Y = nnet.use(X)
                        
                        surface_error=25
                        r[f'ml_total_{sets[j]}_rmse'] = ml.rmse(Y, T)
                        r[f'ml_total_{sets[j]}_rmse_sfc'] = ml.rmse(Y[:,:surface_error*2], T[:,:surface_error*2])
                        
                        Y = Y.reshape(RAP[:,:,rap_output_dims].shape) # (None, 256, N)

                        (rmse, mean_rmse,
                         rmse_sfc, mean_rmse_sfc) = results_calc.compute_profile_rmses(Y[:,:,TEMP], 
                                                                                       RAOB[:, :, 1], 
                                                                                       surface_error)
                        r[f'ml_temperature_{sets[j]}_rmse'] = rmse.tolist()
                        r[f'ml_temperature_{sets[j]}_mean_rmse'] = mean_rmse
                        r[f'ml_temperature_{sets[j]}_rmse_sfc'] = rmse_sfc.tolist()
                        r[f'ml_temperature_{sets[j]}_mean_rmse_sfc'] = mean_rmse_sfc

                        (rmse, mean_rmse,
                         rmse_sfc, mean_rmse_sfc) = results_calc.compute_profile_rmses(Y[:,:,DEWPT], 
                                                                                       RAOB[:, :, 2], 
                                                                                       surface_error)
                        r[f'ml_dewpoint_{sets[j]}_rmse'] = rmse.tolist()
                        r[f'ml_dewpoint_{sets[j]}_mean_rmse'] = mean_rmse
                        r[f'ml_dewpoint_{sets[j]}_rmse_sfc'] = rmse_sfc.tolist()
                        r[f'ml_dewpoint_{sets[j]}_mean_rmse_sfc'] = mean_rmse_sfc

                    results.append(r)
                    i += 1
                
        return pd.DataFrame(results)
            