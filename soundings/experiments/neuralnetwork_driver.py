import time
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf

from soundings.experiments.experiment_interface import ExperimentInterface
from soundings.experiments import results as results_calc
from soundings.deep_learning import tf_neuralnetwork as nn


class NeuralNetworkDriver(ExperimentInterface):
    
    def get_experiemnts(self, config: dict) -> (list, list):
        # network config
        n_hiddens_list = config['model']['n_hiddens_list']
        optimizers     = config['model']['optimizers']
        learning_rates = config['model']['learning_rates']
        activations    = config['model']['activations']
        losses = config['model']['losses']
        epochs = config['model']['epochs']
        batch_sizes = config['model']['batch_sizes']
        dropout     = config['model']['dropout']
        batchnorm   = config['model']['batchnorm']
        
        network_experiments = list(itertools.product(n_hiddens_list, optimizers, learning_rates,
                                                     activations, losses, epochs, batch_sizes,
                                                     dropout, batchnorm))
        
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
        
        (RAPtrain, RAPval, RAPtest,
         RTMAtrain, RTMAval, RTMAtest,
         GOEStrain, GOESval, GOEStest,
         RAOBtrain, RAOBval, RAOBtest) = data
            
        # train
        Xt = RAPtrain[:,:,rap_input_dims].reshape(RAPtrain.shape[0],-1)
        if rtma_input_channels:
            Xt = np.hstack((Xt, RTMAtrain[:,:,:,rtma_input_channels].reshape(RTMAtrain.shape[0],-1)))
        if goes_input_channels:
            Xt = np.hstack((Xt, GOEStrain[:,:,:,goes_input_channels].reshape(GOEStrain.shape[0],-1)))
        Tt = RAOBtrain[:,:,rap_output_dims].reshape(RAOBtrain.shape[0],-1)
        # validation
        Xv = RAPval[:,:,rap_input_dims].reshape(RAPval.shape[0],-1)
        if rtma_input_channels:
            Xv = np.hstack((Xv, RTMAval[:,:,:,rtma_input_channels].reshape(RTMAval.shape[0],-1)))
        if goes_input_channels:
            Xv = np.hstack((Xv, GOESval[:,:,:,goes_input_channels].reshape(GOESval.shape[0],-1)))
        Tv = RAOBval[:,:,rap_output_dims].reshape(RAOBval.shape[0],-1)
        # test
        Xe = RAPtest[:,:,rap_input_dims].reshape(RAPtest.shape[0],-1)
        if rtma_input_channels:
            Xe = np.hstack((Xe, RTMAtest[:,:,:,rtma_input_channels].reshape(RTMAtest.shape[0],-1)))
        if goes_input_channels:
            Xe = np.hstack((Xe, GOEStest[:,:,:,goes_input_channels].reshape(GOEStest.shape[0],-1)))
        Te = RAOBtest[:,:,rap_output_dims].reshape(RAOBtest.shape[0],-1)

        print('INFO: data dimensions -', Xt.shape, Tt.shape, Xv.shape, Tv.shape, Xe.shape, Te.shape)
        return Xt, Tt, Xv, Tv, Xe, Te
    
    
    def run_experiments(self, config: dict, network_experiments: list,
                        data_experiments: list, data: tuple) -> pd.DataFrame:
        
        (RAPtrain, RAPval, RAPtest,
         RTMAtrain, RTMAval, RTMAtest,
         GOEStrain, GOESval, GOEStest,
         RAOBtrain, RAOBval, RAOBtest) = data
        
        results = []
        i = 0
        for rap_input_dims, rap_output_dims, rtma_input_channels, goes_input_channels in data_experiments:
            Xt, Tt, Xv, Tv, Xe, Te = self.organize_data(data, rap_input_dims, rap_output_dims,
                                                        rtma_input_channels, goes_input_channels)
            n_network_inputs = Xt.shape[1]
            n_network_outputs = Tt.shape[1] # (None, 256, N)
            for _, (hiddens, optim, lr, activ, loss, n_epochs,
                    batch_size, dropout, batchnorm) in enumerate(network_experiments):
                print(f'INFO: trial -- {i + 1}/{len(data_experiments) * len(network_experiments)}', hiddens, optim,
                  lr, activ, loss, n_epochs, batch_size, dropout, batchnorm)
            
                nnet = nn.NeuralNetwork(n_network_inputs, hiddens, n_network_outputs, activation=activ,
                                        dropout=dropout, batchnorm=batchnorm)
                nnet.model.summary()
                
                nnet.train(Xt, Tt, n_epochs, batch_size, method=optim, verbose=False,
                           learning_rate=lr, validation=(Xv, Tv), loss_f=loss)
                
                print(f'INFO: finished training model in {nnet.training_time:.3f} seconds. Benchmarking now.')
                
                r = {'rap_input_dims': rap_input_dims, 'rap_output_dims': rap_output_dims,
                     'rtma_input_channels': rtma_input_channels, 'goes_input_channels': goes_input_channels,
                     'n_network_inputs': n_network_inputs, 'hiddens': hiddens, 'n_network_outputs': n_network_outputs,
                     'optim': optim, 'lr': lr, 'activ': activ, 'loss_f': loss, 'n_epochs': n_epochs, 
                     'batch_size': batch_size, 'batchnorm': batchnorm, 'dropout': dropout}
                
                # metrics from nnet.model.history
                for key, val in nnet.history.items():
                    r[key] = val
                
                TEMP, DEWPT = 0, 1
                sets = ['train', 'val', 'test']
                
                for j, (X, RAP, T) in enumerate([(Xt, RAPtrain, RAOBtrain),
                                                 (Xv, RAPval  , RAOBval),
                                                 (Xe, RAPtest , RAOBtest)]):
                    
                    Y = nnet.use(X).reshape(RAP[:,:,rap_output_dims].shape) # (None, 256, N)
                    
                    (rmse, mean_rmse,
                     rmse_sfc, mean_rmse_sfc) = results_calc.compute_profile_rmses(Y[:,:,TEMP], T[:, :, 1])
                    r[f'ml_temperature_{sets[j]}_rmse'] = rmse.tolist()
                    r[f'ml_temperature_{sets[j]}_mean_rmse'] = mean_rmse
                    r[f'ml_temperature_{sets[j]}_rmse_sfc'] = rmse_sfc.tolist()
                    r[f'ml_temperature_{sets[j]}_mean_rmse_sfc'] = mean_rmse_sfc

                    (rmse, mean_rmse,
                     rmse_sfc, mean_rmse_sfc) = results_calc.compute_profile_rmses(Y[:,:,DEWPT], T[:, :, 2])
                    r[f'ml_dewpoint_{sets[j]}_rmse'] = rmse.tolist()
                    r[f'ml_dewpoint_{sets[j]}_mean_rmse'] = mean_rmse
                    r[f'ml_dewpoint_{sets[j]}_rmse_sfc'] = rmse_sfc.tolist()
                    r[f'ml_dewpoint_{sets[j]}_mean_rmse_sfc'] = mean_rmse_sfc
                    
                results.append(r)
                i += 1
                
        return pd.DataFrame(results)
            