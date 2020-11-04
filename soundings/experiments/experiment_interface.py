import abc
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf

from soundings.experiments import results as results_calc
from soundings.deep_learning import tf_neuralnetwork as nn

class ExperimentInterface(abc.ABC):
    
    @abc.abstractmethod
    def get_experiemnts(self, config: dict) -> list:
        """"""
        raise NotImplementedError

    @abc.abstractmethod
    def run_experiments(self, experiments: list,
                        Xtrain, Ttrain, Xval, Tval, 
                        Xtest, Ttest) -> dict:
        """"""
        raise NotImplementedError
        

class NeuralNetworkDriver(ExperimentInterface):
    
    def get_experiemnts(self, config: dict) -> list:
        n_hiddens_list = config['model']['n_hiddens_list']
        optimizers = config['model']['optimizers']
        learning_rates = config['model']['learning_rates']
        activations = config['model']['activations']
        losses = config['model']['losses']
        epochs = config['model']['epochs']
        batch_sizes = config['model']['batch_sizes']
        batchnorm = config['model']['batchnorm']
        dropout = config['model']['dropout']

        experiments = list(itertools.product(n_hiddens_list, optimizers, learning_rates,
                                             activations, losses, epochs, batch_sizes,
                                             dropout, batchnorm))
        return experiments

    def run_experiments(self, config, experiments: list,
                        Xtrain, Ttrain, Xval, Tval, 
                        Xtest, Ttest) -> pd.DataFrame:
        
        rap_input_dims = config['data']['rap']['input_dims']
        raob_output_dims = config['data']['rap']['output_dims']
        
        # flatten input and output
        Xt = Xtrain[:,:,rap_input_dims].reshape(Xtrain.shape[0],-1)
        Tt = Ttrain[:,:,raob_output_dims].reshape(Ttrain.shape[0],-1)
        
        Xv = Xval[:,:,rap_input_dims].reshape(Xval.shape[0],-1)
        Tv = Tval[:,:,raob_output_dims].reshape(Tval.shape[0],-1)
        
        Xe = Xtest[:,:,rap_input_dims].reshape(Xtest.shape[0],-1)
        Te = Ttest[:,:,raob_output_dims].reshape(Ttest.shape[0],-1)
        
        n_inputs = Xt.shape[1]
        n_outputs = Tt.shape[1]
        
        results = []

        for i, (hiddens, optim, lr, activ, loss, n_epochs,
             batch_size, batchnorm, dropout) in enumerate(experiments):
            
            print(f'INFO: trial -- {i + 1}/{len(experiments)}', hiddens, optim, lr, activ, loss, n_epochs,
                  batch_size, batchnorm, dropout)
            
            nnet = nn.NeuralNetwork(n_inputs, hiddens, n_outputs, activation=activ,
                                    batchnorm=batchnorm, dropout=dropout)

            nnet.train(Xt, Tt, n_epochs, batch_size, method=optim, verbose=False,
                       learning_rate=lr, validation=(Xv, Tv), loss_f=loss)
            
            r = {'n_inputs': n_inputs, 'hiddens': hiddens, 'n_outputs': n_outputs,
                 'optim': optim, 'lr': lr, 'activ': activ, 'loss': loss,
                 'n_epochs': n_epochs, 'batch_size': batch_size,
                 'batchnorm': batchnorm, 'dropout': dropout}
            
            for key, val in nnet.history.items():
                r[key] = val
            
            sets = ['train', 'val', 'test']
            profile = ['temperature', 'dewpoint']
            set_output_shape = [Ttrain[:, :, raob_output_dims].shape,
                                Tval[:, :, raob_output_dims].shape,
                                Ttest[:, :, raob_output_dims].shape]
            
            for t, profile_name in enumerate(profile):
                for i (X, T) in enumerate([(Xt, Tt), (Xv, Tv), (Xe, Te)]):
                    # RAOB profile
                    T = T.reshape(set_output_shape[i])[:, :, t]
                    
                    # ML results
                    # reshape flat output to temperature and dewpoint,
                    # then get desired profile, e.g., temperature/dewpoint
                    Y = nnet.use(X).reshape(set_output_shape[i])[:, :, t]
                    rmse, mean_rmse, rmse_sfc, mean_rmse_sfc = results_calc.compute_profile_rmses(Y, T)
                    r[f'ml_{profile_name}_{sets[i]}_rmse'] = rmse.tolist()
                    r[f'ml_{profile_name}_{sets[i]}_mean_rmse'] = mean_rmse
                    r[f'ml_{profile_name}_{sets[i]}_rmse_sfc'] = rmse_sfc.tolist()
                    r[f'ml_{profile_name}_{sets[i]}_mean_rmse_sfc'] = mean_rmse_sfc
                    
                    # RAP baseline error
                    # different for each set, but same for all experiments
                    Y = X.reshape(set_output_shape[i])[:, :, t]
                    rmse, mean_rmse, rmse_sfc, mean_rmse_sfc = results_calc.compute_profile_rmses(Y, T)
                    r[f'rap_{profile_name}_{sets[i]}_rmse'] = rmse.tolist()
                    r[f'rap_{profile_name}_{sets[i]}_mean_rmse'] = mean_rmse
                    r[f'rap_{profile_name}_{sets[i]}_rmse_sfc'] = rmse_sfc.tolist()
                    r[f'rap_{profile_name}_{sets[i]}_mean_rmse_sfc'] = mean_rmse_sfc
                    
            results.append(r)
            break
            
        return pd.DataFrame(results)
            
            
            
            
            
            
            
            