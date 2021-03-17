import copy
import datetime
import time
import random
import pickle

import os
import numpy as np
import tensorflow as tf

from soundings.deep_learning import callbacks
from soundings.deep_learning import tf_metrics as metrics

from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops

def weighted_mean_squared_error(y_true, y_pred, scale, lmda):
    """exponential decay weighted MSE"""
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    diff = math_ops.squared_difference(y_pred, y_true)

    x = np.arange(diff.shape[1])
    y = lambda x: scale * np.exp(-lmda * x) + 1
    diff *= y(x)

    return K.mean(diff, axis=-1)

def seperated_mean_squared_error(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    seperation = int(y_true.shape[1] * 0.2)
    diff1 = math_ops.square(math_ops.squared_difference(y_pred[:,:seperation], y_true[:,:seperation]))
    diff2 = math_ops.squared_difference(y_pred[:,seperation:], y_true[:,seperation:])
    diff  = tf.concat([diff1, diff2], axis=1)
    return K.mean(diff, axis=-1)

def loadnn(path):
    try:
        with open(path + '/class.pickle', 'rb') as f:
            c = pickle.load(f)
        tf_model_path = os.path.join(path, 'model.h5')
        c.model = tf.keras.models.load_model(tf_model_path,
           custom_objects={'rmse': metrics.unstd_rmse(c._unstandardizeT),
                           'mse': metrics.unstd_mse(c._unstandardizeT),
                           'truncated_mse': metrics.unstd_truncated_mse(c._unstandardizeT)})
        return c
    except Exception as e:
        raise e


class NeuralNetwork():
    def __init__(self, n_inputs, n_hiddens_list, n_outputs, activation='tanh',
                 batchnorm=False, dropout=False, regularization=False, seed=None):

        if not isinstance(n_hiddens_list, list):
            raise Exception(
                f'{type(self).__name__}: n_hiddens_list must be a list.')

        self.seed = seed
        self._set_seed()
        tf.keras.backend.clear_session()

        self.n_inputs = n_inputs
        self.n_hiddens_list = n_hiddens_list
        self.n_outputs = n_outputs

        kernel_regularizer = tf.keras.regularizers.l2(0.001) if regularization else None
        
        X = Z = tf.keras.Input(shape=(n_inputs,))
        if not (n_hiddens_list == [] or n_hiddens_list == [0]):
            for i, units in enumerate(n_hiddens_list):
                Z = tf.keras.layers.Dense(units, kernel_regularizer=kernel_regularizer)(Z)
                if batchnorm:
                    Z = tf.keras.layers.BatchNormalization()(Z)
                Z = tf.keras.layers.Activation(activation)(Z)
                if dropout:
                    Z = tf.keras.layers.Dropout(0.20)(Z)
        Y = tf.keras.layers.Dense(n_outputs, kernel_regularizer=kernel_regularizer)(Z)
        self.model = tf.keras.Model(inputs=X, outputs=Y)

        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None

        self.Xmax = None
        self.Xmin = None
        self.Tmax = None
        self.Tmin = None
        
        self.history = None
        self.training_time = None

    def __repr__(self):
        str = f'{type(self).__name__}({self.n_inputs}, {self.n_hiddens_list}, {self.n_outputs})'
        if self.history:
            str += f"\n  Final objective value is {self.history['loss'][-1]:.5f} in {self.training_time:.4f} seconds."
        else:
            str += '  Network is not trained.'
        return str

    def _set_seed(self):
        if self.seed:
            np.random.seed(self.seed)
            random.seed(self.seed)
            tf.random.set_seed(self.seed)

    def _setup_standardize(self, X, T):
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
            self.Xconstant = self.Xstds == 0
            self.XstdsFixed = copy.copy(self.Xstds)
            self.XstdsFixed[self.Xconstant] = 1

        if self.Tmeans is None:
            self.Tmeans = T.mean(axis=0)
            self.Tstds = T.std(axis=0)
            self.Tconstant = self.Tstds == 0
            self.TstdsFixed = copy.copy(self.Tstds)
            self.TstdsFixed[self.Tconstant] = 1

    def _standardizeX(self, X):
        result = (X - self.Xmeans) / self.XstdsFixed
        result[:, self.Xconstant] = 0.0
        return result

    def _unstandardizeX(self, Xs):
        return self.Xstds * Xs + self.Xmeans

    def _standardizeT(self, T):
        result = (T - self.Tmeans) / self.TstdsFixed
        result[:, self.Tconstant] = 0.0
        return result

    def _unstandardizeT(self, Ts):
        return self.Tstds * Ts + self.Tmeans
    
    
#     def _setup_standardize(self, X, T):
#         if self.Xmax is None:
#             self.Xmax = X.max(axis=0)
#             self.Xmin = X.min(axis=0)
            
#         if self.Tmax is None:
#             self.Tmax = T.max(axis=0)
#             self.Tmin = T.min(axis=0)
    
#     def _standardizeX(self, X):
#         return (X - self.Xmin) / (self.Xmax - self.Xmin)

#     def _unstandardizeX(self, Xs):
#         return Xs * (self.Xmax - self.Xmin) + self.Xmin

#     def _standardizeT(self, T):
#         return (T - self.Tmin) / (self.Tmax - self.Tmin)

#     def _unstandardizeT(self, Ts):
#         return Ts * (self.Tmax - self.Tmin) + self.Tmin

    
    def train(self, X, T, n_epochs, batch_size, method='sgd',
              verbose=False, learning_rate=0.001, validation=None, loss_f='MSE'):
        """Use Keras Functional API to train model"""

        self._set_seed()
        self._setup_standardize(X, T)
        X = self._standardizeX(X)
        T = self._standardizeT(T)

        if validation is not None:
            try:
                validation = (self._standardizeX(
                    validation[0]), self._standardizeT(validation[1]))
            except:
                raise TypeError(
                    f'validation must be of the following shape: (X, T)')

        try:
            if method == 'sgd':
                algo = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            elif method == 'adam':
                algo = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            elif method == 'Adagrad':
                algo = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
            elif method == 'RMSprop':
                algo = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
            elif method == 'Adadelta':
                algo = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
        except:
            raise Exception(
                "train: method={method} not a valid optimizer in soundings.nn library.")

            
        if loss_f == 'MSE': # default
            loss = tf.keras.losses.MSE
        elif loss_f == 'MAE':
            loss = tf.keras.losses.MAE
        else: # custom loss function
            loss = loss_f
            
        self.model.compile(optimizer=algo, loss=loss,
                           metrics=[tf.keras.metrics.RootMeanSquaredError(),
                                    tf.keras.metrics.MeanSquaredError(),
                                    tf.keras.metrics.MeanAbsoluteError()])
                           # metrics=[metrics.unstd_mse(self._unstandardizeT),
                           #         metrics.unstd_truncated_mse(self._unstandardizeT),
                           #         metrics.unstd_rmse(self._unstandardizeT)]) 0.001
        callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10)] if validation is not None else []
        if verbose:
            callback.append(callbacks.TrainLogger(n_epochs, step=n_epochs//5))

        start_time = time.time()
        self.history = self.model.fit(X, T, batch_size=batch_size, epochs=n_epochs,
                                      verbose=0, callbacks=callback,
                                      validation_data=validation).history
        self.training_time = time.time() - start_time
        return self

    def use(self, X):
        # Set to error logging after model is trained
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        X = self._standardizeX(X)
        Y = self._unstandardizeT(self.model.predict(X))
        return Y

    def save(self, path):
        tf_model_path = os.path.join(path, 'model.h5')
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(tf_model_path)
        del self.model
        with open(os.path.join(path, 'class.pickle'), 'wb') as f:
            pickle.dump(self, f)
        self.model = tf.keras.models.load_model(tf_model_path,
               custom_objects={'rmse': metrics.unstd_rmse(self._unstandardizeT),
                               'mse': metrics.unstd_mse(self._unstandardizeT),
                               'truncated_mse': metrics.unstd_truncated_mse(self._unstandardizeT)})


class ConvolutionalNeuralNetwork(NeuralNetwork):
    def __init__(self, n_inputs, n_units_in_conv_layers,
                 kernels_size_and_stride, n_outputs, activation='tanh',
                 batchnorm=False, dropout=False, seed=None):

        if not isinstance(n_units_in_conv_layers, (list, tuple)):
            raise Exception(
                f'{type(self).__name__}: n_units_in_conv_layers must be a list.')

        if not isinstance(kernels_size_and_stride, list):
            raise Exception(
                f'{type(self).__name__}: kernels_size_and_stride must be a list.')

        self.seed = seed
        self._set_seed()
        tf.keras.backend.clear_session()

        self.n_inputs = n_inputs
        self.n_units_in_conv_layers = n_units_in_conv_layers
        self.kernels_size_and_stride = kernels_size_and_stride
        self.n_outputs = n_outputs

        X = Z = tf.keras.Input(shape=n_inputs)
        for (kernel, stride), units in zip(kernels_size_and_stride, n_units_in_conv_layers):
            Z = tf.keras.layers.Conv1D(
                units, kernel_size=kernel, strides=stride, padding='same')(Z)
            if batchnorm:
                Z = tf.keras.layers.BatchNormalization()(Z)
            Z = tf.keras.layers.Activation(activation)(Z)
            Z = tf.keras.layers.MaxPooling1D(pool_size=2)(Z)
            if dropout:
                Z = tf.keras.layers.Dropout(0.20)(Z)
        Y = tf.keras.layers.Dense(n_outputs)(tf.keras.layers.Flatten()(Z))

        self.model = tf.keras.Model(inputs=X, outputs=Y)

        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None

        self.history = None
        self.training_time = None

    def __repr__(self):
        str = f'{type(self).__name__}({self.n_inputs}, {self.n_units_in_conv_layers}, {self.kernels_size_and_stride}, {self.n_outputs})'
        if self.history:
            str += f"\n  Final objective value is {self.history['loss'][-1]:.5f} in {self.training_time:.4f} seconds."
        else:
            str += '  Network is not trained.'
        return str


class ConvolutionalAutoEncoder(NeuralNetwork):
    def __init__(self, n_inputs, n_units_in_conv_layers,
                 kernels_size_and_stride, n_outputs, activation='relu', dropout=False,
                 n_hidden_dims=128, seed=None):

        if not isinstance(n_units_in_conv_layers, (list, tuple)):
            raise Exception(
                f'{type(self).__name__}: n_units_in_conv_layers must be a list.')

        if not isinstance(kernels_size_and_stride, list):
            raise Exception(
                f'{type(self).__name__}: kernels_size_and_stride must be a list.')

        self.seed = seed
        self._set_seed()
        tf.keras.backend.clear_session()

        self.n_inputs = n_inputs
        self.n_units_in_conv_layers = n_units_in_conv_layers
        self.kernels_size_and_stride = kernels_size_and_stride
        self.n_outputs = n_outputs
        self.n_hidden_dim = n_hidden_dims

        # encoder
        X = tf.keras.Input(shape=n_inputs)
        Z = X
        for (kernel, stride), units in zip(kernels_size_and_stride, n_units_in_conv_layers):
            Z = tf.keras.layers.Conv1D(
                units, kernel_size=kernel, strides=stride, activation=activation, padding='same')(Z)
            Z = tf.keras.layers.MaxPooling1D(pool_size=2)(Z)
            if dropout:
                Z = tf.keras.layers.Dropout(0.20)(Z)

        # latent vector
        conv_shape = Z.shape[1:]
        F = tf.keras.layers.Flatten()(Z)
        Z = tf.keras.layers.Dense(n_hidden_dims, activation='tanh')(F)

        # decoder (input of `n_hidden_dim`)
        Z = tf.keras.layers.Dense(F.shape[1], activation='tanh')(Z)
        Z = tf.keras.layers.Reshape(conv_shape)(Z)

        for (kernel, stride), units in zip(reversed(kernels_size_and_stride), reversed(n_units_in_conv_layers)):
            Z = tf.keras.layers.Conv1D(
                units, kernel_size=kernel, strides=stride, activation=activation, padding='same')(Z)
            Z = tf.keras.layers.UpSampling1D(size=2)(Z)
            if dropout:
                Z = tf.keras.layers.Dropout(0.20)(Z)
        Z = tf.keras.layers.Conv1D(
            1, kernel_size=kernels_size_and_stride[0][0], strides=kernels_size_and_stride[0][1],
            activation=activation, padding='same')(Z)
        if dropout:
            Z = tf.keras.layers.Dropout(0.20)(Z)
        Y = tf.keras.layers.Dense(n_inputs[0])(tf.keras.layers.Flatten()(Z))
        self.model = tf.keras.Model(inputs=X, outputs=Y)

        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None

        self.history = None
        self.training_time = None

    def __repr__(self):
        str = f'{type(self).__name__}({self.n_inputs}, {self.n_units_in_conv_layers}, {self.kernels_size_and_stride}, {self.n_outputs})'
        if self.history:
            str += f"\n  Final objective value is {self.history['loss'][-1]:.5f} in {self.training_time:.4f} seconds."
        else:
            str += '  Network is not trained.'
        return str

    
class SkipNeuralNetwork():
    """Skip Neural Network with multiple inputs (optionally).
    Specify, `n_im_inputs=None` to not add additional inputs.
    """
    def __init__(self, n_rap_inputs, n_im_inputs, n_hiddens_list,
                 n_units_in_conv_layers, kernels_size_and_stride,
                 n_outputs, rap_activation='relu', dense_activation='tanh', 
                 batchnorm=False, dropout=False, seed=None):

        if n_im_inputs is not None:
            assert ((len(n_im_inputs) == 3)), f'Image must be HxWxC dimensions, {n_im_inputs}'
        assert (isinstance(n_hiddens_list, list)), f'{type(self).__name__}: n_hiddens_list must be a list.'
        assert (isinstance(n_units_in_conv_layers, list)), f'{type(self).__name__}: n_units_in_conv_layers must be a list.'
        assert (isinstance(kernels_size_and_stride, list)), f'{type(self).__name__}: kernels_size_and_stride must be a list.'

        self.seed = seed
        self._set_seed()
        tf.keras.backend.clear_session()

        self.n_rap_inputs = n_rap_inputs
        self.n_im_inputs  = n_im_inputs
        self.n_hiddens_list = n_hiddens_list
        self.n_units_in_conv_layers  = n_units_in_conv_layers
        self.kernels_size_and_stride = kernels_size_and_stride
        self.n_outputs = n_outputs
        
        l2_rate = 1e-5
        dropout_rate = 0.1
        
        # encoder
        X1 = Z1 = tf.keras.Input(shape=n_rap_inputs, name='rap')

        for i, ((kernel, stride), units) in enumerate(zip(kernels_size_and_stride[:],
                                                          n_units_in_conv_layers[:])):
            Z1 = tf.keras.layers.Conv1D(units, kernel_size=kernel,
                                        strides=stride, padding='same',
                                        kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(Z1)
            Z1 = tf.keras.layers.Activation(rap_activation)(Z1)
            Z1 = tf.keras.layers.Conv1D(units, kernel_size=kernel,
                                        strides=stride, padding='same', name=f'skip_conv1d_{i}',
                                        kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(Z1)
            Z1 = tf.keras.layers.Activation(rap_activation)(Z1)
            Z1 = tf.keras.layers.MaxPooling1D(pool_size=2)(Z1)
            if dropout:
                Z1 = tf.keras.layers.Dropout(dropout_rate)(Z1)

        skips = list(reversed([layer for layer in tf.keras.Model(X1, Z1).layers if 'skip' in layer.name]))
    
        # bottleneck layer
        Z1 = tf.keras.layers.Conv1D(
                1, 
                kernel_size=kernels_size_and_stride[-1][0], 
                strides=kernels_size_and_stride[-1][1], 
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(Z1)
        Z1 = tf.keras.layers.Activation(rap_activation)(Z1)
        
        # IM Input
        if self.n_im_inputs is not None:
            X2 = tf.keras.Input(shape=n_im_inputs, name='im')
            Z2 = tf.keras.layers.Flatten()(X2)
            bottleneck_shape = Z1.shape.as_list()[1:]
            Z1 = tf.keras.layers.Flatten()(Z1)
            Z1 = tf.keras.layers.Concatenate(axis=1)([Z1, Z2]) # Join IM & RAP
            Z1 = tf.keras.layers.Dense(np.prod(bottleneck_shape),
                                       kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(Z1)
            Z1 = tf.keras.layers.Activation(rap_activation)(Z1)
            Z = tf.keras.layers.Reshape(bottleneck_shape)(Z1)
            inputs = [X1, X2]
        else:
            Z = Z1
            inputs = X1
        if dropout:
            Z = tf.keras.layers.Dropout(dropout_rate)(Z)
    
        # decoder
        for (kernel, stride), units, skip in zip(reversed(kernels_size_and_stride[:]),
                                                 reversed(n_units_in_conv_layers[:]),
                                                 skips):
            Z = tf.keras.layers.Conv1D(units, kernel_size=kernel, 
                                       strides=stride, padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(Z)
            Z = tf.keras.layers.Activation(rap_activation)(Z)
            Z = tf.keras.layers.UpSampling1D(size=2)(Z)
            Z = tf.keras.layers.Concatenate(axis=2)([Z, skip.output])
            # Z = tf.keras.layers.Add()([Z, skip.output])
            Z = tf.keras.layers.Conv1D(units, kernel_size=kernel, 
                                       strides=stride, padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(Z)
            Z = tf.keras.layers.Activation(rap_activation)(Z)
            if dropout:
                Z = tf.keras.layers.Dropout(dropout_rate)(Z)
        
        # final conv layer (linear; no activation)
        Z = tf.keras.layers.Conv1D(
                n_outputs / n_rap_inputs[0], kernel_size=kernels_size_and_stride[0][0], 
                strides=kernels_size_and_stride[0][1], padding='same',
                kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(Z)

        # add only the temperature profile back to Z.
        Z = tf.keras.layers.Add()([X1[:,:,1:3], Z]) # temperature & dewpoint, e.g. (256,4) + (256,2)
        Z = tf.keras.layers.Flatten()(Z) # (512,)
                                       
        # Dense Layers 
        if not (n_hiddens_list == [] or n_hiddens_list == [0]):
            for units in n_hiddens_list:
                if dropout:
                    Z = tf.keras.layers.Dropout(dropout_rate)(Z) 
                Z = tf.keras.layers.Dense(units, activation=dense_activation)(Z) 
            if dropout:
                Z = tf.keras.layers.Dropout(dropout_rate)(Z) 
            # Output Layer
            Y = tf.keras.layers.Dense(n_outputs, name='out')(Z)
        else:
            Y = Z
            
        self.model = tf.keras.Model(inputs=inputs, outputs=Y)   
    
        self.RAPmeans  = None
        self.RAPstds   = None
        self.IMmeans   = None
        self.IMstds    = None
        self.RAOBmeans = None
        self.RAOBstds  = None

        self.history = None
        self.training_time = None

    def __repr__(self):
        str = f'{type(self).__name__}({self.n_outputs})'
        if self.history:
            str += f"\n  Final objective value is {self.history['loss'][-1]:.5f} in {self.training_time:.4f} seconds."
        else:
            str += '  Network is not trained.'
        return str

    def _set_seed(self):
        if self.seed:
            np.random.seed(self.seed)
            random.seed(self.seed)
            tf.random.set_seed(self.seed)

    def _setup_standardize(self, rap, im, raob):
        if self.RAPmeans is None:
            self.RAPmeans = rap.mean(axis=0)
            self.RAPstds = rap.std(axis=0)
            self.RAPconstant = self.RAPstds == 0
            self.RAPstdsFixed = copy.copy(self.RAPstds)
            self.RAPstdsFixed[self.RAPconstant] = 1

        if self.IMmeans is None and self.n_im_inputs is not None:
            # check if IM is used as input to the model
            self.IMmeans = im.mean(axis=0)
            self.IMstds = im.std(axis=0)
            self.IMconstant = self.IMstds == 0
            self.IMstdsFixed = copy.copy(self.IMstds)
            self.IMstdsFixed[self.IMconstant] = 1
            
        if self.RAOBmeans is None:
            self.RAOBmeans = raob.mean(axis=0)
            self.RAOBstds = raob.std(axis=0)
            self.RAOBconstant = self.RAOBstds == 0
            self.RAOBstdsFixed = copy.copy(self.RAOBstds)
            self.RAOBstdsFixed[self.RAOBconstant] = 1

    def _standardizeRAP(self, rap):
        result = (rap - self.RAPmeans) / self.RAPstdsFixed
        result[:, self.RAPconstant] = 0.0
        return result

    def _unstandardizeRAP(self, rap):
        return self.RAPstds * rap + self.RAPmeans

    def _standardizeIM(self, im): 
        # only used if IM is used as input to the model
        result = (im - self.IMmeans) / self.IMstdsFixed
        result[:, self.IMconstant] = 0.0
        return result

    def _unstandardizeIM(self, im):
        # only used if IM is used as input to the model
        return self.IMstds * im + self.IMmeans
    
    def _standardizeRAOB(self, raob):
        result = (raob - self.RAOBmeans) / self.RAOBstdsFixed
        result[:, self.RAOBconstant] = 0.0
        return result

    def _unstandardizeRAOB(self, raob):
        return self.RAOBstds * raob + self.RAOBmeans

    def train(self, rap, im, raob, n_epochs, batch_size, method='sgd',
              verbose=False, learning_rate=0.001, validation=None, loss_f='MSE'):
        """Use Keras Functional API to train model"""

        assert ((len(validation) == 3)), f'Validation must be (rap im, raob) dimensions, {len(validation)}'
        
        self._set_seed()
        self._setup_standardize(rap, im, raob)
        
        rap  = self._standardizeRAP(rap)
        if self.n_im_inputs is not None:
            im = self._standardizeIM(im)
        raob = self._standardizeRAOB(raob)

        if validation:
            try:
                if self.n_im_inputs is not None:
                    inputs = {'rap': self._standardizeRAP(validation[0]), 
                              'im': self._standardizeIM(validation[1])}
                else:
                    inputs = {'rap': self._standardizeRAP(validation[0])}
                validation = (inputs, self._standardizeRAOB(validation[2]))
            except:
                raise TypeError(
                    f'validation must be of the following shape: (rap, im, raob)')

        try:
            if method == 'sgd':
                algo = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            elif method == 'adam':
                algo = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        except:
            raise Exception(
                "train: method={method} not one of 'scg' or 'adam'")
            
        if loss_f == 'MSE': # default
            loss = tf.keras.losses.MSE
        elif loss_f == 'MAE':
            loss = tf.keras.losses.MAE
        elif loss_f == 'Huber':
            loss = tf.keras.losses.Huber()
        elif loss_f == 'WMSE':
            loss = [lambda y_true,y_pred: weighted_mean_squared_error(y_true, y_pred, scale=2, lmda=0.03)]
        elif loss_f == 'SMSE':
            loss = seperated_mean_squared_error
        else: # custom loss function
            loss = loss_f
            
            
        self.model.compile(optimizer=algo, loss=loss, # Huber #LogCosh
                           metrics=[tf.keras.metrics.RootMeanSquaredError(),
                                    tf.keras.metrics.MeanSquaredError(),
                                    tf.keras.metrics.MeanAbsoluteError()])

        callback = [] 
        # [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-1, patience=10)] if validation is not None else []
        if verbose:
            callback.append(callbacks.TrainLogger(n_epochs, step=n_epochs//5))

        start_time = time.time()
        inputs = {'rap': rap, 'im': im} if self.n_im_inputs is not None else {'rap': rap}
        self.history = self.model.fit(inputs, raob, batch_size=batch_size, epochs=n_epochs, 
                                      verbose=0, callbacks=callback, validation_data=validation).history
        self.training_time = time.time() - start_time
        return self

    def use(self, X):
        """
        Inputs:
            X : {'rap': rap, 'im': im}
        """
        # Set to error logging after model is trained
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        rap = self._standardizeRAP(X['rap'])
        if self.n_im_inputs is not None:
            im = self._standardizeIM(X['im'])
        inputs = {'rap': rap, 'im': im} if self.n_im_inputs is not None else {'rap': rap}
        Y = self._unstandardizeRAOB(self.model.predict(inputs))
        return Y

    def save(self, path):
        self.model.save(path)
        del self.model
        with open(path + '/class.pickle', 'wb') as f:
            pickle.dump(self, f)
        self.model = tf.keras.models.load_model(path)


        
class MultiConvolutionalNeuralNetwork():
    """Convolutional Neural Network with multiple inputs (optionally).
    Specify, `n_im_inputs=None` to not add additional inputs.
    """
    def __init__(self, n_rap_inputs, n_im_inputs, n_hiddens_list,
                 n_units_in_conv_layers, kernels_size_and_stride,
                 n_outputs, rap_activation='relu', dense_activation='tanh', 
                 batchnorm=False, dropout=False, regularization=False, seed=None):
        
        if n_im_inputs is not None:
            assert ((len(n_im_inputs) == 3)), f'Image must be HxWxC dimensions, {n_im_inputs}'
        assert (isinstance(n_hiddens_list, list)), f'{type(self).__name__}: n_hiddens_list must be a list.'
        assert (isinstance(n_units_in_conv_layers, list)), f'{type(self).__name__}: n_units_in_conv_layers must be a list.'
        assert (isinstance(kernels_size_and_stride, list)), f'{type(self).__name__}: kernels_size_and_stride must be a list.'

        self.seed = seed
        self._set_seed()
        tf.keras.backend.clear_session()

        self.n_rap_inputs = n_rap_inputs
        self.n_im_inputs  = n_im_inputs
        self.n_hiddens_list = n_hiddens_list
        self.n_units_in_conv_layers  = n_units_in_conv_layers
        self.kernels_size_and_stride = kernels_size_and_stride
        self.n_outputs = n_outputs
        
        kernel_regularizer = tf.keras.regularizers.l2(0.001) if regularization else None
        
        # RAP Input
        X1 = Z1 = tf.keras.Input(shape=n_rap_inputs, name='rap')

        for (kernel, stride), units in zip(kernels_size_and_stride, n_units_in_conv_layers):
            Z1 = tf.keras.layers.Conv1D(units, kernel_size=kernel, kernel_regularizer=kernel_regularizer,
                                        strides=stride, padding='same')(Z1)
            if batchnorm:
                Z1 = tf.keras.layers.BatchNormalization()(Z1)
            Z1 = tf.keras.layers.Activation(rap_activation)(Z1)
            Z1 = tf.keras.layers.MaxPooling1D(pool_size=2)(Z1)
        Z1 = tf.keras.layers.Flatten()(Z1)

        # IM Input
        if self.n_im_inputs is not None:
            X2 = tf.keras.Input(shape=n_im_inputs, name='im')
            Z2 = tf.keras.layers.Flatten()(X2)
            Z  = tf.keras.layers.Concatenate(axis=1)([Z1, Z2]) # Join IM & RAP
            inputs = [X1, X2]
        else:
            Z = Z1
            inputs = X1
                
        # Dense Layers 
        if not (n_hiddens_list == [] or n_hiddens_list == [0]):
            for units in n_hiddens_list:
                if dropout:
                    Z = tf.keras.layers.Dropout(0.20)(Z) 
                Z = tf.keras.layers.Dense(units, activation=dense_activation, kernel_regularizer=kernel_regularizer)(Z) 
        if dropout:
            Z = tf.keras.layers.Dropout(0.20)(Z) 
        Y = tf.keras.layers.Dense(n_outputs, name='out', kernel_regularizer=kernel_regularizer)(Z)
        self.model = tf.keras.Model(inputs=inputs, outputs=Y)            

        self.RAPmeans  = None
        self.RAPstds   = None
        self.IMmeans   = None
        self.IMstds    = None
        self.RAOBmeans = None
        self.RAOBstds  = None

        self.history = None
        self.training_time = None

    def __repr__(self):
        str = f'{type(self).__name__}({self.n_outputs})'
        if self.history:
            str += f"\n  Final objective value is {self.history['loss'][-1]:.5f} in {self.training_time:.4f} seconds."
        else:
            str += '  Network is not trained.'
        return str

    def _set_seed(self):
        if self.seed:
            np.random.seed(self.seed)
            random.seed(self.seed)
            tf.random.set_seed(self.seed)

    def _setup_standardize(self, rap, im, raob):
        if self.RAPmeans is None:
            self.RAPmeans = rap.mean(axis=0)
            self.RAPstds = rap.std(axis=0)
            self.RAPconstant = self.RAPstds == 0
            self.RAPstdsFixed = copy.copy(self.RAPstds)
            self.RAPstdsFixed[self.RAPconstant] = 1

        if self.IMmeans is None and self.n_im_inputs is not None:
            # check if IM is used as input to the model
            self.IMmeans = im.mean(axis=0)
            self.IMstds = im.std(axis=0)
            self.IMconstant = self.IMstds == 0
            self.IMstdsFixed = copy.copy(self.IMstds)
            self.IMstdsFixed[self.IMconstant] = 1
            
        if self.RAOBmeans is None:
            self.RAOBmeans = raob.mean(axis=0)
            self.RAOBstds = raob.std(axis=0)
            self.RAOBconstant = self.RAOBstds == 0
            self.RAOBstdsFixed = copy.copy(self.RAOBstds)
            self.RAOBstdsFixed[self.RAOBconstant] = 1

    def _standardizeRAP(self, rap):
        result = (rap - self.RAPmeans) / self.RAPstdsFixed
        result[:, self.RAPconstant] = 0.0
        return result

    def _unstandardizeRAP(self, rap):
        return self.RAPstds * rap + self.RAPmeans

    def _standardizeIM(self, im): 
        # only used if IM is used as input to the model
        result = (im - self.IMmeans) / self.IMstdsFixed
        result[:, self.IMconstant] = 0.0
        return result

    def _unstandardizeIM(self, im):
        # only used if IM is used as input to the model
        return self.IMstds * im + self.IMmeans
    
    def _standardizeRAOB(self, raob):
        result = (raob - self.RAOBmeans) / self.RAOBstdsFixed
        result[:, self.RAOBconstant] = 0.0
        return result

    def _unstandardizeRAOB(self, raob):
        return self.RAOBstds * raob + self.RAOBmeans

    def train(self, rap, im, raob, n_epochs, batch_size, method='sgd',
              verbose=False, learning_rate=0.001, validation=None, loss_f='MSE'):
        """Use Keras Functional API to train model"""

        assert ((len(validation) == 3)), f'Validation must be (rap im, raob) dimensions, {len(validation)}'
        
        self._set_seed()
        self._setup_standardize(rap, im, raob)
        
        rap  = self._standardizeRAP(rap)
        if self.n_im_inputs is not None:
            im = self._standardizeIM(im)
        raob = self._standardizeRAOB(raob)

        if validation:
            try:
                if self.n_im_inputs is not None:
                    inputs = {'rap': self._standardizeRAP(validation[0]), 
                              'im': self._standardizeIM(validation[1])}
                else:
                    inputs = {'rap': self._standardizeRAP(validation[0])}
                validation = (inputs, self._standardizeRAOB(validation[2]))
            except:
                raise TypeError(
                    f'validation must be of the following shape: (rap, im, raob)')

        try:
            if method == 'sgd':
                algo = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            elif method == 'adam':
                algo = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        except:
            raise Exception(
                "train: method={method} not one of 'scg' or 'adam'")

        if loss_f == 'MSE': # default
            loss = tf.keras.losses.MSE
        elif loss_f == 'MAE':
            loss = tf.keras.losses.MAE
        else: # custom loss function
            loss = loss_f
            
        self.model.compile(optimizer=algo, loss=loss,
                           metrics=[tf.keras.metrics.RootMeanSquaredError(),
                                    tf.keras.metrics.MeanSquaredError(),
                                    tf.keras.metrics.MeanAbsoluteError()])

        callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10)] if validation is not None else []
        if verbose:
            callback.append(callbacks.TrainLogger(n_epochs, step=n_epochs//5))

        start_time = time.time()
        inputs = {'rap': rap, 'im': im} if self.n_im_inputs is not None else {'rap': rap}
        self.history = self.model.fit(inputs, {'out': raob}, batch_size=batch_size, epochs=n_epochs, 
                                      verbose=0, callbacks=callback, validation_data=validation).history
        self.training_time = time.time() - start_time
        return self

    def use(self, X):
        """
        Inputs:
            X : {'rap': rap, 'im': im}
        """
        # Set to error logging after model is trained
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        rap = self._standardizeRAP(X['rap'])
        if self.n_im_inputs is not None:
            im = self._standardizeIM(X['im'])
        inputs = {'rap': rap, 'im': im} if self.n_im_inputs is not None else {'rap': rap}
        Y = self._unstandardizeRAOB(self.model.predict(inputs))
        return Y

    def save(self, path):
        self.model.save(path)
        del self.model
        with open(path + '/class.pickle', 'wb') as f:
            pickle.dump(self, f)
        self.model = tf.keras.models.load_model(path)
