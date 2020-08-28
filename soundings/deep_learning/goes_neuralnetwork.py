import copy
import time
import random

import numpy as np
import tensorflow as tf

from soundings.deep_learning import callbacks

class GOESNeuralNetwork():
    def __init__(self, n_goes_inputs, n_rap_inputs, n_outputs, seed=None):
        
        assert ((len(n_goes_inputs) == 3) and 
                n_goes_inputs[-1] < n_goes_inputs[0]), 'GOES must be HxWxC dimensions'
        
        if seed:
            self.seed = seed
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)
                  
        tf.keras.backend.clear_session()

        self.n_goes_inputs = n_goes_inputs
        self.n_rap_inputs = n_rap_inputs
        self.n_outputs = n_outputs
        
        X1 = Z = tf.keras.Input(shape=n_goes_inputs, name='goes')
        Z = tf.keras.layers.Conv2D(10, kernel_size=(3,3), strides=1,
                                   activation='tanh', padding='same')(Z)
        Z = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(Z)
        Z = tf.keras.layers.Conv2D(10, kernel_size=(3,3), strides=1,
                                   activation='tanh', padding='same')(Z)
        Z = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(Z)
        
        out = tf.keras.layers.Dense(n_rap_inputs[0], activation='tanh')(tf.keras.layers.Flatten()(Z))
        out = tf.keras.layers.Reshape((out.shape[1], 1))(out)

        X2 = tf.keras.Input(shape=n_rap_inputs, name='rap')
        Z = tf.keras.layers.Concatenate(axis=2)([X2, out])

        Z = tf.keras.layers.Conv1D(1, kernel_size=10, strides=1, activation='tanh', padding='same')(Z)
        Z = tf.keras.layers.MaxPooling1D(pool_size=2)(Z)
        Y = tf.keras.layers.Dense(n_outputs, name='out')(tf.keras.layers.Flatten()(Z))

        self.model = tf.keras.Model(inputs=[X1, X2], outputs=Y)
        
        self.GOESmeans = None
        self.GOESstds = None
        self.RAPmeans = None
        self.RAPstds = None
        self.RAOBmeans = None
        self.RAOBstds = None
        
        self.history = None
        self.training_time = None

    def __repr__(self):
        str = f'{type(self).__name__}({self.n_outputs})'
        if self.history:
            str += f"\n  Final objective value is {self.history['loss'][-1]:.5f} in {self.training_time:.4f} seconds."
        else:
            str += '  Network is not trained.'
        return str

    def _setup_standardize(self, goes, rap, raob):
        if self.GOESmeans is None:
            self.GOESmeans = goes.mean(axis=0)
            self.GOESstds = goes.std(axis=0)
            self.GOESconstant = self.GOESstds == 0
            self.GOESstdsFixed = copy.copy(self.GOESstds)
            self.GOESstdsFixed[self.GOESconstant] = 1

        if self.RAPmeans is None:
            self.RAPmeans = rap.mean(axis=0)
            self.RAPstds = rap.std(axis=0)
            self.RAPconstant = self.RAPstds == 0
            self.RAPstdsFixed = copy.copy(self.RAPstds)
            self.RAPstdsFixed[self.RAPconstant] = 1
            
        if self.RAOBmeans is None:
            self.RAOBmeans = raob.mean(axis=0)
            self.RAOBstds = raob.std(axis=0)
            self.RAOBconstant = self.RAOBstds == 0
            self.RAOBstdsFixed = copy.copy(self.RAOBstds)
            self.RAOBstdsFixed[self.RAOBconstant] = 1

    def _standardizeGOES(self, goes):
        result = (goes - self.GOESmeans) / self.GOESstdsFixed
        result[:, self.GOESconstant] = 0.0
        return result

    def _unstandardizeGOES(self, goes):
        return self.GOESstds * goes + self.GOESmeans

    def _standardizeRAP(self, rap):
        result = (rap - self.RAPmeans) / self.RAPstdsFixed
        result[:, self.RAPconstant] = 0.0
        return result

    def _unstandardizeRAP(self, rap):
        return self.RAPstds * rap + self.RAPmeans

    def _standardizeRAOB(self, raob):
        result = (raob - self.RAOBmeans) / self.RAOBstdsFixed
        result[:, self.RAOBconstant] = 0.0
        return result

    def _unstandardizeRAOB(self, raob):
        return self.RAOBstds * raob + self.RAOBmeans
    
    def train(self, goes, rap, raob, n_epochs, batch_size, method='sgd',
              verbose=False, learning_rate=0.001, validation=None, loss_f=None):
        """Use Keras Functional API to train model"""

        self._setup_standardize(goes, rap, raob)
        
        goes = self._standardizeGOES(goes)
        rap  = self._standardizeRAP(rap)
        roab = self._standardizeRAOB(raob)
        
        if validation is not None:
            try:
                # TODO: Implement this
                validation = None
            except:
                raise TypeError(
                    f'validation must be of the following shape: (goes, rap, raob)')

        try:
            if method == 'sgd':
                algo = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            elif method == 'adam':
                algo = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        except:
            raise Exception(
                "train: method={method} not one of 'scg' or 'adam'")

        loss = tf.keras.losses.MSE if loss_f == None else loss_f
        self.model.compile(optimizer=algo, loss=loss,
                           metrics=[tf.keras.metrics.RootMeanSquaredError()])

        callback = [callbacks.TrainLogger(n_epochs, step=5)] if verbose else None
        start_time = time.time()
        self.history = self.model.fit({'goes': goes, 'rap': rap}, {'out': raob},
                                      batch_size=batch_size, epochs=n_epochs, verbose=0,
                                      callbacks=callback, validation_data=validation).history
        self.training_time = time.time() - start_time
        return self

    def use(self, goes, rap):
        # Set to error logging after model is trained
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        goes = self._standardizeGOES(goes)
        rap  = self._standardizeRAP(rap)
#         Y = self.model.predict({'goes': goes, 'rap': rap})
        Y = self._unstandardizeRAOB(self.model.predict({'goes': goes, 'rap': rap}))
        return Y