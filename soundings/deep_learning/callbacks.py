import tensorflow as tf

class TrainLogger(tf.keras.callbacks.Callback):

    def __init__(self, n_epochs, step=10):
        self.step = step
        self.n_epochs = n_epochs

    def on_epoch_end(self, epoch, logs=None):
        s = (f"epoch: {epoch}, loss: {logs['loss']:7.5f}, mse {logs['mse']:7.5f}")
             #rmse {logs['rmse']:7.5f}, sfc_rmse {logs['sfc_rmse']:7.5f}, "
             #f"val_rmse {logs['val_rmse']:7.5f}, val_sfc_rmse {logs['val_sfc_rmse']:7.5f}")
        if epoch % self.step == 0:
            print(s)
        elif epoch + 1 == self.n_epochs:
            print(s)
            print('finished!')