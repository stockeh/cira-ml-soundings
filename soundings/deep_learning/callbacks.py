import tensorflow as tf

class TrainLogger(tf.keras.callbacks.Callback):

    def __init__(self, n_epochs, step=10):
        self.step = step
        self.n_epochs = n_epochs

    def on_epoch_end(self, epoch, logs=None):

        s = f"epoch: {epoch}, rmse {logs['root_mean_squared_error']:7.5f}"
        if 'val_root_mean_squared_error' in logs:
            s += f", val_rmse {logs['val_root_mean_squared_error']:7.5f}"
         #rmse {logs['rmse']:7.5f}, sfc_rmse {logs['sfc_rmse']:7.5f}, "
         #f"val_rmse {logs['val_rmse']:7.5f}, val_sfc_rmse {logs['val_sfc_rmse']:7.5f}")
        if epoch % self.step == 0:
            print(s)
        elif epoch + 1 == self.n_epochs:
            print(s)
            print('finished!')
