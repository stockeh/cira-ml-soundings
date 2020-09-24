import numpy as np
import matplotlib.pyplot as plt
from soundings.deep_learning import mlutilities as ml
from soundings.plotting import radiosonde_plotting

def plot_altitude_rmse_verticle(nnet, X, T, NWP_Temp, alt, file_name=None):
    """
    Plot the RMSE over different altitudes for some NeuralNetwork architecture.

    :params
    ---
    nnet : class
        Trained Neural Network class that will be used for evaluation
    X : np.array
        Input to the trained nnet
    T : np.array
        Targets to compare to for the nnet. Will often be the temperature profile from the RAOB.
    NWP_Temp : np.array
        Temperature profile from the NWP mode. Should have the same shape T.
    alt : np.array
        Altitude profile from the RAOB
    """
    default_font = 12
    figure_width = 10
    figure_height = 6
    line_width = 2
    
    if file_name:
        default_font = 14
        figure_width = 10
        figure_height = 6
        line_width = 2.5
        
    surface_error = 25
    
    rap_color = radiosonde_plotting.DEFAULT_OPTION_DICT[radiosonde_plotting.NWP_LINE_COLOUR_KEY]
    ml_color = radiosonde_plotting.DEFAULT_OPTION_DICT[radiosonde_plotting.PREDICTED_LINE_COLOUR_KEY]
    
    fig, axs = plt.subplots(1, 2, figsize=(figure_width, figure_height))
    axs = axs.ravel()
    
    rap_rmse = np.sqrt((np.mean((NWP_Temp - T)**2, axis=0)))
    rap_mean_rmse = ml.rmse(NWP_Temp, T)
    
    axs[0].plot(rap_rmse, alt, color=rap_color, linewidth=line_width)
    axs[0].axvline(rap_mean_rmse, label=f'RAP: {rap_mean_rmse:.3f}',
                   color=rap_color, linestyle='--', linewidth=line_width)
    
    rap_rmse = np.sqrt((np.mean((NWP_Temp[:, :surface_error] - T[:, :surface_error])**2, axis=0)))
    rap_mean_rmse = ml.rmse(NWP_Temp[:, :surface_error], T[:, :surface_error])

    axs[1].plot(rap_rmse, alt[:surface_error], color=rap_color, linewidth=line_width)
    axs[1].axvline(rap_mean_rmse, label=f'RAP: {rap_mean_rmse:.3f}',
                   color=rap_color, linestyle='--', linewidth=line_width)
    
    Y = nnet.use(X)
    ml_rmse = np.sqrt((np.mean((Y - T)**2, axis=0)))
    ml_mean_rmse = ml.rmse(Y, T)
    
    axs[0].plot(ml_rmse, alt, color=ml_color, linewidth=line_width)
    axs[0].axvline(ml_mean_rmse, label=f'ML: {ml_mean_rmse:.3f}',
                   color=ml_color, linestyle='--', linewidth=line_width)
    
    ml_rmse = np.sqrt((np.mean((Y[:, :surface_error] - T[:, :surface_error])**2, axis=0)))
    ml_mean_rmse = ml.rmse(Y[:, :surface_error], T[:, :surface_error])
    
    axs[1].plot(ml_rmse, alt[:surface_error], color=ml_color, linewidth=line_width)
    axs[1].axvline(ml_mean_rmse, label=f'ML: {ml_mean_rmse:.3f}',
                   color=ml_color, linestyle='--', linewidth=line_width)

    axs[0].set_ylabel('Altitude [m]', fontsize=default_font)
    axs[0].set_xlabel('RMSE [C]', fontsize=default_font)
    axs[1].set_xlabel('RMSE [C]', fontsize=default_font)
    axs[0].legend(fontsize=default_font)
    axs[1].legend(fontsize=default_font, loc='upper right')
    
    for ax in axs:
        ax.tick_params(axis='x', labelsize=default_font)
        ax.tick_params(axis='y', labelsize=default_font)
        ax.grid(True)
    
    if file_name:
        plt.savefig(file_name, dpi=300)
        plt.show()
        plt.close()
        
    
def plot_loss(nnet):
    train_color = radiosonde_plotting.DEFAULT_OPTION_DICT[
        radiosonde_plotting.NWP_LINE_COLOUR_KEY]
    val_color = radiosonde_plotting.DEFAULT_OPTION_DICT[
        radiosonde_plotting.PREDICTED_LINE_COLOUR_KEY]
    
    fig, ax = plt.subplots(1, figsize=(8, 4))
    ax.plot(nnet.history['rmse'], color=train_color, label='train')
    ax.plot(nnet.history['val_rmse'], color=val_color, label='val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('RMSE [C]')
    ax.legend();