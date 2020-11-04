import numpy as np

from soundings.deep_learning import mlutilities as ml


def compute_profile_rmses(Y, T, surface_error=25):
    """
    ---
    params:
        Y : np.array (None, 256, N)
        T : np.array (None, 256, N)
    """
    rmse = np.sqrt((np.mean((Y - T)**2, axis=0)))
    mean_rmse = ml.rmse(Y, T)
    
    rmse_sfc = np.sqrt((np.mean((Y[:, :surface_error] - T[:, :surface_error])**2, axis=0)))
    mean_rmse_sfc = ml.rmse(Y[:, :surface_error], T[:, :surface_error])
    
    return rmse, mean_rmse, rmse_sfc, mean_rmse_sfc