# cira-ml-soundings
Using Machine Learning to Improve Vertical Profiles of Temperature and Moisture for Severe Weather Nowcasting

## Preprocessing

The preprocessing driver can be found under [soundings/preprocessing/preprocess.py](https://github.com/stockeh/cira-ml-soundings/blob/master/soundings/preprocessing/preprocess.py). This file will use the details specified under [soundings/preprocessing/config.yaml](https://github.com/stockeh/cira-ml-soundings/blob/master/soundings/preprocessing/config.yaml) to extract the GOES, RAOB, RTMA and RAP files into a single (per-sample) NetCDF output file. Once configured, this file can be run as:

```python
python -m soundings.preprocessing.preprocess -c ./soundings/preprocessing/config.yaml
```

Once all `.nc` files are saved to disk, we can load all the samples then save/load them as a single numpy `.npz` file:

```python
# save files once
raob, rap, goes, rtma, sonde_files = soundings.preprocessing.dataloader.
  load_preprocessed_samples(processed_vol, noaa=True, sgp=False, shuffle=False)
f = '/mnt/data1/stock/mlsoundings/preprocessed.npz'
np.savez(f, raob=raob, rap=rap, goes=goes, rtma=rtma, sonde_files=sonde_files)

# some additional filtering (nans, location, etc.)
...

# load files with numpy from here on
container = np.load(f)
raob = container['raob']
rap  = container['rap']
goes = container['goes']
rtma = container['rtma']
sonde_files = container['sonde_files']
```

## Neural Networks

TensorFlow implementations of all the networks are found under [soundings/deep_learning/tf_neuralnetwork.py](master/soundings/deep_learning/tf_neuralnetwork.py), where different networks can be initialized, trained, and used. This includes:

- `NeuralNetwork`
- `ConvolutionalNeuralNetwork`
- `ConvolutionalAutoEncoder`
- `SkipNeuralNetwork`
- `MultiConvolutionalNeuralNetwork`

Example usage:

```python
X = np.arange(100).reshape((-1, 1))
T = np.sin(X * 0.05)

n_hiddens_list = [10, 10]

nnet = nn.NeuralNetwork(X.shape[1], n_hiddens_list, T.shape[1])
nnet.train(X, T, n_epochs=1000, batch_size=32, learning_rate=0.01, method='sgd')
Y = nnet.use(X)
```

### Running Experiments

Experiments are run using the [soundings/experiments/driver.py](https://github.com/stockeh/cira-ml-soundings/blob/master/soundings/experiments/driver.py) file, such that the preprocessed data is loaded, networks are trained according to the configuration under [soundings/experiments/configs/\*](https://github.com/stockeh/cira-ml-soundings/tree/master/soundings/experiments/configs), and results are saved as a `.csv` file with nnet details and metrics. The configuration files should be configured to specify the model and data parameters. Once configured, this file can be run as:

```python 
python -u -m soundings.experiments.driver -c ./soundings/experiments/configs/__config__.yaml
```

## Visualization

The [soundings/plotting/radiosonde_plotting.py](https://github.com/stockeh/cira-ml-soundings/blob/master/soundings/plotting/radiosonde_plotting.py) contains the primary code to plot the atmospheric sounding with the RAOB, RAP, and ML estimates. The error between the ML RMSE and RAP RMSE can be plotted using [soundings/plotting/results.py](https://github.com/stockeh/cira-ml-soundings/blob/master/soundings/plotting/results.py).

Examples for visualizations from the experiments can be found under the [notebooks/results.ipynb](https://github.com/stockeh/cira-ml-soundings/blob/master/notebooks/results.ipynb) Jupyter Notebook.
