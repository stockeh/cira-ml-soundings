import abc
import pandas as pd

class ExperimentInterface(abc.ABC):
    
    @abc.abstractmethod
    def get_experiemnts(self, config: dict) -> (list, list):
        """"""
        raise NotImplementedError

    @abc.abstractmethod
    def run_experiments(self, config: dict, network_experiments: list,
                        data_experiments: list, data: tuple) -> pd.DataFrame:
        """"""
        raise NotImplementedError
        