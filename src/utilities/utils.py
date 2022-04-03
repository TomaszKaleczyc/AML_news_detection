
import yaml
from IPython.display import display
from omegaconf import DictConfig

from pandas import DataFrame


def df_summarise(df: DataFrame) -> None:
    """
    Displays the summary of a given DataFrame
    """
    display(df.shape)
    display(df.columns)
    display(df.head())

def load_config(config_path: str) -> dict:
    """
    Returns DictConfig from a given yaml path
    """
    with open(config_path, 'r') as config:
        config_dict = yaml.safe_load(config)
    return DictConfig(config_dict)