
import yaml
from urllib.request import Request, urlopen
from IPython.display import display

from bs4 import BeautifulSoup
from pandas import DataFrame
from omegaconf import DictConfig


def invert_dictionary(input_dict: dict) -> dict:
    """
    Returns dictionary with inverted keys and values
    """
    return {value: key for key, value in input_dict.items()}


def get_webpage_body(url: str, user_agent: str = 'Mozilla/5.0') -> str:
    """
    Returns the text under a given webpage body
    """
    request = Request(url, headers={'User-Agent': user_agent})
    webpage=urlopen(request).read()
    soup = BeautifulSoup(webpage, features="html.parser")
    return soup.body.get_text()


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