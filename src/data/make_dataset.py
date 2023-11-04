import urllib.request
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import os

def train_test_split(df: pd.DataFrame, train_ratio: float = 0.8, shuffle = False):
    ...
    
def read_data(path: str, **pd_kwargs):
    ...

def save_data(path: str):
    ...

if __name__ == '__main__':
    ...