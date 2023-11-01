import urllib.request
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import os

data_dir = Path(__file__).parent.parent.joinpath("data")

def download_from_url(url: str, path: str = data_dir.joinpath("external")):
    urllib.request.urlretrieve(path)

class ParaNMTDataset(Dataset):
    def __init__(self, path: Path, **pd_kwargs):
        self.data = pd.read_csv(path, pd_kwargs)
    
    def _preprocess(data):
        self.texts = self.data[]
        
        