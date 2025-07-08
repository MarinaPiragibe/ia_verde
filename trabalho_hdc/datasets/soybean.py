from pandas import read_csv
from binhd.datasets.dataset import Dataset

class Soybean(Dataset):
    name = "zoo"
    id = 111
    numeric_features = []
    categorical_features = [
    "hair", "feathers", "eggs", "milk", "airborne", "aquatic",
    "predator", "toothed", "backbone", "breathes", "venomous",
    "fins", "legs", "tail", "domestic", "catsize"
]

    def __init__(self):
        self.load_uci_repo()           
        self.gen_class_ids()