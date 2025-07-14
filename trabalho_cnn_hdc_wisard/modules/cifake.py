from binhd.datasets.dataset import Dataset
from pandas import read_csv

class Cifake(Dataset):
    name = "cifake"

    def set_numeric_features(self, numeric_features):
        self.numeric_features = numeric_features

    def set_numeric_features(self, numeric_features):
        self.numeric_features = numeric_features
        
    def __init__(self, features, targets, path = None):
        self.target_col = "class"
        self.features = features
        self.targets = targets
        self.numeric_features = features
        self.categorical_features = []
        self.num_features = len(self.features)
        self.gen_class_ids()