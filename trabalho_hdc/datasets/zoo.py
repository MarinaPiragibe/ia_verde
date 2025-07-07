from pandas import read_csv
from binhd.datasets.dataset import Dataset

class Zoo(Dataset):
    name = "zoo"
    id = 111
    numeric_features = []
    categorical_features = [ "hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator", "toothed", "backbone"
]


    def __init__(self, path = None):
        if not path:
            # Loading dataset from uci repo
            self.load_uci_repo()           
        else: 
            names = self.numeric_features + ["class"]
            self.target_col = "class"
            data = read_csv(path, names=names)
            self.features = data[self.numeric_features]
            self.targets = data[[self.target_col]]  
            self.num_features = len(self.numeric_features)
        
        self.gen_class_ids()