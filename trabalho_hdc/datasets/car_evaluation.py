from pandas import read_csv
from binhd.datasets.dataset import Dataset

class CarEvaluation(Dataset):
    name = "car_evaluation"
    id = 19
    numeric_features = []
    categorical_features = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]


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