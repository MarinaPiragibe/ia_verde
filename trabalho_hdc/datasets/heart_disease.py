from pandas import read_csv
from binhd.datasets.dataset import Dataset

class HeartDisease(Dataset):
    name = "heart_disease"
    id = 45
    numeric_features = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
        "ca"
    ]
    categorical_features = [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "thal"
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