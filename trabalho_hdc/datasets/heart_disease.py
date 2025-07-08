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

    def __init__(self):
        self.load_uci_repo()          
        self.gen_class_ids()