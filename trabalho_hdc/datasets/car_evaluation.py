from pandas import read_csv
from binhd.datasets.dataset import Dataset

class CarEvaluation(Dataset):
    name = "car_evaluation"
    id = 19
    numeric_features = []
    categorical_features = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]


    def __init__(self):
        self.load_uci_repo()          
        self.gen_class_ids()