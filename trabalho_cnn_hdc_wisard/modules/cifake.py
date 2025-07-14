import numpy as np
from binhd.datasets.dataset import Dataset
import pandas as pd



class Cifake(Dataset):
    name = "cifake"
    def __init__(self):
        # Carrega features e labels
        self.features = np.load('features.npy')
        self.labels = np.load('labels.npy')

        self.samples = []
        self.categorical_features = []

        self.features = pd.DataFrame(self.features, columns=[f'feat_{i}' for i in range(self.features.shape[1])])

        self.samples = [
            {**{f'feat_{i}': self.features[idx, i] for i in range(self.features.shape[1])},
            'class': int(self.labels[idx])}
            for idx in range(len(self.labels))
        ]

        self.numeric_features = [f'feat_{i}' for i in range(self.features.shape[1])]

        # Adiciona: cria targets e target_col para compatibilizar
        self.targets = pd.DataFrame(self.samples)
        self.target_col = 'class'


        # Gera class ids
        self.gen_class_ids()
