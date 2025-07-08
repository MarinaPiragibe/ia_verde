from pandas import read_csv
from binhd.datasets.dataset import Dataset

class Dermatology(Dataset):
    name = "dermatology"
    id = 33
    numeric_features = ['erythema', 'scaling', 'definite-borders', 'itching',
       'koebner phenomenon', 'polygonal papules', 'follicular papules',
       'oral-mucosal involvement', 'knee elbow involvement',
       'scalp involvement', 'melanin incontinence',
       'eosinophils in the infiltrate', 'pnl infiltrate',
       'fibrosis of the papillary dermis', 'exocytosis', 'acanthosis',
       'hyperkeratosis', 'parakeratosis', 'clubbing of the rete ridges',
       'elongation of the rete ridges',
       'thinning of the suprapapillary epidermis', 'spongiform pustule',
       'munro microabcess', 'focal hypergranulosis',
       'disappearance of the granular layer',
       'vacuolisation and damage of the basal layer', 'spongiosis',
       'saw-tooth appearance of retes', 'follicular horn plug',
       'perifollicular parakeratosis', 'inflammatory monoluclear infiltrate',
       'band-like infiltrate', 'age']
    categorical_features = ['family history'
    ]

    def __init__(self):
        self.load_uci_repo()           
        self.gen_class_ids()