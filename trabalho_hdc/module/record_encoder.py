import torch
import torch.nn as nn
from torchhd import embeddings
from binhd.embeddings import ScatterCode
import torchhd
from binhd.functional import multibundle

class RecordEncoder(nn.Module):
    def __init__(self, out_features, size, num_levels):
        super(RecordEncoder, self).__init__()
        # Create random position vectors for each feature (field)
        self.position = embeddings.Random(size, out_features, vsa="BSC", dtype=torch.int8)
        self.value = ScatterCode(num_levels, out_features)

    
    def forward(self, x):
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        return sample_hv