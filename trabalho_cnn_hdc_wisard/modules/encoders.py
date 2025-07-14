import torch
import torch.nn as nn
from torchhd import embeddings
from binhd.embeddings import ScatterCode
import torchhd

class RecordEncoder(nn.Module):
    def __init__(self, out_features, size, levels, low, high):
        super(RecordEncoder, self).__init__() 
        self.position = embeddings.Random(size, out_features, vsa="BSC", dtype=torch.uint8)
        self.value = ScatterCode(levels, out_features, low = low, high = high)
    
    def forward(self, x):
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        return sample_hv

class NGramEncoder(nn.Module):
    def __init__(self, out_features, levels, low, high):
        super(NGramEncoder, self).__init__()
        self.value = ScatterCode(levels, out_features, low = low, high = high)              

    def forward(self, x, oper = "bind"):
        if oper == "bind":
            sample_hv = torchhd.bind_sequence(self.value(x))
        elif oper == "bundle":
            sample_hv = torchhd.bundle_sequence(self.value(x))
        return sample_hv