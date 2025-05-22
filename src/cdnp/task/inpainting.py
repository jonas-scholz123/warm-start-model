from torch.utils.data import Dataset

class InpaintingDataset(Dataset):
    def __init__(self, delegate: Dataset):
