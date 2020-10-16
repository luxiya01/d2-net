"""Custom PyTorch Dataset class for loading side scan sonar data"""

from torch.utils.data import Dataset
import sss_data_processing.src as dataproc


class SSSDataset(Dataset):
    """Side scan sonar image patches dataset"""
    def __init__(self, datadir):
        self.datadir = datadir
        self.correspondence_getter = dataproc.correspondence_getter.CorrespondenceGetter(
            self.datadir)

    def __len__(self):
        pass

    def __getitem__(self, idx):
        return None
