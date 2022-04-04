from typing import Dict, List, Union

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .aml_dataset import AMLDataset


class AMLDataModule(LightningDataModule):
    """
    Manages the model datasets
    """

    def __init__(
        self, 
        sequence_length: int, 
        overlap: int,
        batch_size: int = 1,
        dataset_config_path: str = 'settings/dataset_settings.yaml'
        ):
        super().__init__()
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.batch_size = batch_size
        self.dataset_config_path = dataset_config_path
        self.config = utils.load_config(self.dataset_config_path)

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """
        Returns the training DataLoader
        """
        train_dataset = AMLDataset(
            dataset_name='train',
            sequence_length=self.sequence_length,
            overlap=self.overlap
        )
        return DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=int(self.config.NUM_WORKERS)
            )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """
        Returns the validation DataLoader
        that will be used at the end of each epoch
        """
        test_dataloader = AMLDataset(
            dataset_name='test',
            sequence_length=self.sequence_length,
            overlap=self.overlap
        )
        return DataLoader(
            test_dataloader, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=int(self.config.NUM_WORKERS)
            )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        raise NotImplementedError