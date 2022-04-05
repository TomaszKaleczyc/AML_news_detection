from typing import Dict, List, Union

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .long_text_dataset import LongTextDataset
from utilities import utils


class LongTextDataModule(LightningDataModule):
    """
    Manages the model datasets
    """

    def __init__(
            self, 
            config_path: str,
            sequence_length: int, 
            overlap: int,
            batch_size: int = 1,
        ):
        super().__init__()
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.batch_size = batch_size
        self.config_path = config_path
        self.config = utils.load_config(self.config_path)

    @property
    def num_classes(self) -> int:
        return len(self.config.CLASS_MAPPING)

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """
        Returns the training DataLoader
        """
        train_dataset = LongTextDataset(
            dataset_name=self.config.TRAIN_NAME,
            sequence_length=self.sequence_length,
            overlap=self.overlap,
            config_path=self.config_path,
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
        test_dataloader = LongTextDataset(
            dataset_name=self.config.VAL_NAME,
            sequence_length=self.sequence_length,
            overlap=self.overlap,
            config_path=self.config_path,
        )
        return DataLoader(
            test_dataloader, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=int(self.config.NUM_WORKERS)
            )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        raise NotImplementedError