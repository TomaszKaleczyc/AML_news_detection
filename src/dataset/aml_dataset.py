import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from pandas import DataFrame
from tqdm.notebook import tqdm

import torch
from torch import Tensor
from torch.utils.data import Dataset
import transformers
from transformers.tokenization_utils_base import BatchEncoding

from utilities import utils


OverlappingTokens = List[List[BatchEncoding]]


class AMLDataset(Dataset):
    """
    Manages the model dataset
    """

    def __init__(
            self, 
            dataset_name: str, 
            sequence_length: int,
            overlap: int,
            dataset_config_path: str = 'settings/dataset_settings.yaml',
            verbose=True
        ):
        super().__init__()
        self.dataset_name = dataset_name
        self.verbose = verbose
        self.sequence_length = sequence_length
        self.overlap = overlap
        assert self.overlap < self.sequence_length, 'The overlap should not exceed the sequence length'
        
        self.config = utils.load_config(dataset_config_path)
        self.articles = self._get_articles()
        self.tokeniser = transformers.BertTokenizer.from_pretrained(self.config.BERT_MODEL)
        self.tokens = self._get_tokens()
        if self.verbose:
            self._summarise_dataset()

    @property
    def labels(self) -> List[int]:
        return self.articles['label'].values.tolist()

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    def _get_articles(self) -> DataFrame:
        """
        Returns the dataset DataFrame
        """
        dataset_path = Path(self.config.DATASET_PATH)
        articles = pd.read_csv(dataset_path/f'{self.dataset_name}.csv', index_col=0)
        articles['label'] = articles[self.config.CLASS_COLUMN_NAME].replace(self.config.CLASS_MAPPING)
        return articles

    def _summarise_dataset(self):
        """
        Displays the dataset summary
        """
        print('='*60)
        print(f'Loaded dataset {self.dataset_name}')
        print('Number of tokens:', len(self.tokens))
        print('Number of classes:', self.num_classes)

    def _get_tokens(self) -> OverlappingTokens:
        """
        Returns the list of model tokens
        """
        article_contents = self.articles[self.config.ARTICLE_CONTENTS_COLUMN_NAME]
        tokens = []
        print('Tokenisation of content strips:')
        for article_content in tqdm(article_contents):
            overlapping_content = self._get_overlapping_contents(article_content)
            article_tokens = self._get_article_tokens(overlapping_content)
            tokens.append(article_tokens)
        return tokens

    def _get_overlapping_contents(self, article_contents: str) -> List[str]:
        """
        Returns list of article parts with overlap
        as described in the RoBERT paper
        """
        output = []
        clean_article_contents = self._clean_article_contents(article_contents)
        separated_content = clean_article_contents.split()
        sequence_step = (self.sequence_length - self.overlap)
        num_splits = len(separated_content) // sequence_step  + 1
        for split_number in range(num_splits):
            split_start = split_number * sequence_step
            split = separated_content[
                split_start : split_start + self.sequence_length
                ]
            output.append(' '.join(split))
        return output

    def _get_article_tokens(self, overlapping_content: List[str]) -> Dict[str, Any]:
        """
        Returns all article part tokens in a single dictionary
        """
        article_tokens_list = [
            self.tokeniser(
                article_part, 
                max_length=self.sequence_length, 
                truncation=True, 
                return_tensors="pt",
                padding="max_length"
                )
            for article_part in overlapping_content
        ]
        article_tokens = {
            'input_ids': torch.cat([atricle_token['input_ids'] for atricle_token in article_tokens_list]),
            'attention_mask': torch.cat([atricle_token['attention_mask'] for atricle_token in article_tokens_list]),
            'token_type_ids': torch.cat([atricle_token['token_type_ids'] for atricle_token in article_tokens_list]),
            'num_splits': len(article_tokens_list)
        }
        return article_tokens

    @staticmethod
    def _clean_article_contents(article_contents: str) -> str:
        """
        Returns string with alphanumeric characters removed
        """ 
        return re.sub("(\\W)+", " ", article_contents)

    def show_tokens(self, article_idx: int = 0) -> List[str]:
        """
        Returns the text version of tokens
        """
        tokens = self.tokens[article_idx]
        return [self.tokeniser.decode(token['input_ids']) for token in tokens]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Tuple[OverlappingTokens, Tensor]:
        overlapping_tokens = self.tokens[idx]
        target = torch.tensor(self.labels[idx])
        return overlapping_tokens, target
