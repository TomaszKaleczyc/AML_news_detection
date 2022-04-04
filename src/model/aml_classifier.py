import numpy as np

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from transformers import BertModel

from utilities import utils


class AMLClassifier(LightningModule):
    """
    RoBERT model for AML article classification
    """

    def __init__(
            self, 
            num_classes: int = 3,
            dropout_rate: float = 0.5,
            model_config_path: str = 'settings/model_settings.yaml'
        ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.config = utils.load_config(model_config_path)

        # define the model architecture:
        self._setup_feature_extractor()
        self._setup_overarching_network()
        self._setup_predictor()

    def _setup_feature_extractor(self):
        """
        Defines the base feature extraction model
        """
        self.feature_extractor = BertModel.from_pretrained(self.config.BERT_MODEL)
        
    def _setup_overarching_network(self):
        """
        Defines the feature collating network
        """
        self.lstm = nn.LSTM(
            self.config.BERT_MODEL_OUTPUT_SIZE, 
            self.config.OVERARCHING_NETWORK_OUTPUT_SIZE,
            num_layers=1,
            bidirectional=False
            )

    def _setup_predictor(self):
        """
        Defines the model predictor
        """
        self.predictor = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(
                self.config.OVERARCHING_NETWORK_OUTPUT_SIZE, 30
            ),
            nn.ReLU(),
            nn.Linear(30, self.num_classes)
        ) 

    def forward(self, article_batch):
        """
        Defines the model forward pass
        """
        output_embeddings = []
        # TODO: refactor to allow batch processing:
        article_part_features = self.feature_extractor(
            article_batch['input_ids'].squeeze(0),
            attention_mask=article_batch['attention_mask'].squeeze(0),
            token_type_ids=article_batch['token_type_ids'].squeeze(0)
            )
        output_embeddings = article_part_features['pooler_output'].unsqueeze(0)
        lstm_outputs, _ = self.lstm(output_embeddings)
        article_idxs = np.arange(len(lstm_outputs))
        last_time_step_idx = article_batch['num_splits'] - 1
        last_time_step = lstm_outputs[article_idxs, last_time_step_idx]
        return self.predictor(last_time_step)
