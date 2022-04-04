from typing import Any, Dict, List, Tuple
import numpy as np

import torch
from torch import Tensor
import torchmetrics
import torch.nn as nn
from torch.nn import functional as F, Module
from transformers import BertModel

from pytorch_lightning import LightningModule

from utilities import utils


AMLBatch = Tuple[Dict[str, Tensor], Tensor]


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
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.config = utils.load_config(model_config_path)

        # define the model architecture:
        self._setup_feature_extractor()
        self._setup_aggregating_network()
        self._setup_predictor()

        # define evaluation metrics:
        self.accuracy = torchmetrics.Accuracy(num_classes=self.num_classes).to(self.device)

    def _setup_feature_extractor(self):
        """
        Defines the base feature extraction model
        """
        self.feature_extractor = BertModel.from_pretrained(self.config.BERT_MODEL)
        
    def _setup_aggregating_network(self):
        """
        Defines the feature collating network
        """
        self.aggregating_network = nn.LSTM(
            self.config.BERT_MODEL_OUTPUT_SIZE, 
            self.config.AGGREGATING_NETWORK_OUTPUT_SIZE,
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
                self.config.AGGREGATING_NETWORK_OUTPUT_SIZE, 30
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
        aggregated_outputs, _ = self.aggregating_network(output_embeddings)
        article_idxs = np.arange(len(aggregated_outputs))
        last_time_step_idx = article_batch['num_splits'] - 1
        last_time_step = aggregated_outputs[article_idxs, last_time_step_idx]
        return self.predictor(last_time_step)

    def _loss_step(
            self, 
            batch: AMLBatch, 
            eval: bool, 
            criterion: Module = F.cross_entropy
        ) -> Tensor:
        """
        Definition of a standard loss step
        """
        tokens, labels = batch
        logits = self(tokens)
        loss = criterion(logits, labels)
        if eval:
            self.accuracy.update(logits, labels)
        return loss

    def training_step(self, batch: AMLBatch, batch_idx: int) -> Tensor:
        loss = self._loss_step(batch, eval=False)
        self.log('train/loss', loss)
        return loss
        
    def validation_step(self, batch: AMLBatch, batch_idx: int) -> Tensor:
        loss = self._loss_step(batch, eval=True)
        self.log('val/loss', loss)

    def validation_epoch_end(self, outputs) -> None:
        accuracy = self.accuracy.compute()
        self.log('val/accuracy', accuracy)
        print('val/accuracy', accuracy)
        self.accuracy.reset()

    def configure_optimizers(self):
        parameter_groups = [
            {'params': self.feature_extractor.parameters(), 'weight_decay': float(self.config.FEATURE_EXTRACTOR_WEIGHT_DECAY)},
            {'params': self.aggregating_network.parameters(), 'weight_decay': float(self.config.AGGREGATING_NETWORK_WEIGHT_DECAY)},
            {'params': self.predictor.parameters(), 'weight_decay': float(self.config.PREDICTOR_WEIGHT_DECAY)}
        ]
        return torch.optim.Adam(parameter_groups, lr=float(self.config.LEARNING_RATE))

    def predict(self, tokens: Dict[str, Tensor]) -> List[Tuple[str, float]]:
        """
        Returns prediction results on a given batch
        """
        self.eval()
        logits = self(tokens)
        probabilities = F.softmax(logits)
        predicted_classes = probabilities.argmax(dim=1)
        output = [
            (
                self.config.OUTPUT_MAPPING[predicted_class.item()], 
                probabilities[idx][predicted_class.item()].item()
            )
            for idx, predicted_class in enumerate(predicted_classes)
        ]
        return output
