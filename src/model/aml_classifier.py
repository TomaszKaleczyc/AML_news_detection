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
        self.predictor = nn.Linear(
            self.config.OVERARCHING_NETWORK_OUTPUT_SIZE, 
            self.num_classes
            )

    def forward(self, article_part_tokens):
        """
        Defines the model forward pass
        """
        output_embeddings = []
        # TODO: turn this into a single tensor for efficiency:
        for article_part_token in article_part_tokens:
            article_part_features = self.feature_extractor(
                article_part_token['input_ids'].squeeze(1),
                attention_mask=article_part_token['attention_mask'].squeeze(1),
                token_type_ids=article_part_token['token_type_ids'].squeeze(1)
                )
            output_embeddings.append(article_part_features['pooler_output'])
        output_embeddings = torch.cat(output_embeddings).unsqueeze(0)
        _, (h_t, h_c) = self.lstm(output_embeddings)
        return self.predictor(h_t.view(-1, 100))
