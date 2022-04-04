from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback


class SwitchPretrainedWeightsState(Callback):
    """
    Switches the feature extractor pretrained
    weights state between frozen/unfrozen
    """

    def on_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Conditions and actions at the start of a training epoch
        """
        switch_trigger_1 = pl_module.pretrained_weights_frozen and pl_module.current_epoch == pl_module.num_epochs_freeze_pretrained
        switch_trigger_2 = not pl_module.pretrained_weights_frozen and pl_module.current_epoch < pl_module.num_epochs_freeze_pretrained
        if switch_trigger_1 or switch_trigger_2:
            self._switch_pretrained_weights_state(pl_module)

    def _switch_pretrained_weights_state(self, pl_module: LightningModule) -> None:
        """
        Toggles the state of the pretrained weights in the feature extractor
        """
        new_state = not pl_module.pretrained_weights_frozen
        for param in pl_module.feature_extractor.parameters():
            param.requires_grad = new_state
        pl_module.pretrained_weights_frozen = new_state
        print(f'Feature extractor weights frozen: {new_state}')
        
        
        