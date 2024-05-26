import torch

import typing

import lightning

from src.model.utils.typing import FPTensor

class FormerModule(lightning.LightningModule):
    Batch = typing.Dict[str, torch.Tensor]

    def __init__(self,
                 loss: torch.nn.Module,
                 model: torch.nn.Module,
                 empty_every: int = 1024) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.__model, self.__loss = model, loss
        self.__empty_every = max(1, empty_every)

    @property
    def loss(self) -> torch.nn.Module:
        return self.__loss

    @property
    def model(self) -> torch.nn.Module:
        return self.__model
    
    @property
    def empty_every(self) -> int:
        return self.__empty_every
    
    def __predict_next(self, batch: Batch) -> FPTensor:
        input: FPTensor = batch["sequence"]
        preds: FPTensor = self.model(input)
        assert preds.size() == input.size()
        return preds

    def __compute_loss(self, batch: Batch, preds: FPTensor) -> FPTensor:
        gtrs: FPTensor = batch["groundtruth"]
        output = self.loss(preds, gtrs)
        assert output.size() == (1,)
        return output
    
    def empty_if_needed(self, batch_idx: int) -> None:
        if torch.cuda.is_available():
            if batch_idx % self.empty_every == 0:
                torch.cuda.empty_cache()

    def training_step(self, batch: Batch, batch_idx: int) -> FPTensor:
        self.empty_if_needed(batch_idx)
        preds = self.__predict_next(batch)
        loss = self.__compute_loss(batch, preds)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
