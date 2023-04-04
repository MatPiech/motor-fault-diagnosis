from typing import Optional, List

import pytorch_lightning as pl
import torch
import torch.nn.functional
import torchmetrics
import timm


class Classifier(pl.LightningModule):
    def __init__(self,
                 model_name: str,
                 input_channels: int,
                 classes: List[str],
                 loss_function: str,
                 lr: float,
                 lr_patience: int
                 ):
        super().__init__()

        self._model_name = model_name
        self._input_channels = input_channels
        self._classes = classes
        self._loss_function = loss_function
        self._lr = lr
        self._lr_patience = lr_patience

        self.network = timm.create_model(self._model_name, pretrained=False, num_classes=len(self._classes), in_chans=self._input_channels)

        if loss_function == 'CrossEntropy':
            self.loss = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f'Unsupported loss function: {loss_function}')

        metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(num_classes=len(self._classes), average='weighted'),
            torchmetrics.F1Score(num_classes=len(self._classes), average='weighted'),
            torchmetrics.Precision(num_classes=len(self._classes), average='weighted'),
            torchmetrics.Recall(num_classes=len(self._classes), average='weighted')
        ])
        self.train_metrics = metrics.clone('train_')
        self.valid_metrics = metrics.clone('val_')
        self.test_metrics = metrics.clone('test_')

        self.save_hyperparameters()

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Optional[torch.Tensor]:
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        if torch.isinf(loss):
            return None

        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(self.train_metrics(y_pred, y))

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x, y = batch
        y_pred = self.forward(x)

        loss = self.loss(y_pred, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(self.valid_metrics(y_pred, y))

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        y_pred = self.forward(x)

        loss = self.loss(y_pred, y)

        self.log('test_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(self.test_metrics(y_pred, y))

    def predict_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch
        y_pred = self.forward(x)

        return y, y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self._lr)
        reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                                          patience=self._lr_patience, min_lr=1e-6,
                                                                          verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': reduce_lr_on_plateau,
            'monitor': 'train_loss_epoch'
        }
