import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from model_training.training_utils import get_optimizer, get_loss
from utils.metrics import get_accuracy, get_f1, get_auc


class LitClassifier(pl.LightningModule):
    def __init__(self, model, config, **kwargs):
        super().__init__()
        self.loss = None
        self.optim = None
        self.model = model
        self.config = config

    def forward(self, x):
        x = self.model(x)
        return x

    def default_step(self, x, y, stage):
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log_dict(
            {f"{stage}_loss": loss,
             f"{stage}_acc": self.get_accuracy(y_hat, y),
             f"{stage}_auc": self.get_auc(y_hat, y),
             f"{stage}_f1": self.get_f1(y_hat, y),
             },
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.default_step(x, y, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.default_step(x, y, stage="valid")

    def test_step(self, batch, batch_idx):
        x, y = batch
        self.default_step(x, y, stage="test")

    def set_optimizer(self, optim_name, params, lr):
        self.optim = get_optimizer(optim_name, params, lr)

    def set_loss(self, loss_name, weights=None):
        self.loss = get_loss(loss_name, weights)

    def configure_optimizers(self):
        sche = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optim, milestones=[80, 120], gamma=0.1)
        scheduler = {
            "scheduler": sche,
            "name": "lr_history",
        }

        return [self.optim], [scheduler]

    @staticmethod
    def get_accuracy(y_hat, y):
        return get_accuracy(y_hat, y)

    @staticmethod
    def get_f1(y_hat, y):
        return get_f1(y_hat, y)

    @staticmethod
    def get_auc(y_hat, y):
        return get_auc(y_hat, y)

    def state_dict(self, **kwargs):
        return self.model.state_dict()


class Vanilla(LitClassifier):
    def __init__(self, model, config):
        super().__init__(model, config)

    def configure_callbacks(self):
        return [Freeze()]


class Freeze(Callback):
    def __init__(self, last_layer_freeze=None):
        super().__init__()
        self.last_layer_freeze = last_layer_freeze

    def on_train_epoch_start(self, trainer, pl_module):
        print("Freezing conv+bn layers.")
        for n, m in pl_module.model.named_modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

            if self.last_layer_freeze is not None and self.last_layer_freeze == n:
                print(f"Stop at layer {n}")
                break
