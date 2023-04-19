import torch
from model_training.correction_methods.base_correction_method import LitClassifier, Freeze
from utils import cd

class CDEP(LitClassifier):
    def __init__(self, model, config, **kwargs):
        super().__init__(model, config, **kwargs)
        self.lamb = self.config["lamb"]  # 100
        self.avg_layer = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.model_name = config['model_name']

    def training_step(self, batch, batch_idx):
        x, y, z = batch
        loss = self.default_step(x, y, z, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, z = batch
        self.default_step(x, y, z, stage="valid")

    def test_step(self, batch, batch_idx):
        x, y, z = batch
        self.default_step(x, y, z, stage="test")

    def get_features(self, x):
        if self.model_name == "vgg16":
            return self.avg_layer(self.model.features(x)).view(-1).detach().cpu().numpy()
        elif self.model_name == "resnet18":
            """     x = self.conv1(x)
                    x = self.bn1(x)
                    x = self.relu(x)
                    x = self.maxpool(x)
                
                    x = self.layer1(x)
                    x = self.layer2(x)
                    x = self.layer3(x)
                    x = self.layer4(x)
                
                    x = self.last_relu(self.last_conv(x))  # added identity
                
                    x = self.avgpool(x)
                    x = torch.flatten(x, 1)
                    x = self.fc(x)
                    """

            return self.model.avgpool(
                self.model.layer4(
                    self.model.layer3(
                        self.model.layer2(
                            self.model.layer1(
                                self.model.maxpool(
                                    self.model.relu(
                                        self.model.bn1(
                                            self.model.conv1(x))))))))).view(-1).detach().cpu().numpy()
        else:
            raise ValueError(f"CDEP not implemented for model type {self.model_name}")

    def compute_cd(self, cd_features, inputs):
        if self.model_name == "vgg16":
            rel, irrel = cd.cd_vgg_classifier(cd_features[:, 0], cd_features[:, 1], inputs, self.model.classifier)
        elif self.model_name == "resnet18":
            rel, irrel = cd.cd_resnet_classifier(cd_features[:, 0], cd_features[:, 1], inputs, self.model)
        else:
            raise ValueError(f"CDEP not implemented for model type {self.model_name}")
        return rel, irrel

    def default_step(self, x, y, z, stage):
        cd_features = z
        mask = (cd_features[:, 0, 0] != -1).bool()

        with torch.enable_grad():
            x.requires_grad = True
            y_hat = self(x)

            if mask.any():
                if self.model.training:
                    self.model.eval()
                    inputs = self.get_features(x)
                    self.model.train()
                else:
                    inputs = self.get_features(x)

                rel, irrel = self.compute_cd(cd_features, inputs)
                cur_cd_loss = 0
                for class_ in range(rel.shape[1]):
                    cur_cd_loss += torch.nn.functional.softmax(
                        torch.stack((rel[:, class_].masked_select(mask),
                                     irrel[:, class_].masked_select(mask)), dim=1), dim=1)[:, 0].mean()
                aux_loss = cur_cd_loss / rel.shape[1]
            else:
                aux_loss = 0.0

        loss = self.loss(y_hat, y) + self.lamb * aux_loss
        self.log_dict(
            {f"{stage}_loss": loss,
             f"{stage}_acc": self.get_accuracy(y_hat, y),
             f"{stage}_auc": self.get_auc(y_hat, y),
             f"{stage}_f1": self.get_f1(y_hat, y),
             f"{stage}_auxloss": aux_loss},
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def configure_callbacks(self):
        return [Freeze()]