import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import modules
import pytorch_lightning as pl

class GeneExpPredVisiumHD(pl.LightningModule):
    def __init__(self, num_genes, 
                 converter, feature_extractor,
                 device, domain_weight = 5.0, lr=1e-3):
        super(GeneExpPredVisiumHD, self).__init__()
        # modules
        self.translator = modules.Translator().to(device)
        self.domain_discriminator = modules.DomainDiscriminator(alpha=domain_weight).to(device)
        self.predictor = modules.Predictor(input_size=1024, hidden_size=4056, output_size=num_genes).to(device)
        # feature extractors
        self.feature_extractor = feature_extractor
        self.converter = converter
        # hyperparameters
        self.lr = lr
        self.domain_weight = domain_weight
        self.device = device
        # loss functions
        self.criterion = nn.MSELoss().to(device)
        self.domain_criterion = nn.BCELoss().to(device)

    def training_step(self, batch, batch_idx):
        he_image, mtx, pcm_image = batch
        # obtain the features
        he_features = self.feature_extractor(he_image.to(self.device))
        pcm_features = self.feature_extractor(self.converter(pcm_image.to(self.device)))
        # Translator part
        he_translated = self.translator(he_features)
        pcm_translated = self.translator(pcm_features)
        # DANN part
        pred_discriminator_fake = self.domain_discriminator(he_translated, pcm_translated)
        pred_discriminator_real = self.domain_discriminator(he_translated, he_translated)
        fake_labels = torch.zeros_like(pred_discriminator_fake)
        real_labels = torch.ones_like(pred_discriminator_real)
        discriminator_loss = (self.domain_criterion(pred_discriminator_fake, fake_labels) + 
                              self.domain_criterion(pred_discriminator_real, real_labels)) / 2
        # Predictor part
        exp_pred = self.predictor(he_translated)
        prediction_loss = self.criterion(exp_pred, mtx.to(self.device))
        # DANN part
        domain_loss = self.domain_weight * (self.domain_criterion(pred_discriminator_fake, fake_labels) +
                                            self.domain_criterion(pred_discriminator_real, real_labels)) / 2
        # total loss for training
        total_loss = prediction_loss + domain_loss
        self.log('train_loss', total_loss, prog_bar=False)
        self.log('tran_discriminator_loss', discriminator_loss, prog_bar=False)
        self.log('tran_prediction_loss', prediction_loss, prog_bar=True)
        return total_loss
 
    def validation_step(self, batch, batch_idx):
        he_image, mtx, pcm_image = batch
        # obtain the features
        he_features = self.feature_extractor(he_image.to(self.device))[:, 0, :]
        pcm_features = self.feature_extractor(self.converter(pcm_image.to(self.device)))[:, 0, :]
        # Translator part
        he_translated = self.translator(he_features)
        pcm_translated = self.translator(pcm_features)
        # DANN part
        pred_discriminator_fake = self.domain_discriminator(he_translated, pcm_translated)
        pred_discriminator_real = self.domain_discriminator(he_translated, he_translated)
        fake_labels = torch.zeros_like(pred_discriminator_fake)
        real_labels = torch.ones_like(pred_discriminator_real)
        discriminator_loss = (self.domain_criterion(pred_discriminator_fake, fake_labels) + 
                              self.domain_criterion(pred_discriminator_real, real_labels)) / 2
        # Predictor part
        exp_pred = self.predictor(he_translated)
        prediction_loss = self.criterion(exp_pred, mtx.to(self.device))
        # DANN part
        domain_loss = self.domain_weight * (self.domain_criterion(pred_discriminator_fake, fake_labels) +
                                            self.domain_criterion(pred_discriminator_real, real_labels)) / 2
        # total loss for validation
        total_loss = prediction_loss + domain_loss
        self.log('val_loss', total_loss, prog_bar=False)
        self.log('val_discriminator_loss', discriminator_loss, prog_bar=False)
        self.log('val_prediction_loss', prediction_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]