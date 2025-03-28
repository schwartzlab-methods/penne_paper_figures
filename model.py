import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import modules
import os
import pytorch_lightning as pl

class LitModel(pl.LightningModule):
    def __init__(self, batch_size, num_genes, 
                 converter, feature_extractor,
                 device, domain_weight = 5.0):
        super(LitModel, self).__init__()
        # modules
        self.translator = modules.Translator().to(device)
        self.domain_discriminator = modules.DomainDiscriminator().to(device)
        self.predictor = modules.Predictor(input_size=1024, hidden_size=4056, output_size=num_genes).to(device)
        # feature extractors
        self.feature_extractor = feature_extractor
        self.converter = converter
        # hyperparameters
        self.batch_size = batch_size
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
        # translate the features
        he_translated = self.translator(he_features)
        pcm_translated = self.translator(pcm_features)
        # DANN part
        pred_discriminator_fake = self.domain_discriminator(he_translated, pcm_translated)
        pred_discriminator_real = self.domain_discriminator(he_translated, he_translated)
        fake_labels = torch.zeros_like(pred_discriminator_fake)
        real_labels = torch.ones_like(pred_discriminator_real)
        discriminator_loss = (self.domain_criterion(pred_discriminator_fake, fake_labels) + 
                              self.domain_criterion(pred_discriminator_real, real_labels)) / 2
        # generator part
        exp_pred = self.predictor(he_translated)
        prediction_loss = self.criterion(exp_pred, mtx.to(self.device))
        # total loss
        #todo: figure out how to do DANN
        




    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass