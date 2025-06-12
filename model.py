import torch
import torch.nn as nn
import modules
import pytorch_lightning as pl

class GeneExpPredVisiumHD(pl.LightningModule):
    def __init__(self, num_genes, 
                 converter, feature_extractor,
                 num_cell_types,
                 domain_weight = 5.0, 
                 second_order_weight=0.1,
                 cell_type_weight=0.001,
                 lr=1e-3):
        super(GeneExpPredVisiumHD, self).__init__()
        # modules
        self.translator = modules.Translator().to(self.device)
        self.domain_discriminator = modules.DomainDiscriminator(alpha=domain_weight).to(self.device)
        self.predictor = modules.Predictor(input_size=1024, hidden_size=4056, output_size=num_genes).to(self.device)
        self.cell_type_classifier = modules.CellTypeClassifier(input_size=num_genes, hidden_size=512, num_classes=num_cell_types).to(self.device)
        # feature extractors
        self.feature_extractor = feature_extractor
        self.converter = converter
        # hyperparameters
        self.lr = lr
        self.domain_weight = domain_weight
        self.coral_loss_weight = second_order_weight
        self.cell_type_weight = cell_type_weight
        # loss functions
        self.criterion = nn.MSELoss().to(self.device)
        self.domain_criterion = nn.BCELoss().to(self.device)
        self.cell_type_criterion = nn.CrossEntropyLoss().to(self.device)
        
        self.save_hyperparameters(ignore=["converter", "feature_extractor"])
    
    @staticmethod
    def coral_loss(source, target):
        """
        CORAL = covariance alignment
        Compute CORAL loss between source and target feature maps.
        Use to align the second-order statistics of the source and target distributions.
        """
        d = source.size(1)  # feature dimension
        ns = source.size(0)
        nt = target.size(0)

        # Source covariance
        xm = source - source.mean(dim=0, keepdim=True)
        xc = xm.t() @ xm / (ns - 1)

        # Target covariance
        xmt = target - target.mean(dim=0, keepdim=True)
        xct = xmt.t() @ xmt / (nt - 1)

        # Frobenius norm
        loss = torch.mean((xc - xct) ** 2)
        return loss / (4 * d * d)

    def forward(self, x, if_convert=False):
        # if_convert is used to determine whether to use the converter or not
        if if_convert:
            x = self.converter(self.device, x)
        x = self.feature_extractor(self.device, x)[:, 0, :].view(x.shape[0], -1).detach()
        x = self.translator(x)
        x = self.predictor(x)
        return x
    
    def compute_feature(self, x, if_convert=False, if_translate=True):
        if if_convert:
            x = self.converter(self.device, x)
        x = self.feature_extractor(self.device, x)[:, 0, :].view(x.shape[0], -1).detach()
        if if_translate:
            x = self.translator(x)
        return x

    def training_step(self, batch, batch_idx):
        he_image, mtx, pcm_image, _, cell_type = batch
        self.translator.train()
        self.domain_discriminator.train()
        self.predictor.train()
        # obtain the features
        he_features = self.feature_extractor(self.device, he_image)[:, 0, :].view(he_image.shape[0], -1).detach()
        pcm_features = self.feature_extractor(self.device, self.converter(self.device, pcm_image))[:, 0, :].view(pcm_image.shape[0], -1).detach()
        
        # Translator part
        he_translated = self.translator(he_features)
        pcm_translated = self.translator(pcm_features)
    
        # DANN part
        # pred_discriminator_fake = self.domain_discriminator(he_translated, pcm_translated)
        # pred_discriminator_real = self.domain_discriminator(he_translated, he_translated)
        pred_discriminator_fake = self.domain_discriminator(he_translated)
        pred_discriminator_real = self.domain_discriminator(pcm_translated)
        fake_labels = torch.zeros_like(pred_discriminator_fake)
        real_labels = torch.ones_like(pred_discriminator_real)
        domain_loss = (self.domain_criterion(pred_discriminator_fake, fake_labels) + 
                       self.domain_criterion(pred_discriminator_real, real_labels)) / 2
        coral_loss = self.coral_loss(he_translated, pcm_translated)

        # Predictor part
        exp_pred = self.predictor(he_translated)
        prediction_loss = self.criterion(exp_pred, mtx.to(self.device))

        # cell type classification part
        cell_type_pred = self.cell_type_classifier(self.predictor(pcm_translated))
        cell_type_loss = self.cell_type_weight * self.cell_type_criterion(cell_type_pred, cell_type.to(self.device))

        # total loss for training
        total_loss = prediction_loss + domain_loss + self.coral_loss_weight * coral_loss + cell_type_loss
        metrics = {"train_loss": total_loss.item(), "train_discriminator_loss": domain_loss.item()+self.coral_loss_weight*coral_loss.item(), 
                   "train_prediction_loss": prediction_loss.item(), "train_cell_type_loss": cell_type_loss.item()}
        self.log_dict(metrics,prog_bar=True)
        return total_loss
 
    def validation_step(self, batch, batch_idx):
        he_image, mtx, pcm_image, _, cell_type = batch
        self.translator.eval()
        self.domain_discriminator.eval()
        self.predictor.eval()
        with torch.no_grad():
            # obtain the features
            he_features = self.feature_extractor(self.device, he_image)[:, 0, :].view(he_image.shape[0], -1).detach()
            pcm_features = self.feature_extractor(self.device, self.converter(self.device, pcm_image))[:, 0, :].view(pcm_image.shape[0], -1).detach()
            # Translator part
            he_translated = self.translator(he_features)
            pcm_translated = self.translator(pcm_features)
            # DANN part
            pred_discriminator_fake = self.domain_discriminator(he_translated)
            pred_discriminator_real = self.domain_discriminator(pcm_translated)
            fake_labels = torch.zeros_like(pred_discriminator_fake)
            real_labels = torch.ones_like(pred_discriminator_real)
            domain_loss = (self.domain_criterion(pred_discriminator_fake, fake_labels) + 
                        self.domain_criterion(pred_discriminator_real, real_labels)) / 2
            coral_loss = self.coral_loss(he_translated, pcm_translated)
            # Predictor part
            exp_pred = self.predictor(he_translated)
            prediction_loss = self.criterion(exp_pred, mtx.to(self.device))
            # cell type classification part
            cell_type_pred = self.cell_type_classifier(self.predictor(pcm_translated))
            cell_type_loss = self.cell_type_weight * self.cell_type_criterion(cell_type_pred, cell_type.to(self.device))

            # total loss for validation
            total_loss = prediction_loss + domain_loss + self.coral_loss_weight * coral_loss + cell_type_loss 
            metrics = {"val_loss": total_loss.item(), "val_discriminator_loss": domain_loss.item()+self.coral_loss_weight*coral_loss.item(), 
                       "val_prediction_loss": prediction_loss.item(),"val_cell_type_loss": cell_type_loss.item()}
            self.log_dict(metrics,prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]