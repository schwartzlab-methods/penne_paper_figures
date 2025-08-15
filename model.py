import torch
import torch.nn as nn
import torch.nn.functional as F
import modules
import pytorch_lightning as pl
import random
from modules import SpaghettiGenerator

class GeneExpPredVisiumHD(pl.LightningModule):
    def __init__(self, num_genes, 
                 converter, feature_extractor,
                 end_to_end=False,
                 num_cell_types=0,
                 up_marker_genes=None,
                 domain_weight = 5.0, 
                 second_order_weight=0.1,
                 cell_type_weight=0.0,
                 marker_gene_weight=0.0,
                 orthogonal_loss_weight=0.0,
                 lr=1e-3,
                 do_gmlp=True, across_cell=False):
        '''Gene expression prediction model for Visium HD data.

        Args:
            num_genes (int): 
                Number of genes to predict.
            converter (nn.Module): 
                Module to convert image features to gene expression.
            feature_extractor (list(Callable, Callable)): 
                Preprocessor and Model to extract features from images.
            end_to_end (bool, optional): 
                Whether to train the model end-to-end. Defaults to False.
            num_cell_types (int, optional): 
                Number of cell types in the dataset. Defaults to 0.
            up_marker_genes (dict[int, list[str]], optional): 
                Mapping from cell type indices to lists of upregulated marker genes. Defaults to None.
            domain_weight (float, optional): 
                Weight for domain adaptation loss. Defaults to 5.0.
            second_order_weight (float, optional): 
                Weight for second-order loss. Defaults to 0.1.
            cell_type_weight (float, optional): 
                Weight for cell type classification loss. Defaults to 0.0.
            marker_gene_weight (float, optional): 
                Weight for marker gene prediction loss. Defaults to 0.0.
            lr (float, optional): 
                Learning rate for the optimizer. Defaults to 1e-3.
            do_gmlp (bool, optional): 
                Whether to use Gated MLP for prediction. Defaults to True.
            across_cell (bool, optional): 
                Whether to use across-cell information. Defaults to False.
            orthogonal_loss_weight (float, optional): 
                Weight for orthogonal loss. Defaults to 0.0.
            orthogonal_features (bool, optional): 
                Whether to use orthogonal features. Defaults to False. 
        '''
        super(GeneExpPredVisiumHD, self).__init__()
        # modules
        self.translator = modules.Translator()
        if orthogonal_loss_weight > 0:
            self.feature_biology_translator = modules.OrthogonalTranslator(feature_in=1024, feature_out=756)
            self.feature_domain_translator = modules.OrthogonalTranslator(feature_in=1024, feature_out=268)
            self.orthogonal_loss_weight = orthogonal_loss_weight
        self.domain_discriminator = modules.DomainDiscriminator(feature_in=268 if orthogonal_loss_weight > 0 else 1024, 
                                                                alpha=domain_weight)
        predictor_input_size = 756 if orthogonal_loss_weight > 0 else 1024
        if do_gmlp: # Use Gated MLP for prediction
            self.predictor = modules.PredictorGMLP(input_size=predictor_input_size, hidden_size=4056, output_size=num_genes)
        else:
            self.predictor = modules.Predictor(input_size=predictor_input_size, hidden_size=4056, output_size=num_genes)
        if up_marker_genes:
            self.cell_type_classifier = modules.CellTypeClassifier(input_size=num_genes, hidden_size=512, num_classes=num_cell_types)
            self.up_marker_genes_dict = up_marker_genes
        # feature extractors
        self.image_processor, self.feature_extractor = feature_extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        if converter:
            self.converter = converter
        else:
            self.converter = SpaghettiGenerator(3, 9)
        if not end_to_end:
            for param in self.converter.parameters():
                param.requires_grad = False
            converter.eval()
        # hyperparameters
        self.end_to_end = end_to_end
        self.lr = lr
        self.domain_weight = domain_weight
        self.coral_loss_weight = second_order_weight
        self.cell_type_weight = cell_type_weight
        self.marker_gene_weight = marker_gene_weight
        self.second_stage_training = True if up_marker_genes else False
        self.make_ortho = orthogonal_loss_weight > 0
        self.across_cell = across_cell
        # loss functions
        self.criterion = nn.MSELoss().to(self.device)
        self.domain_criterion = nn.BCELoss().to(self.device)
        if up_marker_genes:
            self.cell_type_criterion = nn.CrossEntropyLoss().to(self.device)

        self.save_hyperparameters(ignore=["converter", "feature_extractor"])
    
    @staticmethod
    def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        CORAL = covariance alignment
        Compute CORAL loss between source and target feature maps.
        Use to align the second-order statistics of the source and target distributions.
        Args:
            source (torch.Tensor): Source feature map.
            target (torch.Tensor): Target feature map.
        Returns:
            torch.Tensor: CORAL loss.
        """
        d = source.size(1)  # feature dimension
        ns = source.size(0)  # batch size
        nt = target.size(0)

        # Source covariance
        xm = source - source.mean(dim=0, keepdim=True)
        xc = xm.t() @ xm / (ns - 1)

        # Target covariance
        xmt = target - target.mean(dim=0, keepdim=True)
        xct = xmt.t() @ xmt / (nt - 1)

        # Frobenius norm
        loss = torch.mean((xc - xct) ** 2)
        return loss / (4 * d * d)  # normalize by feature dimension
    
    @staticmethod
    def orthogonal_loss(biology: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        '''Compute the orthogonal loss between biology and target feature representations.

        Args:
            biology (torch.Tensor): 
                Biology feature representation of shape (batch_size, num_features).
            domain (torch.Tensor): 
                Domain feature representation of shape (batch_size, num_features).

        Returns:
            torch.Tensor: 
                Orthogonal loss. 
                Computed using the cosine similarity between the biology and domain feature representations.
        '''
        biology = biology - biology.mean(dim=0, keepdim=True)
        domain = domain - domain.mean(dim=0, keepdim=True)

        orthogonality_matrix = torch.matmul(biology.T, domain) / biology.size(0)
        loss = torch.sum(orthogonality_matrix ** 2)
        return loss

    def forward(self, x: torch.Tensor, if_convert: bool=False) -> torch.Tensor:
        '''Forward pass for the model.

        Args:
            x (torch.Tensor): 
                Input tensor of the image in the shape of (batch_size, num_channels, height, width).
            if_convert (bool, optional): 
                Whether to use the converter to convert into H&E like images. Defaults to False.

        Returns:
            torch.Tensor: 
                Output tensor of the gene expression inference in the shape of (batch_size, num_genes).
        '''
        # if_convert is used to determine whether to use the converter or not
        if if_convert:
            x = self.converter(x)
        x_converted = self.image_processor(x)      
        x = self.feature_extractor(x_converted).last_hidden_state[:, 0, :].view(x.shape[0], -1)
        x = self.translator(x)
        x = self.predictor(x)
        x = torch.clamp(x, min=0)
        x = x / (torch.sum(x, dim=-1, keepdim=True)+1e-10) * 1e6
        x = torch.log2(x + 1)
        return x
    
    def compute_feature(self, x: torch.Tensor, if_convert: bool=False, 
                        if_translate: bool=True, if_ortho: bool=True) -> torch.Tensor:
        '''Compute feature representation for the input tensor.

        Args:
            x (torch.Tensor): 
                Input tensor of the image in the shape of (batch_size, num_channels, height, width).
            if_convert (bool, optional): 
                Whether to use the converter to convert into H&E like images. Defaults to False.
            if_translate (bool, optional): 
                Whether to use the translator for better domain alignment. Defaults to True.
            if_ortho (bool, optional): 
                Whether to use the orthogonal translator. Defaults to True.

        Returns:
            torch.Tensor: 
                Feature representation of the input tensor in the shape of (batch_size, num_features).
        '''

        with torch.no_grad():
            if if_convert:
                x = self.converter(x)
            x = self.image_processor(x)
            x = self.feature_extractor(x).last_hidden_state[:, 0, :].view(x.shape[0], -1)
            if if_translate:
                x = self.translator(x)
            if if_ortho:
                x = self.feature_biology_translator(x)
            return x

    def compute_gate(self, x: torch.Tensor, if_convert: bool=False, 
                     if_translate: bool=True, if_ortho: bool=True) -> torch.Tensor:
        '''Compute gating vector for the input tensor.

        Args:
            x (torch.Tensor): 
                Input tensor of the image in the shape of (batch_size, num_channels, height, width).
            if_convert (bool, optional): 
                Whether to use the converter to convert into H&E like images. Defaults to False.
            if_translate (bool, optional): 
                Whether to use the translator for better domain alignment. Defaults to True.
            if_ortho (bool, optional): 
                Whether to use the orthogonal translator. Defaults to True.

        Returns:
            torch.Tensor: 
                Gating vector of the input tensor in the shape of (batch_size, num_features).
        '''
        with torch.no_grad():
            if if_convert:
                x = self.converter(x)
            x = self.image_processor(x)
            x = self.feature_extractor(x).last_hidden_state[:, 0, :].view(x.shape[0], -1)
            if if_translate:
                x = self.translator(x)
            if if_ortho:
                x = self.feature_biology_translator(x)
            x = self.predictor(x, return_gate=True)
            return x

    def _marker_margin_loss(self, pred_expr: torch.Tensor, cell_types: torch.Tensor,
                            marker_dict: dict, margin: float=1.0,
                            across_cell: bool=False) -> torch.Tensor:
        '''Compute the marker margin loss.

        Args:
            pred_expr (torch.Tensor): 
                Predicted gene expression values in the shape of (batch_size, num_genes).
            cell_types (torch.Tensor): 
                Cell type labels in the shape of (batch_size,).
            marker_dict (dict[Int, List[Int]]): 
                Dictionary mapping cell type indices to marker gene indices in exp matrix.
            margin (float, optional): 
                Margin for the loss. Defaults to 1.0.
            across_cell (bool, optional): 
                Whether to compute the loss across cell types instead of within cell types. Defaults to False.

        Returns:
            torch.Tensor: Computed loss value of the marker margin loss.
        '''
        loss = 0.0
        if across_cell:
            # Compare the marker gene exp across all cells. 
            # The marker genes should be more expressed in the target cell type than in any other cell type
            unique_cell_types = torch.unique(cell_types)
            for cell_type in unique_cell_types:
                marker_genes = torch.tensor(marker_dict[cell_type.item()])
                num_marker_genes = marker_genes.sum().item()
                if num_marker_genes == 0:
                    continue
                # get the marker gene exp for this cell type and other cell types
                marker_genes = marker_genes.bool().to(self.device)
                current_cell_mask = (cell_types == cell_type).to(self.device)
                pred_expr_current_cell = pred_expr[current_cell_mask, :]
                pred_expr_other_cells = pred_expr[~current_cell_mask, :]
                if (pred_expr_current_cell.size(0) == 0) or (pred_expr_other_cells.size(0) == 0):
                    continue
                # get the mean marker gene exp for each gene
                marker_genes = marker_genes.to(self.device)
                marker_expr_current_cell = pred_expr_current_cell[:, marker_genes].mean(dim=0, keepdim=True).view(1, -1)
                marker_expr_other_cells = pred_expr_other_cells[:, marker_genes].mean(dim=0, keepdim=True).view(1, -1)
                # calculate loss
                loss += F.margin_ranking_loss(
                    marker_expr_current_cell, marker_expr_other_cells,
                    target=torch.ones_like(marker_expr_current_cell), margin=margin
                )
            return loss / unique_cell_types.size(0) 
        else: # Compare the marker gene exp within each cell type
            for i in range(pred_expr.size(0)):
                cell_type = cell_types[i].item()
                marker_genes = torch.tensor(marker_dict[cell_type])
                num_marker_genes = marker_genes.sum().item()
                if num_marker_genes == 0:
                    continue
                non_marker_genes = 1 - marker_genes
                # Sample a few non-marker genes to compare against
                idx_non_marker = non_marker_genes.nonzero(as_tuple=False).flatten()
                sampled_non = random.sample(range(idx_non_marker.shape[0]), num_marker_genes)
                sampled_non_marker_genes_idx = idx_non_marker[sampled_non]
                marker_vals = pred_expr[i, marker_genes.bool().to(self.device)].view(1,-1)
                non_marker_vals = pred_expr[i, sampled_non_marker_genes_idx].view(1,-1)
                # calculate loss
                loss += F.margin_ranking_loss(
                    marker_vals, non_marker_vals, target=torch.ones_like(marker_vals), margin=margin
                )
            return loss / pred_expr.size(0)

    def training_step(self, batch: tuple, batch_idx: int):
        '''Perform a single training step.

        Args:
            batch (tuple): A tuple containing the input data for the batch. This contains the following elements:
                - he_image (torch.Tensor): The HE image tensor in the shape of (batch_size, channels, height, width).
                - mtx (torch.Tensor): The gene expression matrix.
                - pcm_image (torch.Tensor): The PCM image tensor.
                - _ (str): HE image path, not used during training
                - cell_type (torch.Tensor): The cell type labels in the shape of (batch_size,).
            batch_idx (int): The index of the batch.

        Returns:
            loss (torch.Tensor): The computed loss for the batch.
        '''
        he_image, mtx, pcm_image, _, cell_type = batch
        # obtain the features
        pcm_converted = self.converter(pcm_image)
        pcm_converted = self.image_processor(pcm_converted)
        he_converted = self.image_processor(he_image)
        pcm_features = self.feature_extractor(pcm_converted).last_hidden_state[:, 0, :].view(pcm_image.shape[0], -1).requires_grad_()#.detach()
        he_features = self.feature_extractor(he_converted).last_hidden_state[:, 0, :].view(he_image.shape[0], -1).requires_grad_()#.detach()
        # Translator part
        # this part translates the features into a common space
        he_translated = self.translator(he_features)
        pcm_translated = self.translator(pcm_features)
        if hasattr(self, "feature_biology_translator"):
            he_translated_biology = self.feature_biology_translator(he_translated)
            pcm_translated_biology = self.feature_biology_translator(pcm_translated)
            he_translated_domain = self.feature_domain_translator(he_translated)
            pcm_translated_domain = self.feature_domain_translator(pcm_translated)
            # loss for this part
            ortho_loss = (self.orthogonal_loss_weight
                          * (self.orthogonal_loss(he_translated_biology, he_translated_domain) 
                          + self.orthogonal_loss(pcm_translated_biology, pcm_translated_domain)) / 2)
        else:
            he_translated_biology = he_translated
            pcm_translated_biology = pcm_translated
            he_translated_domain = he_translated
            pcm_translated_domain = pcm_translated

        # DANN part
        # this part is for domain adaptation, uses domain features only
        pred_discriminator_fake = self.domain_discriminator(he_translated_domain)
        pred_discriminator_real = self.domain_discriminator(pcm_translated_domain)
        fake_labels = torch.zeros_like(pred_discriminator_fake)
        real_labels = torch.ones_like(pred_discriminator_real)
        domain_loss = (self.domain_criterion(pred_discriminator_fake, fake_labels) + 
                       self.domain_criterion(pred_discriminator_real, real_labels)) / 2
        coral_loss = self.coral_loss(he_translated_domain, pcm_translated_domain)

        # Predictor part
        # This part leverages the biological translated features
        exp_pred = self.predictor(he_translated_biology)
        # normalize to counts per million and log2 transform
        exp_pred = torch.clamp(exp_pred, min=0)
        exp_pred = exp_pred / (torch.sum(exp_pred, dim=-1, keepdim=True)+1e-10) * 1e6
        exp_pred = torch.log2(exp_pred + 1)
        prediction_loss = self.criterion(exp_pred, mtx.to(self.device))

        if self.second_stage_training:
            pred_exp_pcm = self.predictor(pcm_translated_biology)
            # normalize to counts per million and log2 transform
            pred_exp_pcm = torch.clamp(pred_exp_pcm, min=0)
            pred_exp_pcm = pred_exp_pcm / (torch.sum(pred_exp_pcm, dim=-1, keepdim=True)+1e-10) * 1e6
            pred_exp_pcm = torch.log2(pred_exp_pcm + 1)
            # cell type classification part
            cell_type_pred = self.cell_type_classifier(pred_exp_pcm)
            cell_type_loss = self.cell_type_weight * self.cell_type_criterion(cell_type_pred, cell_type.to(self.device))

            # marker gene loss
            marker_gene_loss = self.marker_gene_weight * self._marker_margin_loss(pred_exp_pcm, cell_type, 
                                                                                  self.up_marker_genes_dict, across_cell=self.across_cell)

            # total loss for training
            total_loss = prediction_loss + domain_loss + self.coral_loss_weight * coral_loss + cell_type_loss + marker_gene_loss
            metrics = {"train_loss": total_loss.item(), "train_discriminator_loss": domain_loss.item()+self.coral_loss_weight*coral_loss.item(), 
                        "train_prediction_loss": prediction_loss.item(), "train_cell_type_loss": cell_type_loss.item(), 
                        "train_marker_gene_loss": marker_gene_loss}
            if self.make_ortho:
                total_loss += ortho_loss
                metrics["train_ortho_loss"] = ortho_loss.item()
        else:
            # total loss for training
            total_loss = prediction_loss + domain_loss + self.coral_loss_weight * coral_loss
            metrics = {"train_loss": total_loss.item(), "train_discriminator_loss": domain_loss.item()+self.coral_loss_weight*coral_loss.item(), 
                        "train_prediction_loss": prediction_loss.item()}
            if self.make_ortho:
                total_loss += ortho_loss
                metrics["train_ortho_loss"] = ortho_loss.item()
                
        self.log_dict(metrics,prog_bar=True)
        return total_loss
 
    def validation_step(self, batch, batch_idx):
        '''Perform a single validation step.

        Args:
            batch (tuple): A tuple containing the input data for the batch. This contains the following elements:
                - he_image (torch.Tensor): The HE image tensor in the shape of (batch_size, channels, height, width).
                - mtx (torch.Tensor): The gene expression matrix.
                - pcm_image (torch.Tensor): The PCM image tensor.
                - _ (str): HE image path, not used during validation
                - cell_type (torch.Tensor): The cell type labels in the shape of (batch_size,).
            batch_idx (int): The index of the batch.

        Returns:
            loss (torch.Tensor): The computed loss for the batch.
        '''
        he_image, mtx, pcm_image, _, cell_type = batch
        with torch.no_grad():
            # obtain the features
            pcm_converted = self.converter(pcm_image)
            pcm_converted = self.image_processor(pcm_converted)
            he_converted = self.image_processor(he_image)
            pcm_features = self.feature_extractor(pcm_converted).last_hidden_state[:, 0, :].view(pcm_image.shape[0], -1).detach()
            he_features = self.feature_extractor(he_converted).last_hidden_state[:, 0, :].view(he_image.shape[0], -1).detach()
            # Translator part
            he_translated = self.translator(he_features)
            pcm_translated = self.translator(pcm_features)
            if hasattr(self, "feature_biology_translator"):
                he_translated_biology = self.feature_biology_translator(he_translated)
                pcm_translated_biology = self.feature_biology_translator(pcm_translated)
                he_translated_domain = self.feature_domain_translator(he_translated)
                pcm_translated_domain = self.feature_domain_translator(pcm_translated)
                # loss for this part
                ortho_loss = (self.orthogonal_loss_weight
                            * (self.orthogonal_loss(he_translated_biology, he_translated_domain) 
                            + self.orthogonal_loss(pcm_translated_biology, pcm_translated_domain)) / 2)
            else:
                he_translated_biology = he_translated
                pcm_translated_biology = pcm_translated
                he_translated_domain = he_translated
                pcm_translated_domain = pcm_translated
            # DANN part
            pred_discriminator_fake = self.domain_discriminator(he_translated_domain)
            pred_discriminator_real = self.domain_discriminator(pcm_translated_domain)
            fake_labels = torch.zeros_like(pred_discriminator_fake)
            real_labels = torch.ones_like(pred_discriminator_real)
            domain_loss = (self.domain_criterion(pred_discriminator_fake, fake_labels) + 
                        self.domain_criterion(pred_discriminator_real, real_labels)) / 2
            coral_loss = self.coral_loss(he_translated_domain, pcm_translated_domain)
            # Predictor part
            exp_pred = self.predictor(he_translated_biology)
            exp_pred = torch.clamp(exp_pred, min=0)
            exp_pred = exp_pred / (torch.sum(exp_pred, dim=-1, keepdim=True)+1e-10) * 1e6
            exp_pred = torch.log2(exp_pred + 1)
            prediction_loss = self.criterion(exp_pred, mtx.to(self.device))
            if self.second_stage_training:
                pred_exp_pcm = self.predictor(pcm_translated_biology)
                pred_exp_pcm = torch.clamp(pred_exp_pcm, min=0)
                pred_exp_pcm = pred_exp_pcm / (torch.sum(pred_exp_pcm, dim=-1, keepdim=True)+1e-10) * 1e6
                pred_exp_pcm = torch.log2(pred_exp_pcm + 1)
                # cell type classification part
                cell_type_pred = self.cell_type_classifier(pred_exp_pcm)
                cell_type_loss = self.cell_type_weight * self.cell_type_criterion(cell_type_pred, cell_type.to(self.device))

                # marker gene loss
                marker_gene_loss = self.marker_gene_weight * self._marker_margin_loss(pred_exp_pcm, cell_type, 
                                                                                      self.up_marker_genes_dict, across_cell=self.across_cell)

                # total loss for validation
                total_loss = prediction_loss + domain_loss + self.coral_loss_weight * coral_loss + cell_type_loss + marker_gene_loss
                metrics = {"val_loss": total_loss.item(), "val_discriminator_loss": domain_loss.item()+self.coral_loss_weight*coral_loss.item(), 
                            "val_prediction_loss": prediction_loss.item(), "val_cell_type_loss": cell_type_loss.item(), 
                            "val_marker_gene_loss": marker_gene_loss}
                if self.make_ortho:
                    total_loss += ortho_loss
                    metrics["val_ortho_loss"] = ortho_loss.item()
            else:
                # total loss for validation
                total_loss = prediction_loss + domain_loss + self.coral_loss_weight * coral_loss
                metrics = {"val_loss": total_loss.item(), "val_discriminator_loss": domain_loss.item()+self.coral_loss_weight*coral_loss.item(), 
                            "val_prediction_loss": prediction_loss.item()}
                if self.make_ortho:
                    total_loss += ortho_loss
                    metrics["val_ortho_loss"] = ortho_loss.item()
            self.log_dict(metrics,prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]