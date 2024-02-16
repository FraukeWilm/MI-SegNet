import torchmetrics
import pytorch_lightning as pl
from MI_SegNet import Mine_Conv, Seg_encoder_LM,Seg_decoder_LM,Recon_encoder_LM, Recon_decoder_LM
from torchvision import transforms
from wandb.sdk.data_types.image import Image
from torchmetrics.classification.jaccard import JaccardIndex
from loss import *
from segmentation_models_pytorch import Unet
import torch.nn as nn


class UnetModule(pl.LightningModule):
    def __init__(self, cfg, device, **kwargs):
        super().__init__()
        self.automatic_optimization = False
        self._lr = cfg.training.lr
        self._log_imgs = False
        #num_classes = kwargs['num_classes']

        # save all named parameters
        self.save_hyperparameters()
        unet = Unet(encoder_name='resnet34', classes=3)
        self.seg_encoder = unet.encoder.to(device)
        self.seg_decoder = unet.decoder.to(device)
        self.segmentation_head = unet.segmentation_head.to(device)
        self.transform_image = transforms.Normalize(0.5, 0.5)

        # create loss and metric functions
        self._scheduler = 'none' #step

        self._train_loss_agg = torchmetrics.MeanMetric()
        self._val_loss_agg = torchmetrics.MeanMetric()
        self._seg_loss_agg = torchmetrics.MeanMetric()
        self.jaccard_source = JaccardIndex(task="multiclass", num_classes=cfg.model.output_channel, average='none', ignore_index=-1)
        self.jaccard_target = JaccardIndex(task="multiclass", num_classes=cfg.model.output_channel, average='none', ignore_index=-1)

        # intialize loss functions and weights
        self.ce = CELoss()
        self.dice = DiceLoss()

        # optimizers
        self.optimizer_idxs = ['optimizer_seg_en', 'optimizer_seg_de']


    def forward(self, x):
        # the method used for inference
        output = self._model(x)
        return output

    def training_step(self, batch, batch_idx):
        # get training batch
        inputs_1 = batch[0].float().to(self.device) # domain 1 anatomy 1
        inputs_2 = batch[1].float().to(self.device) # domain 2 anatomy 2
        inputs_12 = batch[2].float().to(self.device) # domain 1 anatomy 2
        inputs_21 = batch[3].float().to(self.device) # domain 2 anatomy 1
        labels_1 = batch[4].to(self.device)
        labels_2 = batch[5].to(self.device)

        # normalize inputs to 0.5 mu and sigma
        inputs_1_trans = inputs_1 #self.transform_image(inputs_1)
        inputs_2_trans = inputs_2 #self.transform_image(inputs_2)

        # segmentation forward pass
        seg_loss_1, seg_results_1 = self.update_Seg(inputs_1_trans, labels_1, True)
        seg_loss_2, seg_results_2 = self.update_Seg(inputs_2_trans, labels_2, True)
        seg_loss = seg_loss_1 + seg_loss_2

        loss = seg_loss
        self._train_loss_agg.update(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # get validation batch
        inputs_1 = batch[0].float().to(self.device) # domain 1 anatomy 1
        inputs_2 = batch[1].float().to(self.device)  # domain 2 anatomy 2
        inputs_12 = batch[2].float().to(self.device) # domain 1 anatomy 2
        inputs_21 = batch[3].float().to(self.device) # domain 2 anatomy 1
        labels_1 = batch[4].to(self.device)
        labels_2 = batch[5].to(self.device)

        # normalize inputs to 0.5 mu and sigma
        inputs_1_trans = self.transform_image(inputs_1)
        inputs_2_trans = self.transform_image(inputs_2)

        # segmentation forward pass
        seg_loss_1, seg_results_1 = self.update_Seg(inputs_1_trans, labels_1, False)
        seg_loss_2, seg_results_2 = self.update_Seg(inputs_2_trans, labels_2, False)
        seg_loss = seg_loss_1 + seg_loss_2

        # compute validation loss 
        val_loss = seg_loss
        self._val_loss_agg.update(val_loss)
        self._seg_loss_agg.update(seg_loss)
        # compute source data validation metrics
        self.jaccard_source.update(seg_results_1, labels_1)
        # compute target data validation metrics
        self.jaccard_target.update(seg_results_2, labels_2)

        # we could also log example images to wandb here
        if batch_idx == 0 and self.current_epoch % 10 ==0:
            self.log_images(inputs_1, inputs_2, inputs_12, inputs_21, labels_1, labels_2, seg_results_1, seg_results_2)
        return val_loss

    def log_images(self, inputs_1, inputs_2, inputs_12, inputs_21, labels_1, labels_2, seg_results_1, seg_results_2):
        with torch.no_grad():
            class_labels = {0: "excluded", 1: "background", 2: "non-tumor", 3: "tumor"}
            for i in range(inputs_1.shape[0]):
                input = Image(255 * torch.cat((inputs_1[i], inputs_21[i], inputs_2[i], inputs_12[i]),dim=-1).permute(1, 2, 0).cpu().numpy())
                mask_img = Image(255 * torch.cat((inputs_1[i], inputs_2[i]), dim=-1).permute(1, 2, 0).cpu().numpy(), masks={
                    "ground_truth": {
                        "mask_data": torch.cat((labels_1[i], labels_2[i]), dim=-1).cpu().numpy() + 1,
                        "class_labels": class_labels
                    },
                    "prediction": {
                        "mask_data": torch.cat((torch.argmax(seg_results_1[i], dim=0),torch.argmax(seg_results_2[i], dim=0)), dim=-1).cpu().numpy() + 1,
                        "class_labels": class_labels
                    },
                })
                self.logger.log_metrics({"Input": input})
                self.logger.log_metrics({"Segmentation Output": mask_img})


    def on_train_epoch_end(self):
        # required if values returned in the training_steps have to be processed in a specific way
        self.log("Train Loss", self._train_loss_agg.compute(), sync_dist=True)
        self._train_loss_agg.reset()

    def on_validation_epoch_end(self):
        # required if values returned in the validation_step have to be processed in a specific way
        self.log("Val Loss", self._val_loss_agg.compute(), sync_dist=True)
        self._val_loss_agg.reset()

        # log partial losses
        self.log("Seg Loss", self._seg_loss_agg.compute(), sync_dist=True)
        self._seg_loss_agg.reset()

        iou_source = self.jaccard_source.compute()
        self.log("Source mIoU", iou_source.mean(), sync_dist=True)
        self.log_dict({"Source IoU Background": iou_source[0], "Source IoU Normal": iou_source[1], "Source IoU Tumor": iou_source[2]}, sync_dist=True)
        self.jaccard_source.reset()

        iou_target = self.jaccard_target.compute()
        self.log("Target mIoU", iou_target.mean(), sync_dist=True)
        self.log_dict({"Target IoU Background": iou_target[0], "Target IoU Normal": iou_target[1], "Target IoU Tumor": iou_target[2]}, sync_dist=True)
        self.jaccard_target.reset()

    def configure_optimizers(self):
        schedulers = []

        # configure optimizers
        optimizer_seg_en = torch.optim.Adam(self.seg_encoder.parameters(), lr=self._lr)
        optimizer_seg_de = torch.optim.Adam(list(self.seg_decoder.parameters()) + list(self.segmentation_head.parameters()), lr=self._lr)
        optimizers = [optimizer_seg_en, optimizer_seg_de]

        # configure shedulers based on command line parameters
        if self._scheduler == "none":
            return optimizers
        elif self._scheduler == "step":
            schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1) for optimizer in optimizers]
        # add more options if required
        return optimizers, schedulers

    def reset_grad(self):
        for optimizer in self.optimizers():
            optimizer.zero_grad()

    def update_Seg(self, inputs, labels, train, ignore_index=-1):
        z = self.seg_encoder(inputs)
        seg_results = self.seg_decoder(*z)
        seg_results = self.segmentation_head(seg_results)
        output = seg_results.transpose(1, -1).contiguous()
        output = output.view(-1, output.shape[-1])
        labels = labels.transpose(1, -1).contiguous().view(-1)
        # CE loss with ignore index
        loss_ce = self.ce(output, labels)
        # Dice loss with ignore index
        loss_dice = self.dice(output, labels)
        loss = (1/2)*loss_ce + (1/2)*loss_dice

        if train:
            loss.backward()
            self.optimizers()[self.optimizer_idxs.index('optimizer_seg_en')].step()
            self.optimizers()[self.optimizer_idxs.index('optimizer_seg_de')].step()
            self.reset_grad()

        return loss, seg_results