import torchmetrics
import pytorch_lightning as pl
from MI_SegNet import Mine_Conv, Seg_encoder_LM, Seg_decoder_LM,Recon_encoder_LM, Recon_decoder_LM
from StyleGANDiscriminator.discriminator import Discriminator
from torchvision import transforms
from wandb.sdk.data_types.image import Image
from torchmetrics.classification.jaccard import JaccardIndex
from loss import *
from perceptual_loss import LPIPS
import numpy as np

class MISegModule(pl.LightningModule):
    def __init__(self, cfg, device, **kwargs):
        super().__init__()
        self.automatic_optimization = False
        self._lr = cfg.training.lr
        self._log_imgs = False
        #num_classes = kwargs['num_classes']

        # save all named parameters
        self.save_hyperparameters()

        self.mine = Mine_Conv(in_channels_x=64 * 16, in_channels_y=16 * 16, inter_channels=64, last_pooling_size=int(np.power(2, cfg.data.patch_size//256))).to(device)
        self.rec_encoder = Recon_encoder_LM(in_channels=cfg.model.input_channel, init_features=16).to(device)
        self.rec_decoder = Recon_decoder_LM(in_channels_a=64 * 16, in_channels_d=16 * 16,
                                            out_channels=cfg.model.input_channel, init_features=16).to(device)
        self.seg_encoder = Seg_encoder_LM(cfg.model.input_channel, init_features=64, num_blocks=2).to(device)
        self.seg_decoder = Seg_decoder_LM(cfg.model.output_channel, init_features=64, num_blocks=2).to(device)
        self.transform_image = transforms.Normalize(0.5, 0.5)

        # create loss and metric functions
        self._scheduler = 'none' #step

        self._train_loss_agg = torchmetrics.MeanMetric()
        self._train_disc_loss_agg = torchmetrics.MeanMetric()
        self._val_loss_agg = torchmetrics.MeanMetric()
        self._seg_loss_agg = torchmetrics.MeanMetric()
        self._recon_loss_agg = torchmetrics.MeanMetric()
        self._mi_loss_agg = torchmetrics.MeanMetric()
        self.jaccard_source = JaccardIndex(task="multiclass", num_classes=cfg.model.output_channel, average='none', ignore_index=-1)
        self.jaccard_target = JaccardIndex(task="multiclass", num_classes=cfg.model.output_channel, average='none', ignore_index=-1)

        # intialize loss functions and weights
        self.ce = CELoss()
        self.dice = DiceLoss()
        self.l1 = L1Loss()
        self.w_l1 = cfg.loss.w_l1
        self.ssim = SSIMLoss(device=device)
        self.w_ssim = cfg.loss.w_ssim
        self.pl = LPIPS()
        self.w_p = cfg.loss.w_p
        self.d_loss = DiscLoss()
        self.w_d = cfg.loss.w_d

        # optimizers
        self.optimizer_idxs = ['optimizer_mine', 'optimizer_rec_en', 'optimizer_rec_de', 'optimizer_seg_en', 'optimizer_seg_de']
        if self.w_d > 0:
            self.optimizer_idxs.append('optimizer_disc')
            self.discriminator = Discriminator(size=cfg.data.patch_size)


    def forward(self, x):
        # the method used for inference
        output = self._model(x)
        return output

    def training_step(self, batch, batch_idx):
        inputs_1 = batch[0].float().to(self.device) # domain 1 anatomy 1
        inputs_2 = batch[1].float().to(self.device) # domain 2 anatomy 2
        inputs_12 = batch[2].float().to(self.device) # domain 1 anatomy 2
        inputs_21 = batch[3].float().to(self.device) # domain 2 anatomy 1
        labels_1 = batch[4].to(self.device)
        labels_2 = batch[5].to(self.device)

        inputs_1_trans = self.transform_image(inputs_1)
        inputs_2_trans = self.transform_image(inputs_2)

        seg_loss_1, seg_results_1 = self.update_Seg(inputs_1_trans, labels_1, True)
        seg_loss_2, seg_results_2 = self.update_Seg(inputs_2_trans, labels_2, True)
        seg_loss = seg_loss_1 + seg_loss_2

        recon_loss_1, rec_results_1 = self.update_Rec(inputs_1_trans, inputs_1, True)
        recon_loss_2, rec_results_2 = self.update_Rec(inputs_2_trans, inputs_2, True)
        recon_loss = recon_loss_1 + recon_loss_2

        rec_adv_loss_1, rec_results_12 = self.update_Rec_Adv(inputs_1_trans, inputs_2_trans, inputs_12, True)
        rec_adv_loss_2, rec_results_21 = self.update_Rec_Adv(inputs_2_trans, inputs_1_trans, inputs_21, True)
        rec_adv_loss = rec_adv_loss_1 + rec_adv_loss_2

        mi_loss_1 = self.update_MI(inputs_1_trans, True)
        mi_loss_2 = self.update_MI(inputs_2_trans, True)
        mi_loss = mi_loss_1 + mi_loss_2

        for _ in range(5):
            learn_mi_loss_1 = self.learn_mine(inputs_1_trans)
            learn_mi_loss_2 = self.learn_mine(inputs_2_trans)

        # Train discriminator
        if self.w_d > 0:
            disc_loss_1 = self.update_Disc(inputs_1_trans)
            disc_loss_2 = self.update_Disc(inputs_2_trans)
            self._train_disc_loss_agg(disc_loss_1 + disc_loss_2)

        loss = seg_loss + recon_loss + rec_adv_loss
        self._train_loss_agg.update(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs_1 = batch[0].float().to(self.device) # domain 1 anatomy 1
        inputs_2 = batch[1].float().to(self.device)  # domain 2 anatomy 2
        inputs_12 = batch[2].float().to(self.device) # domain 1 anatomy 2
        inputs_21 = batch[3].float().to(self.device) # domain 2 anatomy 1
        labels_1 = batch[4].to(self.device)
        labels_2 = batch[5].to(self.device)

        inputs_1_trans = self.transform_image(inputs_1)
        inputs_2_trans = self.transform_image(inputs_2)

        # forward
        seg_loss_1, seg_results_1 = self.update_Seg(inputs_1_trans, labels_1, False)
        seg_loss_2, seg_results_2 = self.update_Seg(inputs_2_trans, labels_2, False)
        seg_loss = seg_loss_1 + seg_loss_2

        recon_loss_1, rec_results_1 = self.update_Rec(inputs_1_trans, inputs_1, False)
        recon_loss_2, rec_results_2 = self.update_Rec(inputs_2_trans, inputs_2, False)
        recon_loss = recon_loss_1 + recon_loss_2

        rec_adv_loss_1, rec_results_12 = self.update_Rec_Adv(inputs_1_trans, inputs_2_trans, inputs_12, False)
        rec_adv_loss_2, rec_results_21 = self.update_Rec_Adv(inputs_2_trans, inputs_1_trans, inputs_21, False)
        rec_adv_loss = rec_adv_loss_1 + rec_adv_loss_2

        mi_loss_1 = self.update_MI(inputs_1_trans, False)
        mi_loss_2 = self.update_MI(inputs_2_trans, False)
        mi_loss = mi_loss_1 + mi_loss_2

        val_loss = seg_loss + recon_loss + rec_adv_loss
        self._val_loss_agg.update(val_loss)
        self._seg_loss_agg.update(seg_loss)
        self._recon_loss_agg.update(recon_loss)
        self._mi_loss_agg.update(mi_loss)
        self.jaccard_source.update(seg_results_1, labels_1)
        self.jaccard_target.update(seg_results_2, labels_2)

        # we could also log example images to wandb here
        if batch_idx == 0 and self.current_epoch % 10 ==0:
            self.log_images(inputs_1, inputs_2, inputs_12, inputs_21, labels_1, labels_2, seg_results_1, seg_results_2, rec_results_1, rec_results_2, rec_results_12, rec_results_21)
        return val_loss

    def log_images(self, inputs_1, inputs_2, inputs_12, inputs_21, labels_1, labels_2, seg_results_1, seg_results_2, rec_results_1, rec_results_2, rec_results_12, rec_results_21):
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
                rec_img = Image(255 * torch.cat((rec_results_1[i], rec_results_2[i], rec_results_12[i], rec_results_21[i]), dim=-1).permute(1, 2, 0).cpu().numpy())
                self.logger.log_metrics({"Input": input})
                self.logger.log_metrics({"Segmentation Output": mask_img})
                self.logger.log_metrics({"Reconstruction Output": rec_img})


    def training_epoch_end(self, outputs):
        # required if values returned in the training_steps have to be processed in a specific way
        self.log("Train Loss", self._train_loss_agg.compute(), sync_dist=True)
        self._train_loss_agg.reset()
        self.log("Disc Loss", self._train_disc_loss_agg.compute(), sync_dist=True)
        self._train_disc_loss_agg.reset()

    def validation_epoch_end(self, outputs):
        # required if values returned in the validation_step have to be processed in a specific way
        self.log("Val Loss", self._val_loss_agg.compute(), sync_dist=True)
        self._val_loss_agg.reset()

        # log partial losses
        self.log("Seg Loss", self._seg_loss_agg.compute(), sync_dist=True)
        self._seg_loss_agg.reset()
        self.log("Recon Loss", self._recon_loss_agg.compute(), sync_dist=True)
        self._recon_loss_agg.reset()
        self.log("MI Loss", self._mi_loss_agg.compute(), sync_dist=True)
        self._mi_loss_agg.reset()

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
        optimizer_mine = torch.optim.Adam(self.mine.parameters(), lr=1e-4)
        optimizer_rec_en = torch.optim.Adam(self.rec_encoder.parameters(), lr=self._lr)
        optimizer_rec_de = torch.optim.Adam(self.rec_decoder.parameters(), lr=self._lr)
        optimizer_seg_en = torch.optim.Adam(self.seg_encoder.parameters(), lr=self._lr)
        optimizer_seg_de = torch.optim.Adam(self.seg_decoder.parameters(), lr=self._lr)
        optimizers = [optimizer_mine, optimizer_rec_en, optimizer_rec_de, optimizer_seg_en, optimizer_seg_de]

        if self.w_d > 0:
            optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self._lr)
            optimizers.append(optimizer_disc)

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
        seg_results = self.seg_decoder(z)
        output = seg_results.transpose(1, -1).contiguous()
        output = output.view(-1, output.shape[-1])
        labels = labels.transpose(1, -1).contiguous().view(-1)
        # CE loss with ignore index
        loss_ce = self.ce(output, labels)
        # Dice loss with ignore index
        loss_dice = self.dice(output, labels)
        loss = (1/3)*loss_ce + (2/3)*loss_dice

        if train:
            loss.backward()
            self.optimizers()[self.optimizer_idxs.index('optimizer_seg_en')].step()
            self.optimizers()[self.optimizer_idxs.index('optimizer_seg_de')].step()
            self.reset_grad()

        return loss, seg_results

    def update_Rec(self, inputs, gt, train):
        z_a = self.seg_encoder(inputs)
        z_d = self.rec_encoder(inputs)

        recon_result = self.rec_decoder(z_a, z_d)

        l1_loss = self.l1(recon_result, gt)
        ssim_loss = 1 - self.ssim(recon_result, gt)
        p_loss = self.pl(recon_result.contiguous(), gt.contiguous()).mean()
        rec_loss = self.w_l1*l1_loss + self.w_ssim*ssim_loss + self.w_p*p_loss

        if self.w_d > 0:
            pred_rec = self.discriminator(recon_result)

            # For generator training original image patch and reconstruction should be misclassified
            disc_loss = self.d_loss(pred_rec, should_be_classified_as_real=True)
            rec_loss += self.w_d*disc_loss

        rec_loss /= self.w_l1+self.w_ssim+self.w_p+self.w_d


        if train:
            rec_loss.backward()
            self.optimizers()[self.optimizer_idxs.index('optimizer_rec_en')].step()
            self.optimizers()[self.optimizer_idxs.index('optimizer_rec_de')].step()
            self.optimizers()[self.optimizer_idxs.index('optimizer_seg_en')].step()
            self.reset_grad()

        return rec_loss, recon_result

    def update_Rec_Adv(self, inputs_1, inputs_2, inputs_12, train):
        z_a = self.seg_encoder(inputs_2)
        z_d = self.rec_encoder(inputs_1)

        recon_result = self.rec_decoder(z_a, z_d)

        l1_loss = self.l1(recon_result, inputs_12)
        ssim_loss = 1 - self.ssim(recon_result, inputs_12)
        p_loss = self.pl(recon_result.contiguous(), inputs_12.contiguous()).mean()
        rec_loss = self.w_l1 * l1_loss + self.w_ssim * ssim_loss + self.w_p * p_loss

        if self.w_d > 0:
            pred_rec = self.discriminator(recon_result)

            # For generator training original image patch and reconstruction should be misclassified
            disc_loss = self.d_loss(pred_rec, should_be_classified_as_real=True)
            rec_loss += self.w_d*disc_loss

        rec_loss /= self.w_l1+self.w_ssim+self.w_p+self.w_d

        if train:
            rec_loss.backward()
            self.optimizers()[self.optimizer_idxs.index('optimizer_rec_en')].step()
            self.optimizers()[self.optimizer_idxs.index('optimizer_rec_de')].step()
            self.optimizers()[self.optimizer_idxs.index('optimizer_seg_en')].step()
            self.reset_grad()

        return rec_loss, recon_result

    def update_Disc(self, inputs):
        with torch.no_grad():
            z_a = self.seg_encoder(inputs)
            z_d = self.rec_encoder(inputs)

            recon_result = self.rec_decoder(z_a, z_d)
        pred_real = self.discriminator(inputs)
        pred_rec = self.discriminator(recon_result)

        loss_real = self.d_loss(pred_real, should_be_classified_as_real=True)
        loss_rec = self.d_loss(pred_rec, should_be_classified_as_real=False)

        disc_loss = loss_real + loss_rec

        disc_loss.backward()
        self.optimizers()[self.optimizer_idxs.index('optimizer_disc')].step()
        self.reset_grad()

        return disc_loss


    def update_MI(self, inputs, train):
        z_a = self.seg_encoder(inputs)
        z_d = self.rec_encoder(inputs)

        z_d_shuffle = torch.index_select(z_d, 0, torch.randperm(z_d.shape[0]).to(self.device))

        mutual_loss, _, _ = self.mi_estimator(z_a, z_d, z_d_shuffle)

        mutual_loss = F.elu(mutual_loss)

        if train:
            mutual_loss.backward()
            self.optimizers()[self.optimizer_idxs.index('optimizer_rec_en')].step()
            self.optimizers()[self.optimizer_idxs.index('optimizer_seg_en')].step()
            self.reset_grad()

        return mutual_loss

    def learn_mine(self, inputs, ma_rate=0.001):
        with torch.no_grad():
            z_a = self.seg_encoder(inputs)
            z_d = self.rec_encoder(inputs)

            z_d_shuffle = torch.index_select(z_d, 0, torch.randperm(z_d.shape[0]).to(self.device))

        et = torch.mean(torch.exp(self.mine(z_a, z_d_shuffle)))
        if self.mine.ma_et is None:
            self.mine.ma_et = et.detach().item()
            self.mine.ma_et += ma_rate * (et.detach().item() - self.mine.ma_et)
        mutual_information = torch.mean(self.mine(z_a, z_d)) - torch.log(et) * et.detach() / self.mine.ma_et

        loss = -mutual_information

        loss.backward()
        self.optimizers()[self.optimizer_idxs.index('optimizer_mine')].step()
        self.reset_grad()

        return mutual_information

    def mi_estimator(self, x, y, y_):
        joint, marginal = self.mine(x, y), self.mine(x, y_)
        return torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal))), joint, marginal