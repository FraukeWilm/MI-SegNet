import torch
import torch.nn.functional as F
from taming.model import Encoder, Decoder
from taming.quantize import VectorQuantizer2 as VectorQuantizer
from taming.perceptual_loss import VQLPIPSWithDiscriminator, LPIPS
from module import MISegModule
from wandb.sdk.data_types.image import Image



class VQModule(MISegModule):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        #self.rec_encoder = Encoder(**cfg.model)
        #self.rec_decoder = Decoder(**cfg.model)
        #self.loss = VQLPIPSWithDiscriminator(disc_conditional= False, disc_in_channels= 3, disc_start= 10000,
        #disc_weight= 0.8, codebook_weight= 1.0)
        self.perceptual_loss = LPIPS().eval()
        self.quantize = VectorQuantizer(cfg.model.n_embed, cfg.model.embed_dim, beta=0.25,
                                        remap=None, sane_index_shape=cfg.model.sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(cfg.model.z_channels, cfg.model.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(cfg.model.embed_dim, cfg.model.z_channels, 1)
        self.ae_lr_rate = 4.5e-6*cfg.data.batch_size

    def configure_optimizers(self):
        schedulers = []

        # configure optimizers
        optimizer_mine = torch.optim.Adam(self.mine.parameters(), lr=1e-4)
        optimizer_seg_en = torch.optim.Adam(self.seg_encoder.parameters(), lr=self._lr)
        optimizer_seg_de = torch.optim.Adam(self.seg_decoder.parameters(), lr=self._lr)
        optimizer_rec_en = torch.optim.Adam(list(self.rec_encoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters()),
                                  lr=self.ae_lr_rate, betas=(0.5, 0.9))
        optimizer_rec_de = torch.optim.Adam(
                                  list(self.rec_decoder.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=self.ae_lr_rate, betas=(0.5, 0.9))
        #opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
        #                            lr=self.ae_lr_rate, betas=(0.5, 0.9))
        optimizers = [optimizer_mine, optimizer_rec_en, optimizer_rec_de, optimizer_seg_en, optimizer_seg_de]

        # configure shedulers based on command line parameters
        if self._scheduler == "none":
            return optimizers
        elif self._scheduler == "step":
            schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1) for optimizer in optimizers]
        # add more options if required
        return optimizers, schedulers


    def update_Rec(self, inputs, gt, train):
        z_a = self.seg_encoder(inputs)
        z_d = self.rec_encoder(inputs)
        z_d = self.quant_conv(z_d)
        quant, emb_loss, info = self.quantize(z_d)
        quant = self.post_quant_conv(quant)

        recon_result = self.rec_decoder(z_a, quant)

        rec_loss = F.l1_loss(torch.squeeze(recon_result), torch.squeeze(gt), reduction='mean')
        # Add perceptual loss
        p_loss = self.perceptual_loss(gt.contiguous(), recon_result.contiguous())
        rec_loss = rec_loss + p_loss.mean()
        loss = emb_loss.mean() + rec_loss

        if train:
            loss.backward()
            self.optimizers()[self.optimizer_idxs.index('optimizer_rec_en')].step()
            self.optimizers()[self.optimizer_idxs.index('optimizer_rec_de')].step()
            self.optimizers()[self.optimizer_idxs.index('optimizer_seg_en')].step()
            self.reset_grad()

        return loss, recon_result

    def update_Rec_Adv(self, inputs_1, inputs_2, inputs_12, train):
        z_a = self.seg_encoder(inputs_2)
        z_d = self.rec_encoder(inputs_1)
        z_d = self.quant_conv(z_d)
        quant, emb_loss, info = self.quantize(z_d)
        quant = self.post_quant_conv(quant)

        recon_result = self.rec_decoder(z_a, quant)

        rec_loss = F.l1_loss(torch.squeeze(recon_result), torch.squeeze(inputs_12), reduction='mean')
        # Add perceptual loss
        p_loss = self.perceptual_loss(inputs_12.contiguous(), recon_result.contiguous())
        rec_loss = rec_loss + p_loss.mean()
        loss = emb_loss.mean() + rec_loss

        if train:
            loss.backward()
            self.optimizers()[self.optimizer_idxs.index('optimizer_rec_en')].step()
            self.optimizers()[self.optimizer_idxs.index('optimizer_rec_de')].step()
            self.optimizers()[self.optimizer_idxs.index('optimizer_seg_en')].step()
            self.reset_grad()

        return loss, recon_result


