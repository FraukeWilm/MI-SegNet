#https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
import torch
import torchvision

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        vgg_features = torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT').features
        blocks.append(vgg_features[:4].eval())
        blocks.append(vgg_features[4:9].eval())
        blocks.append(vgg_features[9:16].eval())
        blocks.append(vgg_features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        #input = (input-self.mean) / self.std
        #target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
    
class ResNetPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(ResNetPerceptualLoss, self).__init__()
        blocks = []
        resnet = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')
        # SSL pre-trained weight from https://github.com/ozanciga/self-supervised-histopathology?tab=readme-ov-file
        state = torch.load('ckpts/resnet.ckpt', map_location='cpu')
        state_dict = {}
        for key in list(state['state_dict'].keys())[:-4]:
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state['state_dict'].pop(key)
        resnet.load_state_dict(state_dict, strict=False)
        blocks.append(torch.nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool).eval())
        blocks.append(torch.nn.Sequential(resnet.layer1).eval())
        blocks.append(torch.nn.Sequential(resnet.layer2).eval())
        blocks.append(torch.nn.Sequential(resnet.layer3).eval())
        blocks.append(torch.nn.Sequential(resnet.layer4).eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor(state['hyper_parameters']['dataset_mean']).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(state['hyper_parameters']['dataset_std']).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss