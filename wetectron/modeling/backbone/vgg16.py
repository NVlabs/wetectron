# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict

import torch.nn as nn
from wetectron.modeling import registry
from wetectron.modeling.poolers import Pooler


# to auto-load imagenet pre-trainied weights
class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x


class VGG_Base(nn.Module):
    def __init__(self, features, cfg, init_weights=True):
        super(VGG_Base, self).__init__()
        self.features = features
        if init_weights:
            self._initialize_weights()
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def forward(self, x):
        x = self.features(x)
        return [x]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        assert freeze_at in [1, 2, 3, 4, 5]
        layer_index = [5, 10, 17, 23, 29]
        for layer in range(layer_index[freeze_at - 1]):
            for p in self.features[layer].parameters(): p.requires_grad = False


def make_layers(cfg, dim_in=3, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'I':
            layers += [Identity()]
        # following OICR paper, make conv5_x layers to have dilation=2
        elif isinstance(v, str) and '-D' in v:
            _v = int(v.split('-')[0])
            conv2d = nn.Conv2d(in_channels, _v, kernel_size=3, padding=2, dilation=2)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(_v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = _v          
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    # remove the last relu
    return nn.Sequential(*layers[:-1])


vgg_cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'VGG16-OICR': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'I', '512-D', '512-D', '512-D'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


@registry.BACKBONES.register("VGG16")
@registry.BACKBONES.register("VGG16-OICR")
def add_conv_body(cfg, dim_in=3):
    archi_name = cfg.MODEL.BACKBONE.CONV_BODY
    body = VGG_Base(make_layers(vgg_cfg[archi_name], dim_in), cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = 512
    return model
    

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("VGG16.roi_head")
class VGG16FC67ROIFeatureExtractor(nn.Module):
    def __init__(self, config, in_channels, init_weights=True):
        super(VGG16FC67ROIFeatureExtractor, self).__init__()
        assert in_channels == 512
        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        self.classifier =  nn.Sequential(
            Identity(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True), 
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True), 
            nn.Dropout()
        )
        self.out_channels = 4096
        
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x, proposals):
        # also pool featurs of multiple images into one huge ROI tensor
        x = self.pooler(x, proposals)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward_pooler(self, x, proposals):
        x = self.pooler(x, proposals)
        return x

    def forward_neck(self, x):
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x