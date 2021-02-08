#! /usr/bin/env python

import torch
import torch.nn as nn
from functools import partial
from torchvision import models


class BlinkEstimationAbstractModel(nn.Module):

    def __init__(self):
        super(BlinkEstimationAbstractModel, self).__init__()

    @staticmethod
    def _create_fc_layers(in_features, out_features, dropout_p=0.2):
        fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, out_features),
        )

        return fc

    def forward(self, eye_patch):
        x = self._features(eye_patch)
        x = torch.flatten(x, 1)

        fc_output = self.fc(x)

        return fc_output

    @staticmethod
    def _init_weights(modules):
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)


class BlinkEstimationModelResnet18(BlinkEstimationAbstractModel):

    def __init__(self, num_out=2):
        super(BlinkEstimationModelResnet18, self).__init__()
        _model = models.resnet18(pretrained=True)

        self._features = nn.Sequential(
            _model.conv1,
            _model.bn1,
            _model.relu,
            _model.maxpool,
            _model.layer1,
            _model.layer2,
            _model.layer3,
            _model.layer4,
            _model.avgpool
        )

        for param in self._features.parameters():
            param.requires_grad = True

        self.fc = BlinkEstimationAbstractModel._create_fc_layers(in_features=_model.fc.in_features,
                                                                 out_features=num_out)
        BlinkEstimationAbstractModel._init_weights(self.modules())


class BlinkEstimationModelResnet50(BlinkEstimationAbstractModel):

    def __init__(self, num_out=2):
        super(BlinkEstimationModelResnet50, self).__init__()
        _model = models.resnet50(pretrained=True)

        self._features = nn.Sequential(
            _model.conv1,
            _model.bn1,
            _model.relu,
            _model.maxpool,
            _model.layer1,
            _model.layer2,
            _model.layer3,
            _model.layer4,
            _model.avgpool
        )

        for param in self._features.parameters():
            param.requires_grad = True

        self.fc = BlinkEstimationAbstractModel._create_fc_layers(in_features=_model.fc.in_features,
                                                                 out_features=num_out)
        BlinkEstimationAbstractModel._init_weights(self.modules())


class BlinkEstimationModelVGG(BlinkEstimationAbstractModel):

    def __init__(self, num_out=2):
        super(BlinkEstimationModelVGG, self).__init__()
        _model = models.vgg16(pretrained=True)

        _modules = [module for module in _model.features]
        _modules.append(_model.avgpool)
        self._features = nn.Sequential(*_modules)
        for param in self._features.parameters():
            param.requires_grad = True

        self.fc = BlinkEstimationAbstractModel._create_fc_layers(in_features=_model.classifier[0].in_features,
                                                                 out_features=num_out)
        BlinkEstimationAbstractModel._init_weights(self.modules())


class BlinkEstimationModelVGG19(BlinkEstimationAbstractModel):

    def __init__(self, num_out=2):
        super(BlinkEstimationModelVGG19, self).__init__()
        _model = models.vgg19(pretrained=True)

        _modules = [module for module in _model.features]
        _modules.append(_model.avgpool)
        self._features = nn.Sequential(*_modules)
        for param in self._features.parameters():
            param.requires_grad = True

        self.fc = BlinkEstimationAbstractModel._create_fc_layers(in_features=_model.classifier[0].in_features,
                                                                 out_features=num_out)
        BlinkEstimationAbstractModel._init_weights(self.modules())


class BlinkEstimationModelDenseNet121(BlinkEstimationAbstractModel):

    def __init__(self, num_out=2):
        super(BlinkEstimationModelDenseNet121, self).__init__()
        _model = models.densenet121(pretrained=True)

        _modules = [module for module in _model.features]
        _modules.append(nn.ReLU(inplace=True))
        _modules.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self._features = nn.Sequential(*_modules)
        for param in self._features.parameters():
            param.requires_grad = True

        self.fc = BlinkEstimationAbstractModel._create_fc_layers(in_features=_model.classifier.in_features,
                                                                 out_features=num_out)
        BlinkEstimationAbstractModel._init_weights(self.modules())
