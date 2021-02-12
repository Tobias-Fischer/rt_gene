#! /usr/bin/env python

import torch
import torch.nn as nn
from torchvision import models


class BlinkEstimationAbstractModel(nn.Module):

    def __init__(self):
        super(BlinkEstimationAbstractModel, self).__init__()

    def _create_fc_layers(self, in_features, out_features, dropout_p=0.6):
        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512, momentum=0.999, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, out_features)
        )

    @staticmethod
    def _init_weights(modules):
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, left_eye, right_eye):
        left_x = self.left_features(left_eye)
        left_x = torch.flatten(left_x, 1)

        right_x = self.right_features(right_eye)
        right_x = torch.flatten(right_x, 1)

        eyes_x = torch.cat((left_x, right_x), dim=1)
        fc_output = self.fc(eyes_x)

        return fc_output


class BlinkEstimationModelResnet18(BlinkEstimationAbstractModel):

    def __init__(self, num_out=1):
        super(BlinkEstimationModelResnet18, self).__init__()
        _left_model = models.resnet18(pretrained=True)
        _right_model = models.resnet18(pretrained=True)

        # remove the last ConvBRelu layer
        self.left_features = nn.Sequential(
            _left_model.conv1,
            _left_model.bn1,
            _left_model.relu,
            _left_model.maxpool,
            _left_model.layer1,
            _left_model.layer2,
            _left_model.layer3,
            _left_model.layer4,
            _left_model.avgpool
        )

        self.right_features = nn.Sequential(
            _right_model.conv1,
            _right_model.bn1,
            _right_model.relu,
            _right_model.maxpool,
            _right_model.layer1,
            _right_model.layer2,
            _right_model.layer3,
            _right_model.layer4,
            _right_model.avgpool
        )

        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True

        self._create_fc_layers(in_features=_left_model.fc.in_features + _right_model.fc.in_features,
                               out_features=num_out)
        self._init_weights(self.modules())


class BlinkEstimationModelResnet50(BlinkEstimationAbstractModel):

    def __init__(self, num_out=1):
        super(BlinkEstimationModelResnet50, self).__init__()
        _left_model = models.resnet50(pretrained=True)
        _right_model = models.resnet50(pretrained=True)

        # remove the last ConvBRelu layer
        self.left_features = nn.Sequential(
            _left_model.conv1,
            _left_model.bn1,
            _left_model.relu,
            _left_model.maxpool,
            _left_model.layer1,
            _left_model.layer2,
            _left_model.layer3,
            _left_model.layer4,
            _left_model.avgpool
        )

        self.right_features = nn.Sequential(
            _right_model.conv1,
            _right_model.bn1,
            _right_model.relu,
            _right_model.maxpool,
            _right_model.layer1,
            _right_model.layer2,
            _right_model.layer3,
            _right_model.layer4,
            _right_model.avgpool
        )

        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True

        self._create_fc_layers(in_features=_left_model.fc.in_features + _right_model.fc.in_features,
                               out_features=num_out)
        self._init_weights(self.modules())


class BlinkEstimationModelVGG16(BlinkEstimationAbstractModel):

    def __init__(self, num_out=1):
        super(BlinkEstimationModelVGG16, self).__init__()
        _left_model = models.vgg16(pretrained=True)
        _right_model = models.vgg16(pretrained=True)

        # remove the last ConvBRelu layer
        _left_modules = [module for module in _left_model.features]
        _left_modules.append(_left_model.avgpool)
        self.left_features = nn.Sequential(*_left_modules)

        _right_modules = [module for module in _right_model.features]
        _right_modules.append(_right_model.avgpool)
        self.right_features = nn.Sequential(*_right_modules)

        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True

        self._create_fc_layers(
            in_features=_left_model.classifier[0].in_features + _right_model.classifier[0].in_features,
            out_features=num_out)
        self._init_weights(self.modules())


class BlinkEstimationModelVGG19(BlinkEstimationAbstractModel):

    def __init__(self, num_out=1):
        super(BlinkEstimationModelVGG19, self).__init__()
        _left_model = models.vgg19(pretrained=True)
        _right_model = models.vgg19(pretrained=True)

        # remove the last ConvBRelu layer
        _left_modules = [module for module in _left_model.features]
        _left_modules.append(_left_model.avgpool)
        self.left_features = nn.Sequential(*_left_modules)

        _right_modules = [module for module in _right_model.features]
        _right_modules.append(_right_model.avgpool)
        self.right_features = nn.Sequential(*_right_modules)

        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True

        self._create_fc_layers(
            in_features=_left_model.classifier[0].in_features + _right_model.classifier[0].in_features,
            out_features=num_out)
        self._init_weights(self.modules())


class BlinkEstimationModelDenseNet121(BlinkEstimationAbstractModel):

    def __init__(self, num_out=1):
        super(BlinkEstimationModelDenseNet121, self).__init__()
        _left_model = models.densenet121(pretrained=True)
        _right_model = models.densenet121(pretrained=True)

        _left_modules = [module for module in _left_model.features]
        _left_modules.append(nn.ReLU(inplace=True))
        _left_modules.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.left_features = nn.Sequential(*_left_modules)

        _right_modules = [module for module in _right_model.features]
        _right_modules.append(nn.ReLU(inplace=True))
        _right_modules.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.right_features = nn.Sequential(*_right_modules)

        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True

        self._create_fc_layers(in_features=_left_model.classifier.in_features + _right_model.classifier.in_features,
                               out_features=num_out)
        self._init_weights(self.modules())
