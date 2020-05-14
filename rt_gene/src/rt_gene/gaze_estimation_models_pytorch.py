import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GazeEstimationAbstractModel(nn.Module):

    def __init__(self):
        super(GazeEstimationAbstractModel, self).__init__()

    @staticmethod
    def _create_fc_layers(in_features, out_features):
        x_l = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        x_r = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )

        concat = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )

        fc = nn.Sequential(
            nn.Linear(514, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_features)
        )

        return x_l, x_r, concat, fc

    def forward(self, left_eye, right_eye, headpose):
        left_x = self.left_features(left_eye)
        left_x = torch.flatten(left_x, 1)
        left_x = self.xl(left_x)

        right_x = self.right_features(right_eye)
        right_x = torch.flatten(right_x, 1)
        right_x = self.xr(right_x)

        eyes_x = torch.cat((left_x, right_x), dim=1)
        eyes_x = self.concat(eyes_x)

        eyes_headpose = torch.cat((eyes_x, headpose), dim=1)

        fc_output = self.fc(eyes_headpose)

        return fc_output

    @staticmethod
    def _init_weights(modules):
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)


class GazeEstimationModelResnet18(GazeEstimationAbstractModel):

    def __init__(self, num_out=2):
        super(GazeEstimationModelResnet18, self).__init__()
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

        self.xl, self.xr, self.concat, self.fc = GazeEstimationAbstractModel._create_fc_layers(in_features=_left_model.fc.in_features, out_features=num_out)
        GazeEstimationAbstractModel._init_weights(self.modules())


class GazeEstimationModelPreactResnet(GazeEstimationAbstractModel):
    class PreactResnet(nn.Module):
        class BasicBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride):
                super().__init__()

                self.bn1 = nn.BatchNorm2d(in_channels)
                self.conv1 = nn.Conv2d(in_channels,
                                       out_channels,
                                       kernel_size=3,
                                       stride=stride,
                                       padding=1,
                                       bias=False)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels,
                                       out_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       bias=False)

                self.shortcut = nn.Sequential()
                if in_channels != out_channels:
                    self.shortcut.add_module(
                        'conv',
                        nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size=1,
                                  stride=stride,
                                  padding=0,
                                  bias=False))

            def forward(self, x):
                x = F.relu(self.bn1(x), inplace=True)
                y = self.conv1(x)
                y = F.relu(self.bn2(y), inplace=True)
                y = self.conv2(y)
                y += self.shortcut(x)
                return y

        def __init__(self, depth=30, base_channels=16, input_shape=(1, 3, 224, 224)):
            super().__init__()

            n_blocks_per_stage = (depth - 2) // 6
            n_channels = [base_channels, base_channels * 2, base_channels * 4]

            self.conv = nn.Conv2d(input_shape[1],
                                  n_channels[0],
                                  kernel_size=(3, 3),
                                  stride=1,
                                  padding=1,
                                  bias=False)

            self.stage1 = self._make_stage(n_channels[0],
                                           n_channels[0],
                                           n_blocks_per_stage,
                                           GazeEstimationModelPreactResnet.PreactResnet.BasicBlock,
                                           stride=1)
            self.stage2 = self._make_stage(n_channels[0],
                                           n_channels[1],
                                           n_blocks_per_stage,
                                           GazeEstimationModelPreactResnet.PreactResnet.BasicBlock,
                                           stride=2)
            self.stage3 = self._make_stage(n_channels[1],
                                           n_channels[2],
                                           n_blocks_per_stage,
                                           GazeEstimationModelPreactResnet.PreactResnet.BasicBlock,
                                           stride=2)
            self.bn = nn.BatchNorm2d(n_channels[2])

            self._init_weights(self.modules())

        @staticmethod
        def _init_weights(module):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

        @staticmethod
        def _make_stage(in_channels, out_channels, n_blocks, block, stride):
            stage = nn.Sequential()
            for index in range(n_blocks):
                block_name = "block{}".format(index + 1)
                if index == 0:
                    stage.add_module(block_name, block(in_channels, out_channels, stride=stride))
                else:
                    stage.add_module(block_name, block(out_channels, out_channels, stride=1))
            return stage

        def forward(self, x):
            x = self.conv(x)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = F.relu(self.bn(x), inplace=True)
            x = F.adaptive_avg_pool2d(x, output_size=1)
            return x

    def __init__(self, num_out=2):
        super(GazeEstimationModelPreactResnet, self).__init__()
        self.left_features = GazeEstimationModelPreactResnet.PreactResnet()
        self.right_features = GazeEstimationModelPreactResnet.PreactResnet()

        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True

        self.xl, self.xr, self.concat, self.fc = GazeEstimationAbstractModel._create_fc_layers(in_features=64, out_features=num_out)
        GazeEstimationAbstractModel._init_weights(self.modules())


class GazeEstimationModelVGG(GazeEstimationAbstractModel):

    def __init__(self, num_out=2):
        super(GazeEstimationModelVGG, self).__init__()
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

        self.xl, self.xr, self.concat, self.fc = GazeEstimationAbstractModel._create_fc_layers(in_features=_left_model.classifier[0].in_features,
                                                                                               out_features=num_out)
        GazeEstimationAbstractModel._init_weights(self.modules())
