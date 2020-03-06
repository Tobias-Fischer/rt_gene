import torch
import torch.nn as nn
from torchvision import models


class RTGENEModelResnet50(nn.Module):

    def __init__(self, num_out=2):  # phi, theta
        super(RTGENEModelResnet50, self).__init__()
        _left_model = models.resnet50(pretrained=True)
        _right_model = models.resnet50(pretrained=True)

        # remove the last ConvBRelu layer
        self.left_features = nn.Sequential(
            *list([_left_model.conv1,
                   _left_model.bn1,
                   _left_model.relu,
                   _left_model.maxpool,
                   _left_model.layer1,
                   _left_model.layer2,
                   _left_model.layer3,
                   _left_model.layer4,
                   _left_model.avgpool])
        )

        self.right_features = nn.Sequential(
            *list([_right_model.conv1,
                   _right_model.bn1,
                   _right_model.relu,
                   _right_model.maxpool,
                   _right_model.layer1,
                   _right_model.layer2,
                   _right_model.layer3,
                   _right_model.layer4,
                   _right_model.avgpool])
        )

        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True

        _num_ftrs = _left_model.fc.in_features + _right_model.fc.in_features + 2  # left, right and head_pose
        self.classifier = nn.Sequential(
            nn.Linear(_num_ftrs, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_out)
        )

        # self.init_weights()

    def init_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, left_eye, right_eye, headpose):
        left_x = self.left_features(left_eye)
        left_x = torch.flatten(left_x, 1)

        right_x = self.right_features(right_eye)
        right_x = torch.flatten(right_x, 1)

        concat = torch.cat((left_x, right_x, headpose), dim=1)

        fc_output = self.classifier(concat)

        return fc_output


class RTGENEModelResnet101(nn.Module):

    def __init__(self, num_out=2):  # phi, theta
        super(RTGENEModelResnet101, self).__init__()
        _left_model = models.resnet101(pretrained=True)
        _right_model = models.resnet101(pretrained=True)

        # remove the last ConvBRelu layer
        self.left_features = nn.Sequential(
            *list([_left_model.conv1,
                   _left_model.bn1,
                   _left_model.relu,
                   _left_model.maxpool,
                   _left_model.layer1,
                   _left_model.layer2,
                   _left_model.layer3,
                   _left_model.layer4,
                   _left_model.avgpool])
        )

        self.right_features = nn.Sequential(
            *list([_right_model.conv1,
                   _right_model.bn1,
                   _right_model.relu,
                   _right_model.maxpool,
                   _right_model.layer1,
                   _right_model.layer2,
                   _right_model.layer3,
                   _right_model.layer4,
                   _right_model.avgpool])
        )

        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True

        _num_ftrs = _left_model.fc.in_features + _right_model.fc.in_features + 2 # left, right and head_pose
        self.classifier = nn.Sequential(
            nn.Linear(_num_ftrs, _num_ftrs),
            nn.BatchNorm1d(_num_ftrs),
            nn.ReLU(),
            nn.Linear(_num_ftrs, _num_ftrs),
            nn.BatchNorm1d(_num_ftrs),
            nn.ReLU(),
            nn.Linear(_num_ftrs, num_out)
        )

        # self.init_weights()

    def init_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, left_eye, right_eye, headpose):
        left_x = self.left_features(left_eye)
        left_x = torch.flatten(left_x, 1)

        right_x = self.right_features(right_eye)
        right_x = torch.flatten(right_x, 1)

        concat = torch.cat((left_x, right_x, headpose), dim=1)

        fc_output = self.classifier(concat)

        return fc_output


class RTGENEModelResnet18(nn.Module):

    def __init__(self, num_out=3):
        super(RTGENEModelResnet18, self).__init__()
        _left_model = models.resnet18(pretrained=True)
        _right_model = models.resnet18(pretrained=True)

        # remove the last ConvBRelu layer
        self.left_features = nn.Sequential(
            *list([_left_model.conv1,
                   _left_model.bn1,
                   _left_model.relu,
                   _left_model.maxpool,
                   _left_model.layer1,
                   _left_model.layer2,
                   _left_model.layer3,
                   _left_model.layer4,
                   _left_model.avgpool])
        )

        self.right_features = nn.Sequential(
            *list([_right_model.conv1,
                   _right_model.bn1,
                   _right_model.relu,
                   _right_model.maxpool,
                   _right_model.layer1,
                   _right_model.layer2,
                   _right_model.layer3,
                   _right_model.layer4,
                   _right_model.avgpool])
        )

        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True

        _num_ftrs = _left_model.fc.in_features + _right_model.fc.in_features + 2  # left, right and head_pose
        self.classifier = nn.Sequential(
            nn.Linear(_num_ftrs, _num_ftrs),
            nn.BatchNorm1d(_num_ftrs),
            nn.ReLU(),
            nn.Linear(_num_ftrs, _num_ftrs),
            nn.BatchNorm1d(_num_ftrs),
            nn.ReLU(),
            nn.Linear(_num_ftrs, num_out)
        )

        # self.init_weights()

    def init_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, left_eye, right_eye, headpose):
        left_x = self.left_features(left_eye)
        left_x = torch.flatten(left_x, 1)

        right_x = self.right_features(right_eye)
        right_x = torch.flatten(right_x, 1)

        concat = torch.cat((left_x, right_x, headpose), dim=1)

        fc_output = self.classifier(concat)

        return fc_output


class RTGENEModelMobileNetV2(nn.Module):

    def __init__(self, num_out=2):  # phi, theta
        super(RTGENEModelMobileNetV2, self).__init__()
        _left_model = models.mobilenet_v2(pretrained=True)
        _right_model = models.mobilenet_v2(pretrained=True)
        _adaptive_max_pooling = torch.nn.AdaptiveAvgPool1d(1024)

        _left_features = [module for module in _left_model.features.children()]
        self.left_features = nn.Sequential(*_left_features)

        _right_features = [module for module in _right_model.features.children()]
        self.right_features = nn.Sequential(
            *list(_right_model.features.children())
        )

        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True

        _num_ftrs = _left_model.classifier[1].in_features + _right_model.classifier[1].in_features + 2  # left, right and head_pose

        # hourglass fc network as per original paper
        self.classifier = nn.Sequential(
            nn.Linear(_num_ftrs, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_out)
        )

       # self.init_weights()

    def init_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, left_eye, right_eye, headpose):
        left_x = self.left_features(left_eye)
        left_x = left_x.mean([2, 3])

        right_x = self.right_features(right_eye)
        right_x = right_x.mean([2, 3])

        concat = torch.cat((left_x, right_x, headpose), dim=1)

        fc_output = self.classifier(concat)

        # angular_output = fc_output[:, :2]
        #
        # sigma = fc_output[:, 2:3]
        # sigma = sigma.view(-1, 1).expand(sigma.size(0), 2)

        return fc_output


class RTGENEModelVGG(nn.Module):

    def __init__(self, num_out=2):  # phi, theta
        super(RTGENEModelVGG, self).__init__()
        _left_model = models.vgg16_bn(pretrained=True)
        _right_model = models.vgg16_bn(pretrained=True)

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

        _num_ftrs = _left_model.classifier[0].in_features + _right_model.classifier[0].in_features + 2  # left, right and head_pose
        self.classifier = nn.Sequential(
            nn.Linear(_num_ftrs, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_out)
        )

        # self.init_weights()

    def init_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, left_eye, right_eye, head_pose):
        left_x = self.left_features(left_eye)
        left_x = torch.flatten(left_x, 1)

        right_x = self.right_features(right_eye)
        right_x = torch.flatten(right_x, 1)

        concat = torch.cat((left_x, right_x, head_pose), dim=1)

        fc_output = self.classifier(concat)

        return fc_output


if __name__ == "__main__":
    from PIL import Image
    from torchvision import transforms
    import os
    import time
    from tqdm import trange
    import numpy as np

    torch.backends.cudnn.benchmark = True

    left_img = Image.open(os.path.abspath(os.path.join("../../../RT_GENE/s001_glasses/", "inpainted/left/", "left_000004_rgb.png")))
    right_img = Image.open(os.path.abspath(os.path.join("../../../RT_GENE/s001_glasses/", "inpainted/right/", "right_000004_rgb.png")))
    head_pose_gen = torch.from_numpy(np.random.random(2)).unsqueeze(0).float().cuda()

    trans = transforms.Compose([transforms.Resize((224, 224), Image.BICUBIC),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

    trans_left_img = trans(left_img).unsqueeze(0).cuda()
    trans_right_img = trans(right_img).unsqueeze(0).cuda()

    model = RTGENEModelResnet18()
    model = model.cuda()
    model.eval()
    start_time = time.time()
    for _ in trange(5000):
        model(trans_left_img, trans_right_img, head_pose_gen)
    print("Evaluation Frequency: {:.3f}Hz".format(1.0 / ((time.time() - start_time) / 5000.0)))
