
import torch
import torch.nn as nn
from torchvision import models


class RTGENEModelVGG(nn.Module):

    def __init__(self, num_out=2):  # phi, theta
        super(RTGENEModelVGG, self).__init__()
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
    import numpy as np
    from PIL import Image
    from torchvision import transforms
    import os
    import time
    from tqdm import trange

    left_img = Image.open(os.path.abspath(os.path.join("../../RT_GENE/s001_glasses/", "inpainted/left/", "left_000004_rgb.png")))
    right_img = Image.open(os.path.abspath(os.path.join("../../RT_GENE/s001_glasses/", "inpainted/right/", "right_000004_rgb.png")))
    head_pose_gen = torch.from_numpy(np.random.random(2)).unsqueeze(0).float().cuda()

    trans = transforms.Compose([transforms.Resize(256, Image.BICUBIC),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

    trans_left_img = trans(left_img).unsqueeze(0).float().cuda()
    trans_right_img = trans(right_img).unsqueeze(0).float().cuda()

    model = RTGENEModelVGG()
    model = model.cuda()
    model.eval()
    start_time = time.time()
    outputs = []
    for _ in trange(100):
        outputs.append(model(trans_left_img, trans_right_img, head_pose_gen))
    print("Evaluation Frequency: {:.3f}Hz".format(1.0 / ((time.time() - start_time) / 100.0)))
