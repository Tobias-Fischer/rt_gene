import torch
import torch.nn as nn
from torchvision import models


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


if __name__ == "__main__":
    from PIL import Image
    from torchvision import transforms
    import os
    import time
    from tqdm import trange

    left_img = Image.open(os.path.abspath(os.path.join("/home/ahmed/Documents/RT_GENE/s001_glasses/", "inpainted/left_new/", "left_000004_rgb.png")))
    right_img = Image.open(os.path.abspath(os.path.join("/home/ahmed/Documents/RT_GENE/s001_glasses/", "inpainted/right_new/", "right_000004_rgb.png")))

    trans = transforms.Compose([transforms.Resize(256, Image.BICUBIC),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

    trans_left_img = trans(left_img).unsqueeze(0).cuda()
    trans_right_img = trans(right_img).unsqueeze(0).cuda()

    model = RTGENEModelMobileNetV2()
    model = model.cuda()
    model.eval()
    start_time = time.time()
    for _ in trange(1000):
        model(trans_left_img, trans_right_img).cpu()
    print("Evaluation Frequency: {:.3f}".format(1.0 / ((time.time() - start_time) / 100.0)))
