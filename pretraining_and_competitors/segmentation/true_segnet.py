import torch.nn.functional as F
from torch import nn
from torchvision import models


class SegNet(nn.Module):
    def __init__(self, num_classes, pretrained=False, fix_weights=False):
        super(SegNet, self).__init__()
        self.name = "true_segnet"
        self.num_classes = num_classes
        # vgg = models.vgg16_bn(pretrained=pretrained)
        # features = list(vgg.features.children())
        # maxpool = nn.MaxPool2d(2, 2, return_indices=True)

        # Encoder
        self.conv11 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(64)

        self.conv21 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn21 = nn.BatchNorm2d(128)
        self.conv22 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn22 = nn.BatchNorm2d(128)

        self.conv31 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn31 = nn.BatchNorm2d(256)
        self.conv32 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn32 = nn.BatchNorm2d(256)
        self.conv33 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn33 = nn.BatchNorm2d(256)

        self.conv41 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn41 = nn.BatchNorm2d(512)
        self.conv42 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn42 = nn.BatchNorm2d(512)
        self.conv43 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn43 = nn.BatchNorm2d(512)

        self.conv51 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn51 = nn.BatchNorm2d(512)
        self.conv52 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn52 = nn.BatchNorm2d(512)
        self.conv53 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn53 = nn.BatchNorm2d(512)

        # Decoder
        self.conv53d = nn.Conv2d(512, 512, 3, padding=1)
        self.bn53d = nn.BatchNorm2d(512)
        self.conv52d = nn.Conv2d(512, 512, 3, padding=1)
        self.bn52d = nn.BatchNorm2d(512)
        self.conv51d = nn.Conv2d(512, 512, 3, padding=1)
        self.bn51d = nn.BatchNorm2d(512)

        self.conv43d = nn.Conv2d(512, 512, 3, padding=1)
        self.bn43d = nn.BatchNorm2d(512)
        self.conv42d = nn.Conv2d(512, 512, 3, padding=1)
        self.bn42d = nn.BatchNorm2d(512)
        self.conv41d = nn.Conv2d(512, 256, 3, padding=1)
        self.bn41d = nn.BatchNorm2d(256)

        self.conv33d = nn.Conv2d(256, 256, 3, padding=1)
        self.bn33d = nn.BatchNorm2d(256)
        self.conv32d = nn.Conv2d(256, 256, 3, padding=1)
        self.bn32d = nn.BatchNorm2d(256)
        self.conv31d = nn.Conv2d(256, 128, 3, padding=1)
        self.bn31d = nn.BatchNorm2d(128)

        self.conv22d = nn.Conv2d(128, 128, 3, padding=1)
        self.bn22d = nn.BatchNorm2d(128)
        self.conv21d = nn.Conv2d(128, 64, 3, padding=1)
        self.bn21d = nn.BatchNorm2d(64)

        self.conv12d = nn.Conv2d(64, 64, 3, padding=1)
        self.bn12d = nn.BatchNorm2d(64)
        self.conv11d = nn.Conv2d(64, num_classes, 3, padding=1)

        if pretrained:
            self.initialize_weights(fix_weights)

    def forward(self, x):
        # maxpool = nn.MaxPool2d(2, 2, return_indices=True)
        # maxunpool = nn.MaxUnpool2d(2, 2)

        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        size1 = x.size()
        x, id1 = F.max_pool2d(x, 2, 2, return_indices=True)

        x = F.relu(self.bn21(self.conv21(x)))
        x = F.relu(self.bn22(self.conv22(x)))
        size2 = x.size()
        x, id2 = F.max_pool2d(x, 2, 2, return_indices=True)

        x = F.relu(self.bn31(self.conv31(x)))
        x = F.relu(self.bn32(self.conv32(x)))
        x = F.relu(self.bn33(self.conv33(x)))
        size3 = x.size()
        x, id3 = F.max_pool2d(x, 2, 2, return_indices=True)

        x = F.relu(self.bn41(self.conv41(x)))
        x = F.relu(self.bn42(self.conv42(x)))
        x = F.relu(self.bn43(self.conv43(x)))
        size4 = x.size()
        x, id4 = F.max_pool2d(x, 2, 2, return_indices=True)

        x = F.relu(self.bn51(self.conv51(x)))
        x = F.relu(self.bn52(self.conv52(x)))
        x = F.relu(self.bn53(self.conv53(x)))
        size5 = x.size()
        x, id5 = F.max_pool2d(x, 2, 2, return_indices=True)

        # Decoding
        x = F.max_unpool2d(x, id5, 2, 2, output_size=size5)
        x = F.relu(self.bn53d(self.conv53d(x)))
        x = F.relu(self.bn52d(self.conv52d(x)))
        x = F.relu(self.bn51d(self.conv51d(x)))

        x = F.max_unpool2d(x, id4, 2, 2, output_size=size4)
        x = F.relu(self.bn43d(self.conv43d(x)))
        x = F.relu(self.bn42d(self.conv42d(x)))
        x = F.relu(self.bn41d(self.conv41d(x)))

        x = F.max_unpool2d(x, id3, 2, 2, output_size=size3)
        x = F.relu(self.bn33d(self.conv33d(x)))
        x = F.relu(self.bn32d(self.conv32d(x)))
        x = F.relu(self.bn31d(self.conv31d(x)))

        x = F.max_unpool2d(x, id2, 2, 2, output_size=size2)
        x = F.relu(self.bn22d(self.conv22d(x)))
        x = F.relu(self.bn21d(self.conv21d(x)))

        x = F.max_unpool2d(x, id1, 2, 2, output_size=size1)
        x = F.relu(self.bn12d(self.conv12d(x)))
        x = self.conv11d(x)
        return x

    def initialize_weights(self, fix_weights):
        vgg = models.vgg16_bn(pretrained=True)
        vgg_layers = [c for c in vgg.features.children() if isinstance(c, nn.Conv2d) or isinstance(c, nn.BatchNorm2d)]

        segnet_layers = list(self._modules.values())
        i = 0
        for l in vgg_layers:
            if isinstance(l, nn.Conv2d) or isinstance(l, nn.BatchNorm2d):
                segnet_layers[i].weight = l.weight
                segnet_layers[i].bias = l.bias
                if isinstance(l, nn.BatchNorm2d):
                    segnet_layers[i].running_mean = l.running_mean
                    segnet_layers[i].running_var = l.running_var
                # Fix weights of VGG
                if fix_weights:
                    for param in segnet_layers[i].parameters():
                        param.requires_grad = False
                i += 1
