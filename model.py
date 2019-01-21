from math import floor
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from SNlayers import SNConv2d, SNLinear, MeanSpectralNorm
from DenseNet_layers import _DenseLayer, _DenseBlock, _Transition,_SNDenseLayer, _SNTransition,_MSNTransition,_MSNDenseLayer

class ConvNet(nn.Module):
    def __init__(self, nc, ndf, data_size, num_classes, num_hidden_layers=8, spectral_norm = True):
        super(ConvNet, self).__init__()
        model_dict = [{'mult': 1, 'kernel': 3, 'stride': 1, 'padding': 1},
                      {'mult': 1, 'kernel': 4, 'stride': 2, 'padding': 1},
                      {'mult': 2, 'kernel': 3, 'stride': 1, 'padding': 1},
                      {'mult': 2, 'kernel': 3, 'stride': 1, 'padding': 1},
                      {'mult': 2, 'kernel': 4, 'stride': 2, 'padding': 1},
                      {'mult': 4, 'kernel': 3, 'stride': 1, 'padding': 1},
                      {'mult': 4, 'kernel': 4, 'stride': 2, 'padding': 1},
                      {'mult': 8, 'kernel': 3, 'stride': 1, 'padding': 1}]
        modules = []
        out_size = 0
        for i in range(num_hidden_layers):
            if i == 0:
                in_size = nc
                H = data_size
            else:
                in_size = ndf*model_dict[i-1]['mult']
            out_size = ndf*model_dict[i]['mult']
            if spectral_norm == True:
                modules.append(SNConv2d(in_size, out_size, model_dict[i]['kernel'],
                                   model_dict[i]['stride'], model_dict[i]['padding'], bias=True))
            else:
                modules.append(nn.Conv2d(in_size, out_size, model_dict[i]['kernel'],
                                        model_dict[i]['stride'], model_dict[i]['padding'], bias=True))
            modules.append(nn.LeakyReLU(0.1, inplace=True))

            H = floor((H + 2*(model_dict[i]['padding']) - (model_dict[i]['kernel'] - 1) - 1)/
                      model_dict[i]['stride'] + 1)
        self.main = nn.Sequential(*modules)

        self.snlinear = SNLinear(out_size * H * H, num_classes)

    def forward(self, input):
        output = self.main(input)
        output = output.view(output.size(0), -1)
        output = self.snlinear(output)
        return output

class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(
        self,
        growth_rate=12,
        block_config=(16, 16, 16),
        compression=0.8,
        num_init_features=24,
        bn_size=4,
        drop_rate=0,
        avgpool_size=8,
        num_classes=10,
    ):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, "compression of densenet should be between 0 and 1"
        self.avgpool_size = avgpool_size

        # First convolution
        self.features = nn.Sequential(
            OrderedDict([("conv0", nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))])
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features, num_output_features=int(num_features * compression)
                )
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module("norm_final", nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1)
        out = self.classifier(out)
        return out

class SNDenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(
        self,
        growth_rate=12,
        block_config=(16, 16, 16),
        compression=0.8,
        num_init_features=24,
        bn_size=4,
        drop_rate=0,
        avgpool_size=8,
        num_classes=10,
    ):

        super(SNDenseNet, self).__init__()
        assert 0 < compression <= 1, "compression of densenet should be between 0 and 1"
        self.avgpool_size = avgpool_size

        # First convolution
        self.features = nn.Sequential(
            OrderedDict([("conv0", SNConv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))])
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _SNTransition(
                    num_input_features=num_features, num_output_features=int(num_features * compression)
                )
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        #self.features.add_module("norm_final", nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = SNLinear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1)
        out = self.classifier(out)
        return out

class MSNDenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(
        self,
        growth_rate=12,
        block_config=(16, 16, 16),
        compression=0.8,
        num_init_features=24,
        bn_size=4,
        drop_rate=0,
        avgpool_size=8,
        num_classes=10,
    ):

        super(MSNDenseNet, self).__init__()
        assert 0 < compression <= 1, "compression of densenet should be between 0 and 1"
        self.avgpool_size = avgpool_size

        # First convolution
        self.features = nn.Sequential(
            OrderedDict([("conv0", SNConv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))])
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _MSNTransition(
                    num_input_features=num_features, num_output_features=int(num_features * compression)
                )
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module("norm_final", MeanSpectralNorm(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1)
        out = self.classifier(out)
        return out

class MNISTConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTConvNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=2),
            #nn.BatchNorm2d(32),
            nn.ReLU())
        self.fc = nn.Linear(16 * 16 * 32, num_classes)

    def forward(self, x):
        out = self.main(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class MNISTSNConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTSNConvNet, self).__init__()
        self.main = nn.Sequential(
            SNConv2d(1, 16, kernel_size=5, stride=1, padding=2),
            #nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(16, 32, kernel_size=3, stride=1, padding=2),
            #nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(32, 32, kernel_size=4, stride=2, padding=2),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True))
            #nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = SNLinear(16 * 16 * 32, num_classes)


    def forward(self, x):
        out = self.main(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class MNISTBNConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTBNConvNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True))
            #nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(16 * 16 * 32, num_classes)


    def forward(self, x):
        out = self.main(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class MNISTMSNConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTMSNConvNet, self).__init__()
        self.main = nn.Sequential(
            SNConv2d(1, 16, kernel_size=5, stride=1, padding=2),
            MeanSpectralNorm(16),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(16, 32, kernel_size=3, stride=1, padding=2),
            MeanSpectralNorm(32),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(32, 32, kernel_size=4, stride=2, padding=2),
            MeanSpectralNorm(32),
            nn.LeakyReLU(0.1, inplace=True))
            #nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = SNLinear(16 * 16 * 32, num_classes)


    def forward(self, x):
        out = self.main(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class VGG(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(VGG, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes)
        )
        # print(self.features)
        # print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, norm='BN'):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            if norm == 'BN':
                layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU(), nn.Dropout(0.3)]
            elif norm == 'SN':
                SNconv2d = SNConv2d(in_channels, out_channels, kernel_size=3, padding=padding)
                layers += [SNconv2d, nn.LeakyReLU(0.1, inplace=True)]
            elif norm == 'MSN':
                SNconv2d = SNConv2d(in_channels, out_channels, kernel_size=3, padding=padding)
                layers += [SNconv2d, MeanSpectralNorm(out_channels), nn.LeakyReLU(0.1, inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(), nn.Dropout(0.3)]
            in_channels = out_channels
    return nn.Sequential(*layers)

def vgg16(n_channel=32, pretrained=None, norm = None):
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = make_layers(cfg, norm=norm)
    model =VGG(layers, n_channel=8*n_channel, num_classes=10)
    return model