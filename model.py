import torch.nn as nn
import torch

# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


# 4
class VGG(nn.Module):  # 大的VGG网络 只有相同的后半部分
    # 分类数量和权重有默认值
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features  # 不同的特征提取方案
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),  # inplace=True 改变输入数据
            nn.Dropout(p=0.5),  # 按概率p随机丢弃元素 一般0.5最好
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)  # 先进行不同的特征提取
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)  # 把每一张图片展成长条 0维是batch_size 1维是通道 2维是矩阵 3维是像素(无变化)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):  # 对权值进行初始化
        for m in self.modules():  # 返回模型的每一层
            if isinstance(m, nn.Conv2d):  # 如果第m层是卷积层 isinstance()用来判断一个对象是否是一个已知的类型
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# 3
def make_features(cfg: list):  # 不同的网络架构
    layers = []
    in_channels = 3
    for v in cfg:  # 根据所选择的vgg模型来完成前半部分的特征提取网络
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)  # 用*将list拆成一个个元素


# 2
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# 1
def vgg(model_name="vgg16", **kwargs):  # 特定的vgg网络
    # 输入的模型名字不在选项中就会有报错自己输入的内容
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    # 正常执行程序
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model
