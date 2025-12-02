import torch
import torch.nn as nn
import torch.nn.init as init
from models.gradnorm import GradNorm


class Generator(nn.Module):
    def __init__(self, z_dim, M=4):
        super().__init__()
        self.M = M
        self.linear = nn.Linear(z_dim, M * M * 512)
        self.main = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh())
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                init.normal_(m.weight, std=0.02)
                init.zeros_(m.bias)

    def forward(self, z, *args, **kwargs):
        x = self.linear(z)
        x = x.view(x.size(0), -1, self.M, self.M)
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, M=32):
        super().__init__()
        self.M = M

        # M
        self.block1 = GradNorm(nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=False)
        ))
        self.block2 = GradNorm(nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=False)
        ))
        # M / 2
        self.block3 = GradNorm(nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=False)
        ))
        self.block4 = GradNorm(nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=False)
        ))
        # M / 4
        self.block5 = GradNorm(nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=False)
        ))
        self.block6 = GradNorm(nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=False)
        ))
        # M / 8
        self.block7 = GradNorm(nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=False)
        ))

        self.linear = GradNorm(nn.Linear(M // 8 * M // 8 * 512, 1))
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.normal_(m.weight, std=0.02)
                init.zeros_(m.bias)

    def rescale_weight(self, min_norm=1.0, max_norm=1.33):
        a = 1.0
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    w_norm = m.weight.norm(p=2)
                    print(m, w_norm)
                    w_norm = max(w_norm, min_norm)
                    w_norm = min(w_norm, max_norm)
                    a = a * w_norm
                    m.weight.data.div_(w_norm)
                    m.bias.data.div_(a)

    def forward(self, x, *args, **kwargs):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


class Generator32(Generator):
    def __init__(self, z_dim, *args):
        super().__init__(z_dim, M=4)


class Generator48(Generator):
    def __init__(self, z_dim, *args):
        super().__init__(z_dim, M=6)


class Discriminator32(Discriminator):
    def __init__(self, *args):
        super().__init__(M=32)


class Discriminator48(Discriminator):
    def __init__(self, *args):
        super().__init__(M=48)
