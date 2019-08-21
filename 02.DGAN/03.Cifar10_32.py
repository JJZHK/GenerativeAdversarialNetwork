'''
@author: JJZHK
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: 03.Cifar10_32.py
@time: 2019-05-22 10:38
@desc: 
'''
import torch
import torchvision as tv
import ELib.utils.progressbar as eup
import ELib.pyt.nuwa.dataset as epfd

DATA_PATH = '/data/input/cifar10'
IMAGE_CHANNEL = 3
IMAGE_SIZE = 32
LEARNING_RATE = 2e-4
BATCH_SIZE = 64
NOISE_DIM = 100
EPOCH = 100

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.module1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(NOISE_DIM, 64 * 4, 4, 2, 0, bias=False),
            torch.nn.BatchNorm2d(64 * 4),
            torch.nn.ReLU(True)
        )

        self.module2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64 * 2),
            torch.nn.ReLU(True)
        )

        self.module3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True)
        )

        self.module4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, IMAGE_CHANNEL, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)

        return x


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.module1 = torch.nn.Sequential(
            torch.nn.Conv2d(IMAGE_CHANNEL, 64, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.module2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64 * 2),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.module3 = torch.nn.Sequential(
            torch.nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64 * 4),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=False)
        )

        self.module4 = torch.nn.Sequential(
            torch.nn.Conv2d(64 * 4, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)

        return x


trans = tv.transforms.Compose([tv.transforms.ToTensor()])

NetG = Generator()
NetD = Discriminator()
criterion = torch.nn.BCELoss()
fix_noise = torch.autograd.Variable(torch.FloatTensor(BATCH_SIZE, NOISE_DIM, 1, 1).normal_(0, 1))
if torch.cuda.is_available():
    NetG = NetG.cuda()
    NetD = NetD.cuda()
    fix_noise = fix_noise.cuda()
    criterion.cuda()

dataset = epfd.Cifar10DataSetForPytorch(train=True, transform=trans)
dataloader=torch.utils.data.DataLoader(dataset,BATCH_SIZE,shuffle = True)

optimizerD = torch.optim.Adam(NetD.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(NetG.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
bar = eup.ProgressBar(EPOCH, len(dataloader), "D Loss:%.3f;G Loss:%.3f")
for epoch in range(1, EPOCH + 1):
    if epoch % 30 == 0:
        optimizerD.param_groups[0]['lr'] /= 10
        optimizerG.param_groups[0]['lr'] /= 10

    for ii, data in enumerate(dataloader,0):
        input,_=data

        input = torch.autograd.Variable(input)
        label = torch.ones(input.size(0))
        label = torch.autograd.Variable(label)
        noise = torch.randn(input.size(0),NOISE_DIM,1,1)
        noise = torch.autograd.Variable(noise)

        if torch.cuda.is_available():
            input = input.cuda()
            label = label.cuda()
            noise = noise.cuda()

        NetD.zero_grad()
        output=NetD(input)
        error_real=criterion(output.squeeze(),label)
        error_real.backward()

        D_x=output.data.mean()
        fake_pic=NetG(noise).detach()
        output2=NetD(fake_pic)
        label.data.fill_(0) # 0 for fake
        error_fake=criterion(output2.squeeze(),label)

        error_fake.backward()
        D_x2=output2.data.mean()
        error_D=error_real+error_fake
        optimizerD.step()

        NetG.zero_grad()
        label.data.fill_(1)
        noise.data.normal_(0,1)
        fake_pic=NetG(noise)
        output=NetD(fake_pic)
        error_G=criterion(output.squeeze(),label)
        error_G.backward()

        optimizerG.step()
        D_G_z2=output.data.mean()
        bar.show(epoch, error_D.item(), error_G.item())

    fake_u=NetG(fix_noise)

    tv.utils.save_image(fake_u.data, "outputs/Cifar10_%03d.png" % epoch)