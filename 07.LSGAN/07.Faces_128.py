'''
@author: JJZHK
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: 04.Cat_64.py
@time: 2019-06-14 10:59
@desc: 
'''
import torch
import torch.nn
import torchvision as tv

import os

import ELib.utils.progressbar as eup

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DATA_PATH = "/data/input/face/128/"
EPOCH = 100
GPU_NUMS = 1
BATCH_SIZE = 64
NOISE_DIM  = 100
IMAGE_CHANNEL = 3
IMAGE_SIZE = 128
LEARNING_RATE = 1e-4


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.deconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(NOISE_DIM, 1024, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU()
        )

        self.deconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU()
        )

        self.deconv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )

        self.deconv4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU()
        )

        self.deconv5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )

        self.deconv6 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, IMAGE_CHANNEL, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )

        weights_init(self)

    def forward(self, x):
        output = self.deconv1(x)
        output = self.deconv2(output)
        output = self.deconv3(output)
        output = self.deconv4(output)
        output = self.deconv5(output)
        output = self.deconv6(output)
        return output


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(IMAGE_CHANNEL, 64, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
        )

        weights_init(self)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        output = output.view(-1)

        return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


NetG = Generator()
NetD = Discriminator()
MSE_LOSS = torch.nn.MSELoss()
optimizerD = torch.optim.Adam(NetD.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(NetG.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
trans = tv.transforms.Compose([
    # tv.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
])
data = tv.datasets.ImageFolder(root=DATA_PATH, transform=trans)
train_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

fix_noise = torch.FloatTensor(100, NOISE_DIM, 1, 1).normal_(0, 1)
fix_noise_var = torch.autograd.Variable(fix_noise)

if torch.cuda.is_available() > 0:
    NetG = NetG.cuda()
    NetD = NetD.cuda()
    MSE_LOSS = MSE_LOSS.cuda()
    fix_noise_var = fix_noise_var.cuda()

bar = eup.ProgressBar(EPOCH, len(train_loader), "D Loss:%.3f;G Loss:%.3f")
for epoch in range(1, EPOCH + 1):
    NetG.train()
    for index, (image, label) in enumerate(train_loader):
        mini_batch = image.shape[0]
        noise = torch.FloatTensor(mini_batch, NOISE_DIM, 1, 1).normal_(0, 1)

        real_var  = torch.autograd.Variable(image)
        noise_var = torch.autograd.Variable(noise)
        label_real_var = torch.autograd.Variable(torch.ones(mini_batch, 1 ))
        label_fake_var = torch.autograd.Variable(torch.zeros(mini_batch, 1))

        if torch.cuda.is_available():
            real_var = real_var.cuda()
            noise_var = noise_var.cuda()
            label_real_var = label_real_var.cuda()
            label_fake_var = label_fake_var.cuda()

        NetD.zero_grad()

        D_real = NetD(real_var)
        D_real_loss = MSE_LOSS(D_real, label_real_var)

        G_ = NetG(noise_var)
        D_fake = NetD(G_)
        D_fake_loss = MSE_LOSS(D_fake, label_fake_var)

        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        optimizerD.step()

        NetG.zero_grad()

        G_ = NetG(noise_var)
        D_fake = NetD(G_)
        G_loss = MSE_LOSS(D_fake, label_real_var)

        G_loss.backward()
        optimizerG.step()

        bar.show(epoch, D_loss.item(), G_loss.item())

    fake_u=NetG(fix_noise_var)
    tv.utils.save_image(fake_u.data,'outputs/Face128_%03d.png' % epoch,normalize=True,
                        nrow=10)