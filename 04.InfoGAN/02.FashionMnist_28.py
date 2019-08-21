'''
@author: JJZHK
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: 01.MNIST_28.py
@time: 2019-05-30 16:55
@desc: 
'''
import torch
import torch.nn
import torchvision as tv
import os
import numpy as np

import ELib.utils.progressbar as eup
import ELib.pyt.nuwa.dataset as epfd

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DATA_PATH = '/data/input/fashionmnist/'
NOISE_DIM = 100
Z_DIM = 128
CC_DIM = 1
DC_DIM = 10
IMAGE_SIZE = 28
IMAGE_CHANNEL = 1
BATCH_SIZE = 128
EPOCHS = 100
CONTINUOUS_WEIGHT = 0.5

if not os.path.exists('outputs'):
    os.mkdir('outputs')

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(Z_DIM + CC_DIM + DC_DIM, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(inplace=True)
        )

        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(1024, 128 * 7 * 7),
            torch.nn.BatchNorm1d(128 * 7 * 7),
            torch.nn.ReLU(inplace=True)
        )

        self.conv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, IMAGE_CHANNEL, 4, 2, 1),
            torch.nn.Tanh()
        )

    def forward(self, input):
        x = self.fc1(input)
        x = self.fc2(x)
        x = x.view(-1, 128, 7, 7)
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(IMAGE_CHANNEL, 64, 4, 2, 1),
            torch.nn.LeakyReLU(0.1, True)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 4, 2, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.1, True)
        )

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(128 * 7 * 7, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU()
        )

        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(128, 1 + CC_DIM + DC_DIM)
        )

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = output.view(-1, 128 * 7 * 7)
        output = self.fc1(output)
        output = self.fc2(output)

        output[:, 0] = torch.nn.functional.sigmoid(output[:, 0].clone())

        output[:, CC_DIM + 1 : CC_DIM + 1 + DC_DIM] = torch.nn.functional.softmax(output[:, CC_DIM + 1 : CC_DIM + 1 + DC_DIM].clone())

        return output


NetG = Generator()
NetD = Discriminator()
optimizerD = torch.optim.Adam(NetD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(NetG.parameters(), lr=0.001, betas=(0.5, 0.999))
trans = tv.transforms.Compose([tv.transforms.ToTensor(),tv.transforms.Normalize([0.5], [0.5])])
dataset = epfd.FashionMnistPytorchData(root=DATA_PATH, train=True, transform=trans)
dataLoader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True)

fixed_noise = torch.Tensor(np.zeros((NOISE_DIM, Z_DIM)))
tmp = np.zeros((NOISE_DIM, CC_DIM))
for k in range(10):
    tmp[k * 10:(k + 1) * 10, 0] = np.linspace(-2, 2, 10)
fixed_cc = torch.Tensor(tmp)
tmp = np.zeros((NOISE_DIM, DC_DIM))
for k in range(10):
    tmp[k * 10 : (k + 1) * 10, k] = 1
fixed_dc = torch.Tensor(tmp)

if torch.cuda.is_available():
    NetG = NetG.cuda()
    NetD = NetD.cuda()
    fixed_noise = fixed_noise.cuda()
    fixed_cc = fixed_cc.cuda()
    fixed_dc = fixed_dc.cuda()

fixed_noise_var = torch.autograd.Variable(fixed_noise)
fixed_cc_var    = torch.autograd.Variable(fixed_cc)
fixed_dc_var    = torch.autograd.Variable(fixed_dc)

bar = eup.ProgressBar(EPOCHS, len(dataLoader), "D Loss:%.3f;G Loss:%.3f")
for epoch in range(1, EPOCHS + 1):
    for i, (images, labels) in enumerate(dataLoader):
        mini_batch = images.size(0)

        cc = torch.Tensor(np.random.randn(mini_batch, CC_DIM) * 0.5 + 0.0)

        codes=[]
        code = np.zeros((mini_batch, DC_DIM))
        random_cate = np.random.randint(0, DC_DIM, mini_batch)
        code[range(mini_batch), random_cate] = 1
        codes.append(code)
        codes = np.concatenate(codes,1)
        dc = torch.Tensor(codes)

        noise = torch.randn(mini_batch, Z_DIM)

        if torch.cuda.is_available():
            images = images.cuda()
            cc = cc.cuda()
            dc = dc.cuda()
            noise = noise.cuda()

        images = torch.autograd.Variable(images)
        cc_var = torch.autograd.Variable(cc)
        dc_var = torch.autograd.Variable(dc)
        noise_var = torch.autograd.Variable(noise)

        fake_images = NetG(torch.cat((noise_var, cc_var, dc_var),1))
        d_output_real = NetD(images)
        d_output_fake = NetD(fake_images)

        d_loss_a = -torch.mean(torch.log(d_output_real[:,0]) + torch.log(1 - d_output_fake[:,0]))

        # Mutual Information Loss
        output_cc = d_output_fake[:, 1:1+CC_DIM]
        output_dc = d_output_fake[:, 1+CC_DIM:]
        d_loss_cc = torch.mean((((output_cc - 0.0) / 0.5) ** 2))
        d_loss_dc = -(torch.mean(torch.sum(dc_var * output_dc, 1)) + torch.mean(torch.sum(dc_var * dc_var, 1)))

        d_loss = d_loss_a + CONTINUOUS_WEIGHT * d_loss_cc + 1.0 * d_loss_dc

        # Optimization
        NetD.zero_grad()
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        # ===================== Train G =====================#
        # Fake -> Real
        g_loss_a = -torch.mean(torch.log(d_output_fake[:,0]))

        g_loss = g_loss_a + CONTINUOUS_WEIGHT * d_loss_cc + 1.0 * d_loss_dc

        # Optimization
        NetG.zero_grad()
        g_loss.backward()
        optimizerG.step()

        bar.show(epoch, d_loss.item(), g_loss.item())

    fake_images = NetG(torch.cat((fixed_noise_var, fixed_cc_var, fixed_dc_var), 1))
    tv.utils.save_image(fake_images.data, "outputs/FashionMnist_%03d.png" % epoch, nrow=10)