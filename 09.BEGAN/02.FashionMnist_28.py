'''
@author: JJZHK
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: 01.MNIST_28.py
@time: 2019-06-24 09:33
@desc: 
'''
import torch
import torch.nn
import torchvision as tv

import ELib.utils.progressbar as eup
import ELib.pyt.nuwa.dataset as epfd
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

DATA_PATH = '/data/input/fashionmnist'
EPOCH = 100
BATCH_SIZE = 64
NOISE_DIM = 62
IMAGE_CHANNEL = 1
IMAGE_SIZE = 28
LEARNING_RATE = 2e-4
K = 0.
GAMMA = 0.75

if not os.path.exists('outputs'):
    os.mkdir('outputs')


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, torch.nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=NOISE_DIM,out_features=1024),
            torch.nn.BatchNorm1d(num_features=1024),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=1024, out_features=128 * 7 * 7),
            torch.nn.BatchNorm1d(num_features=128*7*7),
            torch.nn.ReLU()
        )

        self.model2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=IMAGE_CHANNEL, kernel_size=4, stride=2, padding=1),
            torch.nn.Sigmoid()
        )

        initialize_weights(self)

    def forward(self, x):
        network = self.model1(x)

        network = network.view(-1, 128, 7, 7)
        network = self.model2(network)

        return network


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=IMAGE_CHANNEL, out_channels=64, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU()
        )
        self.model2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=64 * 14 * 14, out_features=32),
            torch.nn.BatchNorm1d(num_features=32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32, out_features= 64 * 14 * 14),
            torch.nn.BatchNorm1d(num_features=64 * 14 * 14),
            torch.nn.ReLU()
        )
        self.model3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1)
        )

        initialize_weights(self)

    def forward(self, x):
        network = self.model1(x)

        network = network.view(network.size()[0], -1)

        network = self.model2(network)

        network = network.view(-1, 64, 14,14)

        network = self.model3(network)

        return network


NetD = Discriminator()
NetG = Generator()
optimizerD = torch.optim.Adam(NetD.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(NetG.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

dataset = epfd.FashionMnistPytorchData(root=DATA_PATH, transform=tv.transforms.Compose([
    # tv.transforms.Resize(CONFIG["IMAGE_SIZE"]),
    tv.transforms.ToTensor(),
    # tv.transforms.Normalize([0.5] * 3, [0.5] * 3)
]))
train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

fix_noise = torch.randn(100, NOISE_DIM)
fix_noise_var = torch.autograd.Variable(fix_noise)

if torch.cuda.is_available() > 0:
    NetG = NetG.cuda()
    NetD = NetD.cuda()
    fix_noise_var = fix_noise_var.cuda()


bar = eup.ProgressBar(EPOCH, len(train_loader), "D Loss:%.3f;G Loss:%.3f")
for epoch in range(1, EPOCH + 1):
    for index, (image, label) in enumerate(train_loader):
        mini_batch = image.shape[0]

        noise = torch.rand(mini_batch, NOISE_DIM)

        real_var  = torch.autograd.Variable(image)
        noise_var = torch.autograd.Variable(noise)

        label_real_var = torch.autograd.Variable(torch.ones(mini_batch, 1))
        label_fake_var = torch.autograd.Variable(torch.zeros(mini_batch, 1))

        if torch.cuda.is_available():
            real_var = real_var.cuda()
            noise_var = noise_var.cuda()
            label_real_var = label_real_var.cuda()
            label_fake_var = label_fake_var.cuda()

        NetD.zero_grad()

        D_real = NetD(real_var)
        D_real_loss = torch.mean(torch.abs(D_real - real_var))

        G_ = NetG(noise_var)
        D_fake = NetD(G_)
        D_fake_loss = torch.mean(torch.abs(D_fake - G_))

        D_loss = D_real_loss - K * D_fake_loss
        D_loss.backward()
        optimizerD.step()

        NetG.zero_grad()

        G_ = NetG(noise_var)
        D_fake = NetD(G_)
        D_fake_loss = torch.mean(torch.abs(D_fake - G_))
        G_loss = D_fake_loss

        G_loss.backward()
        optimizerG.step()

        temp_M = D_real_loss + torch.abs(GAMMA * D_real_loss - D_fake_loss)

        temp_K = K + GAMMA * (GAMMA * D_real_loss - D_fake_loss)
        temp_K = temp_K.item()
        K = min(max(temp_K, 0), 1)
        M = temp_M.item()

        bar.show(epoch, D_loss.item(), G_loss.item())

    fake_u=NetG(fix_noise_var)
    tv.utils.save_image(fake_u.data,'outputs/FashionMnist_%03d.png' % epoch,nrow=10)