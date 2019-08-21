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
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

DATA_PATH = '/data/input/AnimateFace/'
EPOCH = 100
BATCH_SIZE = 64
NOISE_DIM = 100
IMAGE_CHANNEL = 3
IMAGE_SIZE = 96
LEARNING_RATE = 1e-4
K = 0.
GAMMA = 0.5

NUM_STEPS = 4
MIN_SIZE = 6

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


class Coder(torch.nn.Module):
    def __init__(self, in_features = NOISE_DIM, num_steps = 2):
        super(Coder, self).__init__()

        self.in_features = in_features
        self.num_steps = num_steps

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.in_features,out_features=1024),
            torch.nn.BatchNorm1d(num_features=1024),
            torch.nn.ReLU()
        )

        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=1024, out_features=64 * MIN_SIZE * MIN_SIZE),
            torch.nn.BatchNorm1d(num_features=64 * MIN_SIZE * MIN_SIZE),
            torch.nn.ReLU()
        )

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(),
        )

        self.upsample = torch.nn.Upsample(scale_factor=2)

        self.fc3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=IMAGE_CHANNEL, kernel_size=3, stride=1, padding=1),
            torch.nn.Tanh()
        )

        initialize_weights(self)

    def forward(self, x):
        network = self.fc1(x)
        network = self.fc2(network)

        network = network.view(-1, 64, MIN_SIZE, MIN_SIZE)

        for _ in range(self.num_steps):
            network = self.conv(network)
            network = self.conv(network)
            network = self.upsample(network)

        network = self.fc3(network)

        return network


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.coder = Coder(NOISE_DIM, NUM_STEPS)

    def forward(self, x):
        return self.coder.forward(x)


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.inp = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=IMAGE_CHANNEL, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU()
        )

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(),
        )

        self.down = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.coder = Coder(in_features=MIN_SIZE * MIN_SIZE * 64, num_steps=NUM_STEPS)

        initialize_weights(self)

    def forward(self, x):
        network = self.inp(x)

        for _ in range(NUM_STEPS):
            network = self.conv(network)
            network = self.conv(network)
            network = self.down(network)

        network = network.view(-1, MIN_SIZE * MIN_SIZE * 64)

        return self.coder.forward(network)


NetD = Discriminator()
NetG = Generator()
optimizerD = torch.optim.Adam(NetD.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(NetG.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

dataset = tv.datasets.ImageFolder(root=DATA_PATH, transform=tv.transforms.Compose([
    tv.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.5] * 3, [0.5] * 3)
]))
train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

fix_noise = torch.FloatTensor(100, NOISE_DIM).normal_(0, 1)
fix_noise_var = torch.autograd.Variable(fix_noise)

if torch.cuda.is_available() > 0:
    NetG = NetG.cuda()
    NetD = NetD.cuda()
    fix_noise_var = fix_noise_var.cuda()


bar = eup.ProgressBar(EPOCH, len(train_loader), "D Loss:%.3f;G Loss:%.3f")
for epoch in range(1, EPOCH + 1):
    for p in optimizerD.param_groups + optimizerG.param_groups:
        p['lr'] = LEARNING_RATE * 0.95
    for index, (image, label) in enumerate(train_loader):
        mini_batch = image.shape[0]

        noise = torch.FloatTensor(mini_batch, NOISE_DIM).normal_(0, 1)

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
    tv.utils.save_image(fake_u.data,'outputs/AnimateFace_%03d.png' % epoch, normalize=True, nrow=10)