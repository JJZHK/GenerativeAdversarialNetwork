'''
@author: JJZHK
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: 01.MNIST_28.py
@time: 2019-05-22 10:38
@desc: 
'''
import torch
import torchvision as tv
import os
import ELib.pyt.nuwa.dataset as epfd
import ELib.utils.progressbar as eup

DATA_PATH = "/data/input/mnist.npz"
EPOCHS = 100
BATCH_SIZE = 128
IMAGE_SIZE = 28
IMAGE_CHANNEL = 1
NOISE_DIM = 100
LEARNING_RATE = 2e-4

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
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(NOISE_DIM, 1024),
            torch.nn.BatchNorm1d(num_features=1024),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=1024, out_features=128 * 7 * 7),
            torch.nn.BatchNorm1d(num_features=128*7*7),
            torch.nn.ReLU()
        )

        self.deconv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=IMAGE_CHANNEL, kernel_size=4, stride=2, padding=1),
            torch.nn.Tanh()
        )

        initialize_weights(self)

    def forward(self, x):
        output = x.view(-1,NOISE_DIM)
        output = self.fc(output)
        output = output.view(-1, 128, 7, 7)
        output =self.deconv(output)

        return output


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=IMAGE_CHANNEL, out_channels=64, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(negative_slope=0.2)
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128 * 7 * 7, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(1024, 1),
            torch.nn.Sigmoid()
        )

        initialize_weights(self)

    def forward(self, x):
        output = self.conv(x)
        output = output.view(-1, 128 * 7 * 7)
        output = self.fc(output)

        return output


NetG = Generator()
NetD = Discriminator()
optimizerD = torch.optim.Adam(NetD.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(NetG.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = torch.nn.BCELoss()

fix_noise = torch.FloatTensor(BATCH_SIZE, NOISE_DIM, 1, 1).normal_(0,1)
if torch.cuda.is_available():
    NetD = NetD.cuda()
    NetG = NetG.cuda()
    fix_noise = fix_noise.cuda()
    criterion.cuda()

fix_noise_var = torch.autograd.Variable(fix_noise)
transform = tv.transforms.Compose([tv.transforms.ToTensor()])

dataset = epfd.MNISTDataSetForPytorch(root=DATA_PATH, train=True, transform=transform)
dataLoader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True)

bar = eup.ProgressBar(EPOCHS, len(dataLoader), "D Loss:%.3f;G Loss:%.3f")
for epoch in range(1, EPOCHS + 1):
    if epoch % 30 == 0:
        optimizerD.param_groups[0]['lr'] /= 10
        optimizerG.param_groups[0]['lr'] /= 10

    for ii, data in enumerate(dataLoader,0):
        input,_=data
        label_real = torch.ones(input.size(0))
        label_fake = torch.zeros(input.size(0))
        noise = torch.randn(input.size(0),NOISE_DIM,1,1).normal_(0, 1)

        if torch.cuda.is_available():
            input = input.cuda()
            label_real = label_real.cuda()
            label_fake = label_fake.cuda()
            noise = noise.cuda()

        input_var = torch.autograd.Variable(input)
        label_real_var = torch.autograd.Variable(label_real)
        label_fake_var = torch.autograd.Variable(label_fake)
        noise_var = torch.autograd.Variable(noise)

        NetD.zero_grad()
        output=NetD(input_var)
        error_real=criterion(output.squeeze(),label_real_var)
        error_real.backward()

        fake_pic=NetG(noise_var).detach()
        output2=NetD(fake_pic)
        error_fake=criterion(output2.squeeze(),label_fake_var)

        error_fake.backward()
        error_D=error_real+error_fake
        optimizerD.step()

        NetG.zero_grad()
        fake_pic=NetG(noise_var)
        output=NetD(fake_pic)
        error_G=criterion(output.squeeze(),label_real_var)
        error_G.backward()

        optimizerG.step()
        bar.show(epoch, error_D.item(), error_G.item())

    fake_u=NetG(fix_noise_var)

    tv.utils.save_image(fake_u.data[:100], "outputs/MNIST_%03d.png" % epoch,nrow=10)


