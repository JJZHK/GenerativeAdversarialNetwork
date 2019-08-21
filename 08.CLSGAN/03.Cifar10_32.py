'''
@author: JJZHK
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: 01.MNIST_28.py
@time: 2019-06-20 13:36
@desc: 
'''
import torch
import torch.nn
import torchvision as tv

import ELib.pyt.nuwa.dataset as epfd
import ELib.utils.progressbar as eup
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

DATA_PATH = "/data/input/cifar10"
EPOCH = 100
GPU_NUMS = 1
BATCH_SIZE = 64
NOISE_DIM  = 100
IMAGE_CHANNEL = 3
IMAGE_SIZE = 32
LEARNING_RATE = 2e-4


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

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(NOISE_DIM + 10, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU()
        )

        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(1024, 128 * 8 * 8),
            torch.nn.BatchNorm1d(128 * 8 * 8),
            torch.nn.ReLU()
        )

        self.deconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )

        self.deconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, IMAGE_CHANNEL, 4, 2, 1),
            torch.nn.Sigmoid()
        )

        initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = x.view(-1, NOISE_DIM + 10)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 128, 8, 8)
        x = self.deconv1(x)
        x = self.deconv2(x)

        return x


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(IMAGE_CHANNEL + 10, 64, 4, 2, 1),
            torch.nn.LeakyReLU(0.2)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 4, 2, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2)
        )

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(128 * 8 * 8, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.LeakyReLU(0.2)
        )

        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(1024, 1),
            torch.nn.Sigmoid()
        )
        initialize_weights(self)

    def forward(self, input, label):
        output = torch.cat([input, label], 1)
        output = self.conv1(output)
        output = self.conv2(output)
        output = output.view(-1, 128 * 8 * 8)
        output = self.fc1(output)
        output = self.fc2(output)

        return output


NetD = Discriminator()
NetG = Generator()

optimizerD = torch.optim.Adam(NetD.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(NetG.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

dataset = epfd.Cifar10DataSetForPytorch(root=DATA_PATH, transform=tv.transforms.Compose([
    tv.transforms.ToTensor(),
]))
train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

MSE_LOSS = torch.nn.MSELoss()

fill = torch.zeros([10, 10, IMAGE_SIZE, IMAGE_SIZE])
for i in range(10):
    fill[i, i, :, :] = 1

onehot = torch.zeros(10, 10)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)

temp_z_  = torch.randn(10, 100)
fixed_z_ = temp_z_
fixed_y_ = torch.zeros(10, 1)
for i in range(9):
    fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
    temp = torch.ones(10, 1) + i
    fixed_y_ = torch.cat([fixed_y_, temp], 0)
fixed_z_ = fixed_z_.view(-1, 100, 1, 1)
fixed_y_label_ = torch.zeros(100, 10)
fixed_y_label_.scatter_(1, fixed_y_.type(torch.LongTensor), 1)
fixed_y_label_ = fixed_y_label_.view(-1, 10, 1, 1)

with torch.no_grad():
    fixed_z_var    = torch.autograd.Variable(fixed_z_.cuda() )
    fixed_y_label_var = torch.autograd.Variable(fixed_y_label_.cuda())

if torch.cuda.is_available() > 0:
    NetG = NetG.cuda()
    NetD = NetD.cuda()
    MSE_LOSS = MSE_LOSS.cuda()
    fixed_z_var = fixed_z_var.cuda()
    fixed_y_label_var = fixed_y_label_var.cuda()


bar = eup.ProgressBar(EPOCH, len(train_loader), "D Loss:%.3f;G Loss:%.3f")
for epoch in range(1, EPOCH + 1):
    NetG.train()
    for index, (image, label) in enumerate(train_loader):
        mini_batch = label.shape[0]

        label_true = torch.ones(mini_batch)
        label_false = torch.zeros(mini_batch)

        label_true_var  = torch.autograd.Variable(label_true)
        label_false_var = torch.autograd.Variable(label_false)

        label_real = label.squeeze().type(torch.LongTensor)
        label_real = fill[label_real]

        image_var = torch.autograd.Variable(image)
        label_var = torch.autograd.Variable(label_real)

        if torch.cuda.is_available():
            image_var = image_var.cuda()
            label_var = label_var.cuda()
            label_true_var = label_true_var.cuda()
            label_false_var = label_false_var.cuda()

        NetD.zero_grad()
        d_result = NetD(image_var, label_var)
        d_result = d_result.squeeze()
        D_LOSS_REAL = MSE_LOSS(d_result, label_true_var)

        img_fake    = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        label_fake = (torch.rand(mini_batch, 1) * 10).type(torch.LongTensor).squeeze()

        img_fake_var     = torch.autograd.Variable(img_fake)
        label_fake_G_var = torch.autograd.Variable(onehot[label_fake])
        label_fake_D_var = torch.autograd.Variable(fill[label_fake])

        if torch.cuda.is_available():
            img_fake_var = img_fake_var.cuda()
            label_fake_G_var = label_fake_G_var.cuda()
            label_fake_D_var = label_fake_D_var.cuda()

        g_result = NetG(img_fake_var, label_fake_G_var)
        d_result = NetD(g_result, label_fake_D_var)
        d_result = d_result.squeeze()
        D_LOSS_FAKE = MSE_LOSS(d_result, label_false_var)

        D_train_loss = D_LOSS_REAL + D_LOSS_FAKE
        D_train_loss.backward()
        optimizerD.step()

        NetG.zero_grad()
        g_result = NetG(img_fake_var, label_fake_G_var)
        d_result = NetD(g_result, label_fake_D_var)
        d_result = d_result.squeeze()
        G_train_loss= MSE_LOSS(d_result, label_true_var)
        G_train_loss.backward()
        optimizerG.step()

        bar.show(epoch, D_train_loss.item(), G_train_loss.item())

    fake_u=NetG(fixed_z_var, fixed_y_label_var)
    tv.utils.save_image(fake_u.data,'outputs/Cifar10_%03d.png' % epoch,nrow=10)
