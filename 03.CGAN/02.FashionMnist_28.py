'''
@author: JJZHK
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: 01.MNIST_28.py
@time: 2019-05-27 11:13
@desc: 
'''
import torch
import torch.nn
import os
import torchvision as tv

import ELib.pyt.nuwa.dataset as epfd
import ELib.utils.progressbar as eup

DATA_PATH = "/data/input/fashionmnist"
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
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(NOISE_DIM + 10, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(inplace=True)
        )

        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(1024, 128 * 7 * 7),
            torch.nn.BatchNorm1d(128 * 7 * 7),
            torch.nn.ReLU()
        )

        self.deconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )

        self.deconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 1, 4, 2, 1),
            torch.nn.Tanh()
        )

        initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = x.view(-1, NOISE_DIM + 10)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 128, 7, 7)
        x = self.deconv1(x)
        x = self.deconv2(x)

        return x


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1 + 10, 64, 4, 2, 1),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 4, 2, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(128 * 7 * 7, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(1024, 1),
            torch.nn.Sigmoid()
        )

        initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 128 * 7 * 7)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


NetG = Generator()
NetD = Discriminator()
optimizerD = torch.optim.Adam(NetD.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(NetG.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = torch.nn.BCELoss()

trans = tv.transforms.Compose([tv.transforms.ToTensor()])

dataset = epfd.FashionMnistPytorchData(root=DATA_PATH, train=True, transform=trans)
dataLoader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True)

fill = torch.zeros([10, 10, IMAGE_SIZE, IMAGE_SIZE])
for i in range(10):
    fill[i, i, :, :] = 1

onehot = torch.zeros(10, 10)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)

temp_z_ = torch.randn(10, 100)
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

if torch.cuda.is_available():
    NetD = NetD.cuda()
    NetG = NetG.cuda()
    fixed_z_ = fixed_z_.cuda()
    fixed_y_label_ = fixed_y_label_.cuda()
    criterion.cuda()

with torch.no_grad():
    fixed_z_ = torch.autograd.Variable(fixed_z_)
    fixed_y_label_ = torch.autograd.Variable(fixed_y_label_)

bar = eup.ProgressBar(EPOCHS, len(dataLoader), "D Loss:%.3f, G Loss:%.3f")
for epoch in range(1, EPOCHS + 1):
    if epoch % 30 == 0:
        optimizerG.param_groups[0]['lr'] /= 10
        optimizerG.param_groups[0]['lr'] /= 10

    for img_real, label_real in dataLoader:
        mini_batch = label_real.shape[0]


        label_true_var  = torch.autograd.Variable(torch.ones(mini_batch).cuda() if torch.cuda.is_available() else torch.ones(mini_batch))
        label_false_var = torch.autograd.Variable(torch.zeros(mini_batch).cuda() if torch.cuda.is_available() else torch.zeros(mini_batch))

        NetD.zero_grad()
        label_real = label_real.squeeze().type(torch.LongTensor)
        label_real = fill[label_real]

        image_var = torch.autograd.Variable(img_real.cuda() if torch.cuda.is_available() else img_real)
        label_var = torch.autograd.Variable(label_real.cuda() if torch.cuda.is_available() else label_real)

        d_result = NetD(image_var, label_var)
        d_result = d_result.squeeze()
        D_LOSS_REAL = criterion(d_result, label_true_var)

        img_fake = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        label_fake = (torch.rand(mini_batch, 1) * 10).type(torch.LongTensor).squeeze()
        img_fake_var     = torch.autograd.Variable(img_fake.cuda() if torch.cuda.is_available() else img_fake)
        label_fake_G_var = torch.autograd.Variable(onehot[label_fake].cuda() if torch.cuda.is_available() else onehot[label_fake])
        label_fake_D_var = torch.autograd.Variable(fill[label_fake].cuda() if torch.cuda.is_available() else fill[label_fake])

        g_result = NetG(img_fake_var, label_fake_G_var)
        d_result = NetD(g_result, label_fake_D_var)
        d_result = d_result.squeeze()
        D_LOSS_FAKE = criterion(d_result, label_false_var)

        D_train_loss = D_LOSS_REAL + D_LOSS_FAKE
        D_train_loss.backward()
        optimizerD.step()

        NetG.zero_grad()
        img_fake = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        label_fake = (torch.rand(mini_batch, 1) * 10).type(torch.LongTensor).squeeze()
        img_fake_var     = torch.autograd.Variable(img_fake.cuda() if torch.cuda.is_available() else img_fake)
        label_fake_G_var = torch.autograd.Variable(onehot[label_fake].cuda() if torch.cuda.is_available() else onehot[label_fake])
        label_fake_D_var = torch.autograd.Variable(fill[label_fake].cuda() if torch.cuda.is_available() else fill[label_fake])
        g_result = NetG(img_fake_var, label_fake_G_var)
        d_result = NetD(g_result, label_fake_D_var)
        d_result = d_result.squeeze()
        G_train_loss= criterion(d_result, label_true_var)
        G_train_loss.backward()
        optimizerG.step()

        bar.show(epoch, D_train_loss.item(), G_train_loss.item())

    test_images = NetG(fixed_z_, fixed_y_label_)

    tv.utils.save_image(test_images.data[:100],'outputs/fashionmnist_%03d.png' % (epoch),nrow=10)


