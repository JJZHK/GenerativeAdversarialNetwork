'''
@author: JJZHK
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: 03.Cifar10_32.py
@time: 2019-06-14 09:45
@desc: 
'''
import torch
import torch.nn
import torchvision as tv
import os

import ELib.utils.progressbar as eup

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DATA_PATH = "/data/input/Cat64/"
EPOCH = 100
BATCH_SIZE = 64
NOISE_DIM = 100
IMAGE_CHANNEL = 3
IMAGE_SIZE = 64
LAMBDA = 0.25
LEARNING_RATE = 5e-5


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.deconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(NOISE_DIM, 64 * 16, 4, 1, 0, bias=False),
            torch.nn.SELU(inplace=True)
        )
        self.deconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64 * 16, 64 * 8, 4, 2, 1, bias=False),
            torch.nn.SELU(inplace=True)
        )
        self.deconv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            torch.nn.SELU(inplace=True)
        )
        self.deconv4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            torch.nn.SELU(inplace=True)
        )
        self.deconv5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64 * 2, IMAGE_CHANNEL, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )

        initialize_weights(self)

    def forward(self, x):
        output = self.deconv1(x)
        output = self.deconv2(output)
        output = self.deconv3(output)
        output = self.deconv4(output)
        output = self.deconv5(output)
        return output


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(IMAGE_CHANNEL, 64 * 2, 4, 2, 1, bias=False),
            torch.nn.SELU(inplace=True)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            torch.nn.SELU(inplace=True)
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            torch.nn.SELU(inplace=True)
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64 * 8, 64 * 16, 4, 2, 1, bias=False),
            torch.nn.SELU(inplace=True)
        )

        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(64 * 16, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid()
        )
        initialize_weights(self)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = output.view(-1)

        return output


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

x            = torch.FloatTensor(BATCH_SIZE, IMAGE_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
x_both       = torch.FloatTensor(BATCH_SIZE, IMAGE_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
z            = torch.FloatTensor(BATCH_SIZE, NOISE_DIM, 1, 1)
u            = torch.FloatTensor(BATCH_SIZE, 1, 1, 1)
z_test       = torch.FloatTensor(100, NOISE_DIM, 1, 1).normal_(0, 1)
grad_outputs = torch.ones(BATCH_SIZE)
one          = torch.FloatTensor([1])
one_neg      = one * -1

x_var      = torch.autograd.Variable(x)
z_var      = torch.autograd.Variable(z)
z_test_var = torch.autograd.Variable(z_test)


if torch.cuda.is_available() > 0:
    NetG = NetG.cuda()
    NetD = NetD.cuda()
    x_var = x_var.cuda()
    z_var = z_var.cuda()
    u = u.cuda()
    z_test_var = z_test_var.cuda()
    grad_outputs = grad_outputs.cuda()
    one, one_neg = one.cuda(), one_neg.cuda()


bar = eup.ProgressBar(EPOCH, len(train_loader), "D Loss:%.3f;G Loss:%.3f")
for epoch in range(1, EPOCH + 1):
    for index, (image, label) in enumerate(train_loader):
        mini_batch = image.shape[0]
        for p in NetD.parameters():
            p.requires_grad = True

        real_var = torch.autograd.Variable(image)

        if torch.cuda.is_available():
            real_var = real_var.cuda()

        optimizerD.zero_grad()

        D_real = NetD(real_var)
        D_real_loss = D_real.mean()
        D_real_loss.backward(one_neg)

        z_var.data.normal_(0, 1)
        z_volatile = torch.autograd.Variable(z_var.data, volatile = True)
        x_fake = torch.autograd.Variable(NetG(z_volatile).data)
        # Discriminator Loss fake
        D_fake_loss = NetD(x_fake)
        D_fake_loss = D_fake_loss.mean()
        D_fake_loss.backward(one)

        u.uniform_(0, 1)
        x_both = x_var.data*u + x_fake.data*(1-u)
        if torch.cuda.is_available():
            x_both = x_both.cuda()

        x_both = torch.autograd.Variable(x_both, requires_grad=True)
        grad = torch.autograd.grad(outputs=NetD(x_both), inputs=x_both, grad_outputs=grad_outputs, retain_graph=True,
                                   create_graph=True, only_inputs=True)[0]
        grad_penalty = 10*((grad.norm(2, 1).norm(2, 1).norm(2, 1) - 1) ** 2).mean()
        grad_penalty.backward()
        # Optimize
        errD_penalty = D_fake_loss - D_real_loss + grad_penalty
        D_loss = D_fake_loss - D_real_loss
        optimizerD.step()

        G_loss = D_loss
        if ((index+1) % 5) == 0:
            for p in NetD.parameters():
                p.requires_grad = False
            # update G network
            optimizerG.zero_grad()

            z_var.data.normal_(0, 1)
            x_fake = NetG(z_var)
            # Generator Loss
            errG = NetD(x_fake)
            errG = errG.mean()
            # print(errG)
            errG.backward(one_neg)
            optimizerG.step()

        bar.show(epoch, D_loss.item(), G_loss.item())

    fake_u = NetG(z_test_var)
    '''
    如果不加normalize，图片大小比较小，但是会比较黑；如果加了normalize，图片大小比较大，但是比较亮
    '''
    tv.utils.save_image(fake_u.data, 'outputs/Cat64_%03d.png' % epoch, normalize=True, nrow=10)