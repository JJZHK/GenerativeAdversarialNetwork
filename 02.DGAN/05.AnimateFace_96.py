'''
@author: JJZHK
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: 05.AnimateFace_96.py
@time: 2019-05-22 10:39
@desc: 
'''
import torch
import torch.nn
import ELib.utils.progressbar as eup
import torchvision as tv

DATA_PATH = '/data/input/AnimateFace/'
LEARNING_RATE_D = 2e-4
LEARNING_RATE_G = 2e-4
IMAGE_CHANNEL = 3
IMAGE_SIZE = 96
BATCH_SIZE = 64
EPOCH = 100
NOISE_DIM = 100


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # Estimated variance, must be around 1
        m.weight.data.normal_(1.0, 0.02)
        # Estimated mean, must be around 0
        m.bias.data.fill_(0)


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.deconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(NOISE_DIM, 64 * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(64 * 8),
            torch.nn.ReLU(inplace=True)
        )

        self.deconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64 * 4),
            torch.nn.ReLU(inplace=True)
        )

        self.deconv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64 * 2),
            torch.nn.ReLU(inplace=True)
        )

        self.deconv4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )

        self.deconv5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, IMAGE_CHANNEL, 5, 3, 1, bias=False),
            torch.nn.Tanh()
        )

        weights_init(self)

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
            torch.nn.Conv2d(IMAGE_CHANNEL, 64, 5, 3, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64 * 2),
            torch.nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64 * 4),
            torch.nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64 * 8),
            torch.nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid()
        )
        weights_init(self)

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
criterion = torch.nn.BCELoss()

true_labels = torch.autograd.Variable(torch.ones(BATCH_SIZE))
fake_labels = torch.autograd.Variable(torch.zeros(BATCH_SIZE))
fix_noises  = torch.autograd.Variable(torch.randn(100,NOISE_DIM,1,1))
noises      = torch.autograd.Variable(torch.randn(BATCH_SIZE,NOISE_DIM,1,1))


if torch.cuda.is_available() > 0:
    NetG = NetG.cuda()
    NetD = NetD.cuda()
    criterion = criterion.cuda()
    true_labels,fake_labels = true_labels.cuda(), fake_labels.cuda()
    fix_noises,noises = fix_noises.cuda(),noises.cuda()

optimizerD = torch.optim.Adam(NetD.parameters(),lr=LEARNING_RATE_D,betas=(0.5,0.999), weight_decay=0)
optimizerG = torch.optim.Adam(NetG.parameters(),lr=LEARNING_RATE_G,betas=(0.5,0.999), weight_decay=0)

transform=tv.transforms.Compose([
    tv.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)) ,
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.5]*3,[0.5]*3)
])

dataset = tv.datasets.ImageFolder(root=DATA_PATH, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)

bar = eup.ProgressBar(EPOCH, len(dataloader), "D Loss:%.3f;G Loss:%.3f")
for epoch in range(1, EPOCH + 1):
    if epoch % 30 == 0:
        optimizerD.param_groups[0]['lr'] /= 10
        optimizerG.param_groups[0]['lr'] /= 10

    for i, data_batch in enumerate(dataloader, 0):
        images, labels = data_batch
        real_img = torch.autograd.Variable(images.cuda() if torch.cuda.is_available() else images)

        NetD.zero_grad()
        ## 尽可能的把真图片判别为正确
        output = NetD(real_img)
        error_d_real = criterion(output,true_labels)
        error_d_real.backward()

        ## 尽可能把假图片判别为错误
        noises.data.copy_(torch.randn(BATCH_SIZE,NOISE_DIM,1,1))
        fake_img = NetG(noises).detach() # 根据噪声生成假图
        output = NetD(fake_img)
        error_d_fake = criterion(output,fake_labels)
        error_d_fake.backward()
        optimizerD.step()
        error_d = error_d_fake + error_d_real

        NetG.zero_grad()
        noises.data.copy_(torch.randn(BATCH_SIZE,NOISE_DIM,1,1))
        fake_img = NetG(noises)
        output = NetD(fake_img)
        error_g = criterion(output,true_labels)
        error_g.backward()
        optimizerG.step()

        bar.show(epoch, error_d.item(), error_g.item())

    fix_fake_imgs = NetG(fix_noises)
    tv.utils.save_image(fix_fake_imgs.data,'outputs/AnimateFace_%03d.png' % epoch,nrow=10, normalize=True)