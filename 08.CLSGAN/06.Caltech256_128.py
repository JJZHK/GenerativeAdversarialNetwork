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
import torchvision as tv

import ELib.utils.progressbar as eup
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DATA_PATH = "/data/input/Caltech256/"
EPOCHS = 100
BATCH_SIZE = 64
IMAGE_SIZE = 128
IMAGE_CHANNEL = 3
NOISE_DIM = 100
LEARNING_RATE = 2e-4
NUM_CLASSES = 256

if not os.path.exists('outputs'):
    os.mkdir('outputs')


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            # m.bias.data.zero_()
        elif isinstance(m, torch.nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            # m.bias.data.zero_()
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(NOISE_DIM + NUM_CLASSES, NOISE_DIM + NUM_CLASSES)
        )

        self.deconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(NOISE_DIM + NUM_CLASSES, 64 * 16, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(64 * 16),
            torch.nn.ReLU(inplace=True)
        )

        self.deconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64 * 16, 64 * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64 * 8),
            torch.nn.ReLU(inplace=True)
        )

        self.deconv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64 * 4),
            torch.nn.ReLU(inplace=True)
        )

        self.deconv4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64 * 2),
            torch.nn.ReLU(inplace=True)
        )

        self.deconv5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )

        self.deconv6 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, IMAGE_CHANNEL, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )

        initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.fc1(x)
        x = x.view(-1, NOISE_DIM + NUM_CLASSES, 1, 1)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(IMAGE_CHANNEL, 64, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64 * 2),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64 * 4),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64 * 8),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(64 * 8, 64 * 16, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64 * 16),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(64 * 16, 64 * 16, 4, 1, 0, bias=False)
        )

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(1024 + NUM_CLASSES, 256),
            torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.merge_layer = torch.nn.Sequential(
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )

        initialize_weights(self)

    def forward(self, input, label):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(-1, 64 * 16)
        x = torch.cat([x, label], 1)
        x = self.fc1(x)

        x = self.merge_layer(x)

        return x

def one_hot(target):
    y = torch.zeros(target.size()[0], NUM_CLASSES)

    for i in range(target.size()[0]):
        y[i, target[i]] = 1

    return y


trans = tv.transforms.Compose([
    tv.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
dataset = tv.datasets.ImageFolder(root=DATA_PATH, transform=trans)
dataLoader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True)

NetG = Generator()
NetD = Discriminator()
NetG = torch.nn.DataParallel(NetG)
NetD = torch.nn.DataParallel(NetD)
optimizerD = torch.optim.Adam(NetD.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(NetG.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = torch.nn.MSELoss()

predict_noise = torch.randn(NUM_CLASSES, NOISE_DIM)
Predict_y = torch.FloatTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
Predict_y = torch.reshape(Predict_y, shape=(16, 1))

for i in range(2, 17):
    temp = torch.FloatTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) + (i-1) * 16
    temp = torch.reshape(temp, shape=(16, 1))

    Predict_y = torch.cat([Predict_y, temp], 0)

Predict_y = one_hot(Predict_y.long())

if torch.cuda.is_available():
    NetD = NetD.cuda()
    NetG = NetG.cuda()
    predict_noise = predict_noise.cuda()
    criterion.cuda()
    Predict_y = Predict_y.cuda()

Predict_Noise_var = torch.autograd.Variable(predict_noise)
Predict_y_var = torch.autograd.Variable(Predict_y)

bar = eup.ProgressBar(EPOCHS, len(dataLoader), "D Loss:%.3f, G Loss:%.3f")
for epoch in range(1, EPOCHS + 1):
    if epoch % 10 == 0:
        optimizerG.param_groups[0]['lr'] /= 10
        optimizerD.param_groups[0]['lr'] /= 10

    for img_real, label_real in dataLoader:
        mini_batch = label_real.shape[0]

        label_true = torch.ones(mini_batch)
        label_false = torch.zeros(mini_batch)
        label = one_hot(label_real.long().squeeze())
        noise = torch.randn(mini_batch, NOISE_DIM)

        if torch.cuda.is_available():
            label_true = label_true.cuda()
            label_false = label_false.cuda()
            img_real = img_real.cuda()
            label = label.cuda()
            noise = noise.cuda()

        label_true_var  = torch.autograd.Variable(label_true)
        label_false_var = torch.autograd.Variable(label_false)
        image_var       = torch.autograd.Variable(img_real)
        label_var       = torch.autograd.Variable(label)
        Noise_var = torch.autograd.Variable(noise)
        NetD.zero_grad()
        D_real = NetD(image_var, label_var)
        D_real_loss = criterion(D_real, label_true_var)

        image_fake = NetG(Noise_var, label_var)
        D_fake = NetD(image_fake, label_var)
        D_fake_loss = criterion(D_fake, label_false_var)

        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        optimizerD.step()

        NetG.zero_grad()
        image_fake = NetG(Noise_var,label_var)
        D_fake = NetD(image_fake,label_var)

        G_loss = criterion(D_fake, label_true_var)

        G_loss.backward()
        optimizerG.step()

        bar.show(epoch, D_loss.item(), G_loss.item())

    test_images = NetG(Predict_Noise_var, Predict_y_var)

    tv.utils.save_image(test_images.data[:NUM_CLASSES],
                        'outputs/Faces128_%03d.png' % (epoch), normalize=True, nrow=16)


