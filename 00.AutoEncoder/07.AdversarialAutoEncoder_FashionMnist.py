'''
@author: JJZHK
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: 07.AdversarialAutoEncoder_FashionMnist.py
@time: 2019-04-29 10:42
@desc: 
'''
import torch
import torchvision as tv
import ELib.utils.progressbar as eup
import ELib.pyt.nuwa.dataset as epfd

class Q_net(torch.nn.Module):
    def __init__(self,X_dim,N,z_dim):
        super(Q_net, self).__init__()
        self.lin1       = torch.nn.Linear(X_dim, N)
        self.lin2       = torch.nn.Linear(N, N)
        self.lin3_gauss = torch.nn.Linear(N, z_dim)
    def forward(self, x):
        x = self.lin1(x)
        x = torch.nn.Dropout(p=0.25, inplace=True)(x)
        x = torch.nn.ReLU(inplace=True)(x)

        x = self.lin2(x)
        x = torch.nn.Dropout(p=0.25, inplace=True)(x)
        x = torch.nn.ReLU(inplace=True)(x)

        z_gauss = self.lin3_gauss(x)
        return z_gauss

class P_net(torch.nn.Module):
    def __init__(self,X_dim,N,z_dim):
        super(P_net, self).__init__()
        self.lin1 = torch.nn.Linear(z_dim, N)
        self.lin2 = torch.nn.Linear(N, N)
        self.lin3 = torch.nn.Linear(N, X_dim)
    def forward(self, x):
        x = self.lin1(x)
        x = torch.nn.Dropout(p=0.25, inplace=True)(x)
        x = torch.nn.ReLU(inplace=True)(x)

        x = self.lin2(x)
        x = torch.nn.Dropout(p=0.25, inplace=True)(x)

        x = self.lin3(x)
        return torch.nn.functional.sigmoid(x)

class D_net_gauss(torch.nn.Module):
    def __init__(self,N,z_dim):
        super(D_net_gauss, self).__init__()
        self.lin1 = torch.nn.Linear(z_dim, N)
        self.lin2 = torch.nn.Linear(N, N)
        self.lin3 = torch.nn.Linear(N, 1)
    def forward(self, x):
        x = self.lin1(x)
        x = torch.nn.Dropout(p=0.25, inplace=True)(x)
        x = torch.nn.ReLU(inplace=True)(x)

        x = self.lin2(x)
        x = torch.nn.Dropout(p=0.25, inplace=True)(x)
        x = torch.nn.ReLU(inplace=True)(x)
        return torch.nn.functional.sigmoid(self.lin3(x))

img_transform = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 128
num_epochs = 100
dataset = epfd.FashionMnistPytorchData(train=True, transform=img_transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
EPS = 1e-15
gen_lr = 0.0001
reg_lr = 0.00005
z_red_dims = 120
Q = Q_net(784,1000,z_red_dims)
P = P_net(784,1000,z_red_dims)
D_gauss = D_net_gauss(500,z_red_dims)

if torch.cuda.is_available():
    Q = Q.cuda()
    P = P.cuda()
    D_gauss = D_gauss.cuda()

optim_P = torch.optim.Adam(P.parameters(), lr=gen_lr)
optim_Q_enc = torch.optim.Adam(Q.parameters(), lr=gen_lr)
optim_Q_gen = torch.optim.Adam(Q.parameters(), lr=reg_lr)
optim_D = torch.optim.Adam(D_gauss.parameters(), lr=reg_lr)

proBar = eup.ProgressBar(num_epochs, len(dataloader), "DLoss:%.3f;GLoss:%.3f")

for epoch in range(1, num_epochs+1):
    for data in dataloader:
        images, labels = data
        images = images.view(images.size(0), -1)
        images = torch.autograd.Variable(images).cuda() if torch.cuda.is_available() else torch.autograd.Variable(images)
        labels = torch.autograd.Variable(labels).cuda() if torch.cuda.is_available() else torch.autograd.Variable(labels)

        P.zero_grad()
        Q.zero_grad()
        D_gauss.zero_grad()

        z_sample = Q(images)
        X_sample = P(z_sample)
        recon_loss = torch.nn.functional.binary_cross_entropy(X_sample + EPS, images + EPS)

        recon_loss.backward()
        optim_P.step()
        optim_Q_enc.step()

        z_real_gauss = torch.autograd.Variable(torch.randn(images.size()[0], z_red_dims) * 5.).cuda() if torch.cuda.is_available() else torch.autograd.Variable(torch.randn(images.size()[0], z_red_dims) * 5.)
        # 判别器判别一下真的样本, 得到loss
        D_real_gauss = D_gauss(z_real_gauss)

        # 用encoder 生成假样本
        Q.eval()  # 切到测试形态, 这时候, Q(即encoder)不参与优化
        z_fake_gauss = Q(images)
        # 用判别器判别假样本, 得到loss
        D_fake_gauss = D_gauss(z_fake_gauss)

        # 判别器总误差
        D_loss = -torch.mean(torch.log(D_real_gauss + EPS) + torch.log(1 - D_fake_gauss + EPS))

        # 优化判别器
        D_loss.backward()
        optim_D.step()

        # encoder充当生成器
        Q.train()  # 切换训练形态, Q(即encoder)参与优化
        z_fake_gauss = Q(images)
        D_fake_gauss = D_gauss(z_fake_gauss)

        G_loss = -torch.mean(torch.log(D_fake_gauss + EPS))

        G_loss.backward()
        # 仅优化Q
        optim_Q_gen.step()
        proBar.show(epoch, D_loss.item(), G_loss.item())

    pic = X_sample.view(X_sample.size(0), 1, 28, 28)
    tv.utils.save_image(pic, './outputs/AAE_{}.png'.format(epoch))