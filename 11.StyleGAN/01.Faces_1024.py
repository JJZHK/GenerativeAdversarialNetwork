'''
@author: JJZHK
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: 01.Faces.py
@time: 2019-07-01 10:09
@desc: 
'''
import torch
import torch.nn
import os
import random
import numpy as np
import torchvision as tv

import ELib.pyt.nuwa.style_net as epns
import ELib.pyt.nuwa.dataset as epnd
import ELib.pyt.nuwa.transforms as epnt
import ELib.utils.progressbar as eup

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

MANUALSEED = 999
random.seed(MANUALSEED)
torch.manual_seed(MANUALSEED)

DATA_PATH = '/data/input/face/celebA'
EPOCH = 500
BATCH_SIZE = 8
IMAGE_SIZE = 1024
IMAGE_CHANNEL = 3
R1_GAMMA = 10.0
R2_GAMMA = 0.0


def R1Penalty(real_img, f):
    # gradient penalty
    apply_scaling = torch.Tensor([np.float32(np.log(2.0))])
    undo_scaling = torch.Tensor([np.float32(np.log(2.0))])

    if torch.cuda.is_available():
        apply_scaling = apply_scaling.cuda()
        undo_scaling = undo_scaling.cuda()
        real_img = real_img.cuda()

    reals = torch.autograd.Variable(real_img, requires_grad=True)
    apply_loss_scaling = lambda x: x * torch.exp(x * apply_scaling)
    undo_loss_scaling = lambda x: x * torch.exp(-x * undo_scaling)

    real_logit = f(reals)
    real_logit = apply_loss_scaling(torch.sum(real_logit))
    ones = torch.ones(real_logit.size())

    if torch.cuda.is_available():
        ones = ones.cuda()

    real_grads = torch.autograd.grad(real_logit, reals, grad_outputs=ones, create_graph=True)[0].view(reals.size(0), -1)
    real_grads = undo_loss_scaling(real_grads)
    r1_penalty = torch.sum(torch.mul(real_grads, real_grads))
    return r1_penalty


def R2Penalty(fake_img, f):
    # gradient penalty
    fakes = torch.autograd.Variable(fake_img, requires_grad=True).to(fake_img.device)
    fake_logit = f(fakes)
    apply_loss_scaling = lambda x: x * torch.exp(x * torch.Tensor([np.float32(np.log(2.0))]).to(fake_img.device))
    undo_loss_scaling = lambda x: x * torch.exp(-x * torch.Tensor([np.float32(np.log(2.0))]).to(fake_img.device))

    fake_logit = apply_loss_scaling(torch.sum(fake_logit))
    fake_grads = torch.autograd.grad(fake_logit, fakes, grad_outputs=torch.ones(fake_logit.size()).to(fakes.device), create_graph=True)[0].view(fakes.size(0), -1)
    fake_grads = undo_loss_scaling(fake_grads)
    r2_penalty = torch.sum(torch.mul(fake_grads, fake_grads))
    return r2_penalty


train_loader = torch.utils.data.DataLoader(epnd.ImageDataSet(
    root=[[DATA_PATH]],
    transform=tv.transforms.Compose([
        epnt.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        epnt.ToTensor(),
        epnt.ToFloat(),
        epnt.Transpose(epnt.BHWC2BCHW),
        epnt.Normalize(),
    ])),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

NetG = epns.StyleGenerator()
NetD = epns.StyleDiscriminator()

optim_D = torch.optim.Adam(NetD.parameters(), lr=0.0001, betas=(0.5, 0.999))
optim_G = torch.optim.Adam(NetG.parameters(), lr=0.0001, betas=(0.5, 0.999))
scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optim_D, gamma=0.99)
scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optim_G, gamma=0.99)

fix_z = torch.randn([100, 512])
softplus = torch.nn.Softplus()
Loss_D_list = [0.0]
Loss_G_list = [0.0]

if torch.cuda.is_available():
    NetD = NetD.cuda()
    NetG = NetG.cuda()
    fix_z = fix_z.cuda()

bar = eup.ProgressBar(EPOCH, len(train_loader), "D Loss:%.3f;G Loss:%.3f")
for epoch in range(1, EPOCH + 1):
    loss_D_list = []
    loss_G_list = []
    for i, (real_img,) in enumerate(train_loader):
        # =======================================================================================================
        #   (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        # =======================================================================================================
        # Compute adversarial loss toward discriminator
        NetD.zero_grad()

        noise_img = torch.randn([real_img.size(0), 512])
        if torch.cuda.is_available():
            real_img = real_img.cuda()
            noise_img = noise_img.cuda()

        real_logit = NetD(real_img)

        fake_img = NetG(noise_img)
        fake_logit = NetD(fake_img.detach())
        d_loss = softplus(fake_logit).mean()
        d_loss = d_loss + softplus(-real_logit).mean()

        if R1_GAMMA != 0.0:
            r1_penalty = R1Penalty(real_img.detach(), NetD)
            d_loss = d_loss + r1_penalty * (R1_GAMMA * 0.5)

        if R2_GAMMA != 0.0:
            r2_penalty = R2Penalty(fake_img.detach(), NetD)
            d_loss = d_loss + r2_penalty * (R2_GAMMA * 0.5)

        # Update discriminator
        d_loss.backward()
        optim_D.step()

        # =======================================================================================================
        #   (2) Update G network: maximize log(D(G(z)))
        # =======================================================================================================
        NetG.zero_grad()
        fake_logit = NetD(fake_img)
        g_loss = softplus(-fake_logit).mean()

        # Update generator
        g_loss.backward()
        optim_G.step()

        bar.show(epoch, g_loss.item(), d_loss.item())

    with torch.no_grad():
        fake_img = NetG(fix_z).detach().cpu()
        tv.utils.save_image(fake_img.data,'outputs/Face1024_%03d.png' % epoch, normalize=True, nrow=10)








