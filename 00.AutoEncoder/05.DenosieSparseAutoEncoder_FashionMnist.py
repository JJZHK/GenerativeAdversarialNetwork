'''
@author: JJZHK
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: 05.DenosieSparseAutoEncoder_FashionMnist.py
@time: 2019-04-03 14:58
@desc: 
'''

import torch
import torchvision as tv
import ELib.utils.progressbar as eup
import ELib.pyt.nuwa.dataset as epfd
import numpy as np

num_epochs = 20
batch_size = 64
learning_rate = 0.001
noise_factor = 0.5
img_transform = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = epfd.FashionMnistPytorchData(train=True, transform=img_transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

test_images = epfd.FashionMnistPytorchData(train=False).test_data

class autoencoder(torch.nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(True))
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, 28 * 28),
            torch.nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def kl_divergence(p, p_hat):
    funcs = torch.nn.Sigmoid()
    p_hat = torch.mean(funcs(p_hat), 1)
    p_tensor = torch.Tensor([p] * len(p_hat)).cuda() if torch.cuda.is_available() else torch.Tensor([p] * len(p_hat))
    return torch.sum(p_tensor * torch.log(p_tensor) - p_tensor * torch.log(p_hat) + (1 - p_tensor) * torch.log(1 - p_tensor) - (1 - p_tensor) * torch.log(1 - p_hat))

def sparse_loss(autoencoder, images):
    loss = 0
    values = images
    for i in range(3):
        fc_layer = list(autoencoder.encoder.children())[2 * i]
        relu = list(autoencoder.encoder.children())[2 * i + 1]
        values = fc_layer(values)
        loss += kl_divergence(0.3, values)
    for i in range(2):
        fc_layer = list(autoencoder.decoder.children())[2 * i]
        relu = list(autoencoder.decoder.children())[2 * i + 1]
        values = fc_layer(values)
        loss += kl_divergence(0.3, values)
    return loss

model = autoencoder().cuda() if torch.cuda.is_available() else autoencoder()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)
proBar = eup.ProgressBar(num_epochs, len(dataloader), "loss:%.3f")
for epoch in range(1, num_epochs+1):
    for data in dataloader:
        img, _ = data

        noisy_imgs = img.cpu().data.numpy() + noise_factor * np.random.randn(*img.shape)
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)

        img = torch.Tensor(noisy_imgs)

        img = img.view(img.size(0), -1)
        img = torch.autograd.Variable(img).cuda() if torch.cuda.is_available() else torch.autograd.Variable(img)
        # ===================forward=====================
        output = model(img)
        mse_loss =   criterion(output, img)
        kl_loss = sparse_loss(model, img)
        loss = mse_loss + kl_loss * 1e-3

        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        proBar.show(epoch, loss.item())

    if epoch % 5 == 0:
        test_image_show = torch.unsqueeze(torch.Tensor(test_images[:8]), 1)

        noisy_imgs = test_image_show.cpu().data.numpy() + noise_factor * np.random.randn(*test_image_show.shape)
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)

        test_image_show = torch.Tensor(noisy_imgs)

        test_image_show = test_image_show.view(test_image_show.size(0), -1)
        output = model(torch.autograd.Variable(test_image_show))
        output = torch.cat([test_image_show, output.cpu().data], dim=0)
        pic = output.view(output.size(0), 1, 28, 28)
        tv.utils.save_image(pic, './outputs/DSAE_{}.png'.format(epoch))

torch.save(model.state_dict(), "outputs/SAE.pth")