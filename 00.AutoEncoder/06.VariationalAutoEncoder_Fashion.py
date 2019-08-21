'''
@author: JJZHK
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: 06.VariationalAutoEncoder_Fashion.py
@time: 2019-04-24 15:59
@desc: 
'''
import torch
import torchvision as tv
import ELib.utils.progressbar as eup
import ELib.pyt.nuwa.dataset as epfd

num_epochs = 20
batch_size = 128
learning_rate = 0.001

img_transform = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    # tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = epfd.FashionMnistPytorchData(train=True, transform=img_transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

test_images = epfd.FashionMnistPytorchData(train=False).test_data

class VAE(torch.nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1  = torch.nn.Linear(784, 400)
        self.fc21 = torch.nn.Linear(400, 20)
        self.fc22 = torch.nn.Linear(400, 20)
        self.fc3  = torch.nn.Linear(20, 400)
        self.fc4  = torch.nn.Linear(400, 784)

    def encode(self, x):
        h1 = torch.nn.functional.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = torch.autograd.Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = torch.nn.functional.relu(self.fc3(z))
        return torch.nn.functional.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE().cuda() if torch.cuda.is_available() else VAE()
reconstruction_function = torch.nn.MSELoss(size_average=False)

def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)  # mse loss
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return BCE + KLD

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
proBar = eup.ProgressBar(num_epochs, len(dataloader), "loss:%.3f")

for epoch in range(1, num_epochs+1):
    train_loss = 0
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img = torch.autograd.Variable(img).cuda() if torch.cuda.is_available() else torch.autograd.Variable(img)
        # ===================forward=====================

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(img)
        loss = loss_function(recon_batch, img, mu, logvar)
        loss.backward()

        optimizer.step()

        proBar.show(epoch, loss.item() / len(img))

    if epoch % 1 == 0:
        test_image_show = torch.unsqueeze(torch.Tensor(test_images[:8]), 1)
        test_image_show = test_image_show.view(test_image_show.size(0), -1)
        output, mu, logvar = model(torch.autograd.Variable(test_image_show))
        output = torch.cat([test_image_show, output.cpu().data], dim=0)
        pic = output.view(output.size(0), 1, 28, 28)
        tv.utils.save_image(pic, './outputs/AE_{}.png'.format(epoch))
torch.save(model.state_dict(), "outputs/VAE.pth")