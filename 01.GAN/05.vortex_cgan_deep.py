'''
@author: JJZHK
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: 01.batman.py
@time: 2019-05-05 15:41
@desc: 
'''
import skimage.io as si
import ELib.pyt.nuwa.utils as epnu
import ELib.utils.progressbar as eup
import ELib.utils.imageutil as epi
import torch
import os
import numpy

class DeepMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepMLP, self).__init__()
        self.map1 = torch.nn.Linear(input_size, hidden_size)
        self.map2 = torch.nn.Linear(hidden_size, hidden_size)
        self.map3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.nn.LeakyReLU(negative_slope = 0.1)(self.map1(x))
        x = torch.nn.LeakyReLU(negative_slope = 0.1)(self.map2(x))
        return torch.nn.functional.sigmoid(self.map3(x))

z_dim = 2
DIMENSION = 2
iterations = 3000
bs = 2000

INPUT_IMAGE_PATH = "./inputs/vortex"
image_paths = [os.sep.join([INPUT_IMAGE_PATH, x]) for x in os.listdir(INPUT_IMAGE_PATH)]
density_imgs = [si.imread(x, True) for x in image_paths]
luts_2d = [epnu.generate_lut(x) for x in density_imgs]
# Sampling based on visual density, a too small batch size may result in failure with conditions
pix_sums = [numpy.sum(x) for x in density_imgs]
total_pix_sums = numpy.sum(pix_sums)
c_indices = [0] + [int(sum(pix_sums[:i+1])/total_pix_sums*bs+0.5) for i in range(len(pix_sums)-1)] + [bs]

c_dim = len(luts_2d)    # Dimensionality of condition labels <--> number of images


visualizer = epnu.CGANDemoVisualizer('GAN 2D Example Visualization of {}'.format(INPUT_IMAGE_PATH))
generator = DeepMLP(input_size=z_dim+c_dim, hidden_size=50, output_size=DIMENSION)
discriminator = DeepMLP(input_size=DIMENSION+c_dim, hidden_size=100, output_size=1)

if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()

criterion = torch.nn.BCELoss()
d_optimizer = torch.optim.Adadelta(discriminator.parameters(), lr=1)
g_optimizer = torch.optim.Adadelta(generator.parameters(), lr=1)
proBar = eup.ProgressBar(1, iterations, "D Loss:(real/fake) %.3f/%.3f,G Loss:%.3f")

y = numpy.zeros((bs, c_dim))
for i in range(c_dim):
    y[c_indices[i]:c_indices[i + 1], i] = 1  # conditional labels, one-hot encoding
y = torch.autograd.Variable(torch.Tensor(y))
if torch.cuda.is_available():
    y = y.cuda()

for train_iter in range(1, iterations + 1):
    for d_index in range(3):
        # 1. Train D on real+fake
        discriminator.zero_grad()

        #  1A: Train D on real
        real_samples = numpy.zeros((bs, DIMENSION))
        for i in range(c_dim):
            real_samples[c_indices[i]:c_indices[i+1], :] = epnu.sample_2d(luts_2d[i], c_indices[i+1]-c_indices[i])

        # first c dimensions is the condition inputs, the last 2 dimensions are samples
        real_samples = torch.autograd.Variable(torch.Tensor(real_samples))
        if torch.cuda.is_available():
            real_samples = real_samples.cuda()
        d_real_data = torch.cat([y, real_samples], 1)
        if torch.cuda.is_available():
            d_real_data = d_real_data.cuda()

        d_real_decision = discriminator(d_real_data)
        labels = torch.autograd.Variable(torch.ones(bs))
        if torch.cuda.is_available() > 0:
            labels = labels.cuda()
        d_real_loss = criterion(d_real_decision, labels)  # ones = true

        #  1B: Train D on fake
        latent_samples = torch.autograd.Variable(torch.randn(bs, z_dim))
        if torch.cuda.is_available():
            latent_samples = latent_samples.cuda()

        d_gen_input = torch.cat([y, latent_samples], 1)
        d_fake_data = generator(d_gen_input).detach()  # detach to avoid training G on these labels
        conditional_d_fake_data = torch.cat([y, d_fake_data], 1)

        if torch.cuda.is_available():
            conditional_d_fake_data = conditional_d_fake_data.cuda()
        d_fake_decision = discriminator(conditional_d_fake_data)
        labels = torch.autograd.Variable(torch.zeros(bs))
        if torch.cuda.is_available():
            labels = labels.cuda()
        d_fake_loss = criterion(d_fake_decision, labels)  # zeros = fake

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()

        d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()


    for g_index in range(1):
        # 2. Train G on D's response (but DO NOT train D on these labels)
        generator.zero_grad()

        latent_samples = torch.randn(bs, z_dim)
        latent_samples = torch.autograd.Variable(latent_samples)
        if torch.cuda.is_available() > 0:
            latent_samples = latent_samples.cuda()

        g_gen_input = torch.cat([y, latent_samples], 1)
        g_fake_data = generator(g_gen_input)
        conditional_g_fake_data = torch.cat([y, g_fake_data], 1)
        g_fake_decision = discriminator(conditional_g_fake_data)
        labels = torch.autograd.Variable(torch.ones(bs))
        if torch.cuda.is_available():
            labels = labels.cuda()
        g_loss = criterion(g_fake_decision, labels)  # we want to fool, so pretend it's all genuine

        g_loss.backward()
        g_optimizer.step()  # Only optimizes G's parameters

    loss_d_real = d_real_loss.item()
    loss_d_fake = d_fake_loss.item()
    loss_g = g_loss.item()

    proBar.show(1, loss_d_real, loss_d_fake, loss_g)
    if train_iter == 1 or train_iter % 100 == 0:
        # loss_d_real = d_real_loss.data.cpu().numpy()[0] if torch.cuda.is_available() else d_real_loss.data.numpy()[0]
        # loss_d_fake = d_fake_loss.data.cpu().numpy()[0] if torch.cuda.is_available() else d_fake_loss.data.numpy()[0]
        # loss_g = g_loss.data.cpu().numpy()[0] if torch.cuda.is_available() else g_loss.data.numpy()[0]

        msg = 'Iteration {}: D_loss(real/fake): {:.6g}/{:.6g} G_loss: {:.6g}'.format(train_iter, loss_d_real, loss_d_fake, loss_g)

        real_samples_with_y = d_real_data.data.cpu().numpy() if torch.cuda.is_available() else d_real_data.data.numpy()
        gen_samples_with_y = conditional_g_fake_data.data.cpu().numpy() if torch.cuda.is_available() else conditional_g_fake_data.data.numpy()

        visualizer.draw(real_samples_with_y, gen_samples_with_y, msg, show=False)
        visualizer.savefig('outputs/Pytorch_Z_%04d' % train_iter)

maker = epi.ImageToGif()
maker.makeGif("outputs")
torch.save(generator.state_dict(), "outputs/GAN_Z_Pytorch_Generator.pth")