'''
@author: JJZHK
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: utils.py
@time: 2019-05-05 14:27
@desc: 
'''
import functools
import itertools
import numpy
import skimage.transform as st
import skimage.filters as sf
import matplotlib.pyplot as plt
plt.switch_backend('agg')

'''
求一个维度上的边缘(marginal)概率+另一维度上近似的条件概率。比如把图像中白色像素的值作为概率密度的相对大小，然后沿着x求和，
然后在y轴上求出marginal probability density，接着再根据y的位置，近似得到对应x关于y的条件概率。采样的时候先采y的值，再采x
的值就能近似得到符合图像描述的分布的样本
'''
def generate_lut(img):
    EPS = 1e-66
    RESOLUTION = 0.001
    num_grids = int(1/RESOLUTION+0.5)
    density_img = st.resize(img, (num_grids, num_grids))
    x_accumlation = numpy.sum(density_img, axis=1)
    sum_xy = numpy.sum(x_accumlation)
    y_cdf_of_accumulated_x = [[0., 0.]]
    accumulated = 0
    for ir, i in enumerate(range(num_grids-1, -1, -1)):
        accumulated += x_accumlation[i]
        if accumulated == 0:
            y_cdf_of_accumulated_x[0][0] = float(ir+1)/float(num_grids)
        elif EPS < accumulated < sum_xy - EPS:
            y_cdf_of_accumulated_x.append([float(ir+1)/float(num_grids), accumulated/sum_xy])
        else:
            break
    y_cdf_of_accumulated_x.append([float(ir+1)/float(num_grids), 1.])
    y_cdf_of_accumulated_x = numpy.array(y_cdf_of_accumulated_x)

    x_cdfs = []
    for j in range(num_grids):
        x_freq = density_img[num_grids-j-1]
        sum_x = numpy.sum(x_freq)
        x_cdf = [[0., 0.]]
        accumulated = 0
        for i in range(num_grids):
            accumulated += x_freq[i]
            if accumulated == 0:
                x_cdf[0][0] = float(i+1) / float(num_grids)
            elif EPS < accumulated < sum_xy - EPS:
                x_cdf.append([float(i+1)/float(num_grids), accumulated/sum_x])
            else:
                break
        x_cdf.append([float(i+1)/float(num_grids), 1.])
        if accumulated > EPS:
            x_cdf = numpy.array(x_cdf)
            x_cdfs.append(x_cdf)
        else:
            x_cdfs.append(None)

    y_lut = functools.partial(numpy.interp, xp=y_cdf_of_accumulated_x[:, 1], fp=y_cdf_of_accumulated_x[:, 0])
    x_luts = [functools.partial(numpy.interp, xp=x_cdfs[i][:, 1], fp=x_cdfs[i][:, 0]) if x_cdfs[i] is not None else None for i in range(num_grids)]

    return y_lut, x_luts

def sample_2d(lut, N):
    RESOLUTION = 0.001
    y_lut, x_luts = lut
    u_rv = numpy.random.random((N, 2))
    samples = numpy.zeros(u_rv.shape)
    for i, (x, y) in enumerate(u_rv):
        ys = y_lut(y)
        x_bin = int(ys/RESOLUTION)
        xs = x_luts[x_bin](x)
        samples[i][0] = xs
        samples[i][1] = ys

    return samples

class GANDemoVisualizer:

    def __init__(self, title, l_kde=100, bw_kde=5):
        self.title = title
        self.l_kde = l_kde
        self.resolution = 1. / self.l_kde
        self.bw_kde_ = bw_kde
        self.fig, self.axes = plt.subplots(ncols=3, figsize=(13.5, 4))
        self.fig.canvas.set_window_title(self.title)

    def draw(self, real_samples, gen_samples, msg=None, cmap='hot', pause_time=0.05, max_sample_size=500, show=True):
        if msg:
            self.fig.suptitle(msg)
        ax0, ax1, ax2 = self.axes

        self.draw_samples(ax0, 'real and generated samples', real_samples, gen_samples, max_sample_size)
        self.draw_density_estimation(ax1, 'density: real samples', real_samples, cmap)
        self.draw_density_estimation(ax2, 'density: generated samples', gen_samples, cmap)

        if show:
            plt.draw()
            plt.pause(pause_time)

    @staticmethod
    def draw_samples(axis, title, real_samples, generated_samples, max_sample_size):
        axis.clear()
        axis.set_xlabel(title)
        axis.plot(generated_samples[:max_sample_size, 0], generated_samples[:max_sample_size, 1], '.')
        axis.plot(real_samples[:max_sample_size, 0], real_samples[:max_sample_size, 1], 'kx')
        axis.axis('equal')
        axis.axis([0, 1, 0, 1])

    def draw_density_estimation(self, axis, title, samples, cmap):
        axis.clear()
        axis.set_xlabel(title)
        density_estimation = numpy.zeros((self.l_kde, self.l_kde))
        for x, y in samples:
            if 0 < x < 1 and 0 < y < 1:
                density_estimation[int((1-y) / self.resolution)][int(x / self.resolution)] += 1
        density_estimation = sf.gaussian(density_estimation, self.bw_kde_)
        axis.imshow(density_estimation, cmap=cmap)
        axis.xaxis.set_major_locator(plt.NullLocator())
        axis.yaxis.set_major_locator(plt.NullLocator())

    def savefig(self, filepath):
        self.fig.savefig(filepath)

    @staticmethod
    def show():
        plt.show()

class CGANDemoVisualizer(GANDemoVisualizer):

    def __init__(self, title, l_kde=100, bw_kde=5):
        GANDemoVisualizer.__init__(self, title, l_kde, bw_kde)

    def draw(self, real_samples, gen_samples, msg=None, cmap='hot', pause_time=0.05, max_sample_size=500, show=True):
        if msg:
            self.fig.suptitle(msg)
        ax0, ax1, ax2 = self.axes

        self.draw_samples(ax0, 'real and generated samples', real_samples, gen_samples, max_sample_size)
        self.draw_density_estimation(ax1, 'density: real samples', real_samples[:, -2:], cmap)
        self.draw_density_estimation(ax2, 'density: generated samples', gen_samples[:, -2:], cmap)

        if show:
            plt.draw()
            plt.pause(pause_time)

    def draw_samples(self, axis, title, real_samples, generated_samples, max_sample_size):
        axis.clear()
        axis.set_xlabel(title)
        g_samples = numpy.copy(generated_samples)
        r_samples = numpy.copy(real_samples)
        numpy.random.shuffle(g_samples)
        numpy.random.shuffle(r_samples)
        g_samples = g_samples[:max_sample_size, :]
        r_samples = r_samples[:max_sample_size, :]
        color_iter = itertools.cycle('bgrcmy')
        for i in range(g_samples.shape[1]-2):
            c = next(color_iter)
            samples = g_samples[g_samples[:, i] > 0, :][:, -2:]
            axis.plot(samples[:, 0], samples[:, 1], c+'.', markersize=5)
            samples = r_samples[r_samples[:, i] > 0, :][:, -2:]
            axis.plot(samples[:, 0], samples[:, 1], c+'x', markersize=5)
        axis.axis('equal')
        axis.axis([0, 1, 0, 1])

    def savefig(self, filepath):
        self.fig.savefig(filepath)

    @staticmethod
    def show():
        plt.show()

# if __name__ == '__main__':
#     from skimage import io
#     density_img = io.imread('vortex.jpg', True)
#     lut_2d = generate_lut(density_img)
#     samples = sample_2d(lut_2d, 10000)
#
#     from matplotlib import pyplot as plt
#     fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(9, 4))
#     fig.canvas.set_window_title('Test 2D Sampling')
#     ax0.imshow(density_img, cmap='gray')
#     ax0.xaxis.set_major_locator(plt.NullLocator())
#     ax0.yaxis.set_major_locator(plt.NullLocator())
#
#     ax1.axis('equal')
#     ax1.axis([0, 1, 0, 1])
#     ax1.plot(samples[:, 0], samples[:, 1], 'k,')
#     plt.show()