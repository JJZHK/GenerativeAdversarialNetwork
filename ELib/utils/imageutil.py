'''
@author: JJZHK
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: imageutil.py
@time: 2018/11/13 10:23
@desc: 
'''
import numpy as np
import cv2
from PIL import Image
import ELib.utils.functionutil as euf
import os
import imageio

'''
对图像进行水平或垂直翻转
'''
class Flip:
    def __init__(self,
                 dim='horizontal',
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        :param dim:horizontal or vertical
        :param labels_format:
        '''
        if not (dim in {'horizontal', 'vertical'}): raise ValueError("`dim` can be one of 'horizontal' and 'vertical'.")
        self.dim = dim
        self.labels_format = labels_format

    def __call__(self, image, labels=None, mask=None, return_inverter=False, calLabels=True):
        '''
        :param image:
        :param labels: [[classid, xmin,ymin,xmax,ymax],[],[],[],...]
        :param return_inverter:
        :return:
        '''
        img_height, img_width = image.shape[:2]

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        if self.dim == 'horizontal':
            image = image[:,::-1]
            if mask is not None:
                mask = mask[:, ::-1]

            if labels is not None and calLabels:
                labels = np.copy(labels)
                labels[:, [xmin, xmax]] = img_width - labels[:, [xmax, xmin]]

            return image, labels, mask
        else:
            image = image[::-1]
            if mask is not None:
                mask = mask[::-1]
            if labels is not None and calLabels:
                labels = np.copy(labels)
                labels[:, [ymin, ymax]] = img_height - labels[:, [ymax, ymin]]

            return image, labels, mask

'''
随机对图片进行水平或垂直翻转
'''
class RandomFlip:
    def __init__(self,
                 dim='horizontal',
                 prob=0.5,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        self.dim = dim
        self.prob = prob
        self.labels_format = labels_format
        self.flip = Flip(dim=self.dim, labels_format=self.labels_format)

    def __call__(self, image, labels=None, mask=None, calLabels=True):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.flip.labels_format = self.labels_format
            return self.flip(image, labels, mask, calLabels)
        else:
            return image, labels, mask

'''
对图片格式进行转换，从BGR,RGB,HSV变为RGB，HSV，GRAY, BGR
'''
class ConvertColor:
    def __init__(self, current='BGR', to='HSV', keep_3ch=True):
        if not ((current in {'BGR', 'RGB', 'HSV'}) and (to in {'RGB', 'HSV', 'GRAY', 'BGR'})):
            raise NotImplementedError
        self.current = current
        self.to = to
        self.keep_3ch = keep_3ch

    def __call__(self, image, labels=None, mask=None):
        if self.current != self.to:
            if self.current == "BGR":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if self.to == "HSV":
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif self.to == "GRAY":
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    if self.keep_3ch:
                        image = np.stack([image] * 3, axis=-1)
            elif self.current == "RGB":
                if self.to == "HSV":
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif self.to == "BGR":
                    image = cv2.cvtColor(image, cv2, cv2.COLOR_RGB2BGR)
                elif self.to == "GRAY":
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    if self.keep_3ch:
                        image = np.stack([image] * 3, axis=-1)
            elif self.current == "HSV":
                image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
                if self.to == "BGR":
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                elif self.to == "GRAY":
                    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    if self.keep_3ch:
                        image = np.stack([image] * 3, axis=-1)

        return image, labels, mask

'''
将图片内容变为uint8或float32
'''
class ConvertDataType:
    def __init__(self, to='uint8'):
        if not (to == 'uint8' or to == 'float32'):
            raise ValueError("`to` can be either of 'uint8' or 'float32'.")
        self.to = to

    def __call__(self, image, labels=None, mask=None):
        if self.to == 'uint8':
            image = np.round(image, decimals=0).astype(np.uint8)
        else:
            image = image.astype(np.float32)

        return image, labels, mask

'''
将图片变为3通道
'''
class ConvertTo3Channels:
    def __init__(self):
        pass

    def __call__(self, image, labels=None, mask=None):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3:
            if image.shape[2] == 1:
                image = np.concatenate([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:,:,:3]

        return image, labels, mask

'''
调整图像的色调，接收的图片为HSV格式图片，调整HSV中的H
'''
class Hue:
    def __init__(self, delta):
        if not (-180 <= delta <= 180): raise ValueError("`delta` must be in the closed interval `[-180, 180]`.")
        self.delta = delta

    def __call__(self, image, labels=None, mask=None):
        image[:, :, 0] = (image[:, :, 0] + self.delta) % 180.0

        return image, labels, mask

'''
随机调整图像的色调。接收图片为HSV图片
'''
class RandomHue:
    def __init__(self, max_delta=18, prob=0.5):
        '''
        Arguments:
            max_delta (int): An integer in the closed interval `[0, 180]` that determines the maximal absolute
                hue change.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
        '''
        if not (0 <= max_delta <= 180): raise ValueError("`max_delta` must be in the closed interval `[0, 180]`.")
        self.max_delta = max_delta
        self.prob = prob
        self.change_hue = Hue(delta=0)

    def __call__(self, image, labels=None, mask=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.change_hue.delta = np.random.uniform(-self.max_delta, self.max_delta)
            return self.change_hue(image, labels, mask)

        return image, labels, mask

'''
调整图像的饱和度，接收的图片为HSV格式图片，调整HSV中的S
'''
class Saturation:
    def __init__(self, factor):
        if factor <= 0.0: raise ValueError("It must be `factor > 0`.")
        self.factor = factor

    def __call__(self, image, labels=None, mask=None):
        image[:,:,1] = np.clip(image[:,:,1] * self.factor, 0, 255)

        return image, labels, mask

'''
随机调整图像的饱和度，接收的图片为HSV格式图片，调整HSV中的S
'''
class RandomSaturation:
    '''
    Randomly changes the saturation of HSV images.

    Important:
        - Expects HSV input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, lower=0.3, upper=2.0, prob=0.5):
        '''
        Arguments:
            lower (float, optional): A float greater than zero, the lower bound for the random
                saturation change.
            upper (float, optional): A float greater than zero, the upper bound for the random
                saturation change. Must be greater than `lower`.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
        '''
        if lower >= upper: raise ValueError("`upper` must be greater than `lower`.")
        self.lower = lower
        self.upper = upper
        self.prob = prob
        self.change_saturation = Saturation(factor=1.0)

    def __call__(self, image, labels=None, mask=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.change_saturation.factor = np.random.uniform(self.lower, self.upper)
            return self.change_saturation(image, labels)

        return image, labels, mask

'''
调整图片亮度，RGB图片
'''
class Brightness:
    def __init__(self, delta):
        self.delta = delta

    def __call__(self, image, labels=None, mask=None):
        image = np.clip(image + self.delta, 0, 255)

        return image, labels, mask

'''
随机调整图片亮度，RGB或BRG图片，必须为float32格式
'''
class RandomBrightness:
    def __init__(self, lower=-84, upper=84, prob=0.5):
        if lower >= upper: raise ValueError("`upper` must be greater than `lower`.")
        self.lower = float(lower)
        self.upper = float(upper)
        self.prob = prob
        self.change_brightness = Brightness(delta=0)

    def __call__(self, image, labels=None, mask=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.change_brightness.delta = np.random.uniform(self.lower, self.upper)
            return self.change_brightness(image, labels)

        return image, labels, mask

'''
调整图片对比度，RGB图片
'''
class Contrast:
    def __init__(self, factor):
        if factor <= 0.0: raise ValueError("It must be `factor > 0`.")
        self.factor = factor

    def __call__(self, image, labels=None, mask=None):
        image = np.clip(127.5 + self.factor * (image - 127.5), 0, 255)

        return image, labels, mask

'''
随机调整图片对比度，RGB或BRG图片，必须为float32格式
'''
class RandomContrast:
    def __init__(self, lower=0.5, upper=1.5, prob=0.5):
        if lower >= upper: raise ValueError("`upper` must be greater than `lower`.")
        self.lower = lower
        self.upper = upper
        self.prob = prob
        self.change_contrast = Contrast(factor=1.0)

    def __call__(self, image, labels=None, mask=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.change_contrast.factor = np.random.uniform(self.lower, self.upper)
            return self.change_contrast(image, labels)

        return image, labels, mask

'''
调整图片的shape，RGB图片
'''
class ChannelSwap:
    '''
    Swaps the channels of images.
    '''
    def __init__(self, order):
        '''
        Arguments:
            order (tuple): A tuple of integers that defines the desired channel order
                of the input images after the channel swap.
        '''
        self.order = order

    def __call__(self, image, labels=None, mask=None):
        image = image[:,:,self.order]

        return image, labels, mask

'''
随机调整图片的shape，RGB图片
'''
class RandomChannelSwap:
    '''
    Randomly swaps the channels of RGB images.

    Important: Expects RGB input.
    '''
    def __init__(self, prob=0.5):
        '''
        Arguments:
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
        '''
        self.prob = prob
        # All possible permutations of the three image channels except the original order.
        self.permutations = ((0, 2, 1),
                             (1, 0, 2), (1, 2, 0),
                             (2, 0, 1), (2, 1, 0))
        self.swap_channels = ChannelSwap(order=(0, 1, 2))

    def __call__(self, image, labels=None, mask=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            i = np.random.randint(5) # There are 6 possible permutations.
            self.swap_channels.order = self.permutations[i]
            return self.swap_channels(image, labels)

        return image, labels, mask

'''
调整图片大小
'''
class Resize:
    '''
    Resizes images to a specified height and width in pixels.
    '''

    def __init__(self,
                 height,
                 width,
                 interpolation_mode=cv2.INTER_LINEAR,
                 box_filter=None,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            height (int): The desired height of the output images in pixels.
            width (int): The desired width of the output images in pixels.
            interpolation_mode (int, optional): An integer that denotes a valid
                OpenCV interpolation mode. For example, integers 0 through 5 are
                valid interpolation modes.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''
        # if not (isinstance(box_filter, BoxFilter) or box_filter is None):
        #     raise ValueError("`box_filter` must be either `None` or a `BoxFilter` object.")
        self.out_height = height
        self.out_width = width
        self.interpolation_mode = interpolation_mode
        self.box_filter = box_filter
        self.labels_format = labels_format

    def __call__(self, image, labels=None, mask=None, calLabels=True):

        img_height, img_width = image.shape[:2]

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        image = cv2.resize(image,
                           dsize=(self.out_width, self.out_height),
                           interpolation=self.interpolation_mode)
        if mask is not None:
            mask = cv2.resize(mask,
                              dsize=(self.out_width, self.out_height),
                              interpolation=self.interpolation_mode)

        if labels is not None and calLabels:
            labels = np.copy(labels)
            labels[:, [ymin, ymax]] = np.round(labels[:, [ymin, ymax]] * (self.out_height / img_height), decimals=0)
            labels[:, [xmin, xmax]] = np.round(labels[:, [xmin, xmax]] * (self.out_width / img_width), decimals=0)

            if not (self.box_filter is None):
                self.box_filter.labels_format = self.labels_format
                labels = self.box_filter(labels=labels,image_height=self.out_height,image_width=self.out_width)
        return image, labels, mask

'''
使用随机的方式，调整图片的大小
'''
class ResizeRandomInterp:
    '''
    Resizes images to a specified height and width in pixels using a radnomly
    selected interpolation mode.
    '''

    def __init__(self,
                 height,
                 width,
                 interpolation_modes=[cv2.INTER_NEAREST,
                                      cv2.INTER_LINEAR,
                                      cv2.INTER_CUBIC,
                                      cv2.INTER_AREA,
                                      cv2.INTER_LANCZOS4],
                 box_filter=None,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            height (int): The desired height of the output image in pixels.
            width (int): The desired width of the output image in pixels.
            interpolation_modes (list/tuple, optional): A list/tuple of integers
                that represent valid OpenCV interpolation modes. For example,
                integers 0 through 5 are valid interpolation modes.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''
        if not (isinstance(interpolation_modes, (list, tuple))):
            raise ValueError("`interpolation_mode` must be a list or tuple.")
        self.height = height
        self.width = width
        self.interpolation_modes = interpolation_modes
        self.box_filter = box_filter
        self.labels_format = labels_format
        self.resize = Resize(height=self.height,
                             width=self.width,
                             box_filter=self.box_filter,
                             labels_format=self.labels_format)

    def __call__(self, image, labels=None, mask=None, calLabels=True):
        self.resize.interpolation_mode = np.random.choice(self.interpolation_modes)
        self.resize.labels_format = self.labels_format
        return self.resize(image, labels, mask, calLabels)

'''
将CV2的BGR图片转换为PIL.Image对象
'''
class CV2ToPIL:
    def __init__(self):
        pass

    def __call__(self, image):
        image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        return image


class PILToCV2:
    def __init__(self):
        pass

    def __call__(self, image):
        image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        return image


class Blur:
    def __init__(self, ksize=5):
        self.ksize = ksize

    def __call__(self, image, label=None, mask=None):
        return cv2.blur(image, ksize=(self.ksize, self.ksize))


class RandomBlur:
    def __init__(self, prob=0.5, ksize=5):
        self.prob = prob
        self.blur = Blur(ksize)

    def __call__(self, image, label=None, mask=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            image = self.blur(image, label, mask)

        return image, label, mask


class Scale:
    def __init__(self, lower=0.6, upper=1.4):
        self.lower = lower
        self.upper = upper

    def __call__(self, image, label=None, mask=None, calLabels=True):
        img_height, img_width = image.shape[:2]
        scale = np.random.uniform(self.lower,self.upper)
        image = cv2.resize(image,(int(img_width*scale),img_height))
        if mask is not None:
            mask = cv2.resize(mask, (int(img_width*scale),img_height))

        if label is not None and calLabels:
            label = np.asarray(label)
            scale_tensor = [[1, scale,1,scale,1]]
            label = label * scale_tensor

        return image, label, mask


class RandomScale:
    def __init__(self, prob=0.5):
        self.prob = prob
        self.scale = Scale(lower=0.6, upper=1.4)

    def __call__(self, image, label=None, mask=None, calLabels=True):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            return self.scale(image, label, mask, calLabels)

        return image, label,mask

'''
将图片减去均值，默认均值为RGB格式的均值，如果是BGR，mean=(104, 117, 123)
'''
class SubMean:
    def __init__(self, mean=(123,117,104)):
        self.mean = mean

    def __call__(self, image, labels=None, mask=None):
        mean = np.array(self.mean, dtype=np.float32)
        image = image - mean

        return image, labels, mask

'''
将图片进行缩放，注意mean这个均值，如果是RGB图片，需要输入RBG的均值，如果是BGR图片，需要的是BGR的均值
'''
class Expand(object):
    def __init__(self, mean,labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        self.mean = mean
        self.labels_format = labels_format

    def __call__(self, image, labels=None, mask=None, calLabels=True):
        height, width, depth = image.shape
        ratio = np.random.uniform(1, 4)
        left = np.random.uniform(0, width*ratio - width)
        top = np.random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),int(left):int(left + width)] = image
        image = expand_image

        if mask is not None:
            expand_image = np.zeros(
                (int(height*ratio), int(width*ratio), depth),
                dtype=image.dtype)
            expand_image[:, :, :] = self.mean
            expand_image[int(top):int(top + height),int(left):int(left + width)] = mask
            mask = expand_image

        if labels is not None and calLabels:
            labels = np.asarray(labels)
            labels[:, 1:5][:, :2] += (int(left), int(top))
            labels[:, 1:5][:, 2:] += (int(left), int(top))

        return image, labels, mask

'''
对图片进行放大，默认是BGR图片，如果输入为RGB图片，需要调整mean的shape
'''
class RandomExpand:
    def __init__(self, prob=0.5,mean=(104, 117, 123)):
        self.prob = prob
        self.expand = Expand(mean)

    def __call__(self, image, label=None, mask=None, calLabels=True):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            return self.expand(image, label, mask, calLabels)

        return image, label, mask

'''
将Box尺寸归一化
'''
class Normalization:
    def __init__(self):
        pass

    def __call__(self, image, labels=None, mask=None):
        height, width, channels = image.shape
        labels = np.asarray(labels, dtype=np.float32)
        boxes = labels[:, 1:5].copy()
        boxes[:, 0::2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height
        labels[:, 1:5] = boxes
        return image, labels, mask


'''
对图片进行平移
'''
class Shift:
    def __init__(self):
        pass

    def __call__(self, image, labels=None, mask=None, calLabels=True):
        relabels = np.asarray(labels)
        boxes = relabels[:, 1:5]
        label = relabels[:, 0]

        center = (boxes[:,2:]+boxes[:,:2])/2
        height,width,c = image.shape
        after_shfit_image = np.zeros((height,width,c),dtype=image.dtype)
        after_shfit_image[:,:,:] = (104,117,123) #bgr

        after_shift_mask = np.zeros((height,width,c),dtype=image.dtype)
        after_shift_mask[:,:,:] = (104,117,123)

        shift_x = np.random.uniform(-width*0.2,width*0.2)
        shift_y = np.random.uniform(-height*0.2,height*0.2)

        #原图像的平移
        if shift_x>=0 and shift_y>=0:
            after_shfit_image[int(shift_y):,int(shift_x):,:] = image[:height-int(shift_y),:width-int(shift_x),:]
            if mask is not None:
                after_shift_mask[int(shift_y):,int(shift_x):,:] = mask[:height-int(shift_y),:width-int(shift_x),:]
        elif shift_x>=0 and shift_y<0:
            after_shfit_image[:height+int(shift_y),int(shift_x):,:] = image[-int(shift_y):,:width-int(shift_x),:]
            if mask is not None:
                after_shift_mask[:height+int(shift_y),int(shift_x):,:] = mask[-int(shift_y):,:width-int(shift_x),:]
        elif shift_x <0 and shift_y >=0:
            after_shfit_image[int(shift_y):,:width+int(shift_x),:] = image[:height-int(shift_y),-int(shift_x):,:]
            if mask is not None:
                after_shift_mask[int(shift_y):,:width+int(shift_x),:] = mask[:height-int(shift_y),-int(shift_x):,:]
        elif shift_x<0 and shift_y<0:
            after_shfit_image[:height+int(shift_y),:width+int(shift_x),:] = image[-int(shift_y):,-int(shift_x):,:]
            if mask is not None:
                after_shift_mask[:height+int(shift_y),:width+int(shift_x),:] = mask[-int(shift_y):,-int(shift_x):,:]

        shift_xy = np.reshape(np.tile(np.asarray([[int(shift_x),int(shift_y)]]),center.shape[0]), newshape=center.shape)
        center = center + shift_xy
        mask1 = (center[:,0] >0) & (center[:,0] < width)
        mask2 = (center[:,1] >0) & (center[:,1] < height)
        mask = np.reshape((mask1 & mask2) + 0, newshape=(-1,1))
        boxes_in = np.reshape(boxes[np.asarray(np.tile(mask, boxes.shape[1]), dtype=np.bool)], newshape=(-1,boxes.shape[1]))
        if len(boxes_in) == 0:
            relabels[:, 1:5] = boxes
            return image,relabels, mask

        box_shift = np.reshape(np.tile(np.asarray([[int(shift_x),int(shift_y),int(shift_x),int(shift_y)]]), boxes_in.shape[0]), newshape=boxes_in.shape)
        boxes_in = boxes_in+box_shift
        labels_in = label[np.asarray(np.reshape(mask, newshape=label.shape), dtype=np.bool)]
        relabels = np.zeros(shape=(boxes_in.shape[0], boxes_in.shape[1] + 1))
        relabels[:, 1:5] = boxes_in
        relabels[:, 0] = labels_in
        return after_shfit_image,relabels, after_shift_mask


class RandomShift:
    def __init__(self, prob=0.5):
        self.prob = prob
        self.shift = Shift()

    def __call__(self, image, label=None, mask=None, calLabels=True):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            return self.shift(image, label, mask, calLabels)

        return image, label, mask

'''
对图片进行裁剪
'''
class Crop:
    def __init__(self):
        pass

    def __call__(self, image, labels=None, maskImage=None, calLabels=True):
        relabels = np.asarray(labels)
        boxes = relabels[:, 1:5]
        label = relabels[:, 0]

        center = (boxes[:,2:]+boxes[:,:2])/2
        height,width,c = image.shape
        h = np.random.uniform(0.6*height,height)
        w = np.random.uniform(0.6*width,width)
        x = np.random.uniform(0,width-w)
        y = np.random.uniform(0,height-h)
        x,y,h,w = int(x),int(y),int(h),int(w)

        center1 = np.reshape(np.tile(np.asarray([[x,y]]),center.shape[0]), newshape=center.shape)
        center = center - center1 #torch.FloatTensor([[x,y]]).expand_as(center)
        mask1 = (center[:,0]>0) & (center[:,0]<w)
        mask2 = (center[:,1]>0) & (center[:,1]<h)
        mask = np.reshape((mask1 & mask2) + 0, newshape=(-1,1))

        # boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
        boxes_in = np.reshape(boxes[np.asarray(np.tile(mask, boxes.shape[1]), dtype=np.bool)], newshape=(-1,boxes.shape[1]))
        if(len(boxes_in)==0):
            relabels[:, 1:5] = boxes
            return image,relabels, maskImage

        # box_shift = torch.FloatTensor([[x,y,x,y]]).expand_as(boxes_in)
        box_shift = np.reshape(np.tile(np.asarray([[x,y,x,y]]),boxes_in.shape[0]), newshape=boxes_in.shape)
        boxes_in = boxes_in - box_shift
        boxes_in[:,0]= euf.Functional.clamp(boxes_in[:,0], min=0,max=w)
        boxes_in[:,2]= euf.Functional.clamp(boxes_in[:,2], min=0,max=w)
        boxes_in[:,1]= euf.Functional.clamp(boxes_in[:,1], min=0,max=h)
        boxes_in[:,3]= euf.Functional.clamp(boxes_in[:,3], min=0,max=h)

        # labels_in = label[mask.view(-1)]
        labels_in = label[np.asarray(np.reshape(mask, newshape=label.shape), dtype=np.bool)]
        img_croped = image[y:y+h,x:x+w,:]
        mask_croped = None
        if maskImage is not None:
            mask_croped = maskImage[y:y+h,x:x+w,:]
        relabels = np.zeros(shape=(boxes_in.shape[0], boxes_in.shape[1] + 1))
        relabels[:, 1:5] = boxes_in
        relabels[:, 0] = labels_in
        return img_croped,relabels, mask_croped


class RandomCrop:
    def __init__(self, prob=0.5):
        self.prob = prob
        self.crop = Crop()

    def __call__(self, image, label=None, mask=None, calLabels=True):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            return self.crop(image, label, mask, calLabels)

        return image, label, mask

class ImageToGif:
    def __init__(self):
        pass

    def makeGif(self, outputs):
        filenames=sorted((os.path.join(outputs, fn) for fn in os.listdir(outputs) if fn.endswith('.png') or fn.endswith('.jpg')))
        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave('{}.gif'.format(filename[:filename.rfind('.')]), images,duration=0.1)

# if __name__ == '__main__':
#     import lib.utils.BasicDataSet as lub
#     import lib.utils.drawutils as lud
#     import lib.utils.Config as luc
#     import cv2
#
#     data = luc.VOC_Data_Config().getClasses()
#
#     dataset = lub.VOCDataSet(imageset="test", isSegmentation=True)
#
#     arguments = [
#         RandomShift(prob=0.9)
#                 ]
#
#     for i, (image, labels, mask) in enumerate(dataset):
#         img_id = dataset.__getItemInfo__(i)["img_id"]
#         cv2.imwrite("test_%s.jpg" % img_id, image)
#         labels = np.asarray(labels)
#         box = labels[:, 1:5]
#         label = labels[:, 0]
#         for item in arguments:
#             image, labels, mask = item(image, labels, mask=mask, calLabels=True)
#
#         boxes = []
#         for label in labels:
#             boxes.append([(label[1], label[2]),(label[3], label[4]), data[int(label[0])], "", 1.00])
#
#         lud.draw_box_by_cv2(image, boxes, "test_%s_.jpg" % img_id, mask=mask)
#         if i > 5:
#             break
