'''
@author: JJZHK
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: transforms.py
@time: 2019-07-01 10:06
@desc: 
'''
import torch
import numpy as np
import skimage.transform

UNDER_SAMPLING = 0
OVER_SAMPLING = 1
BCHW2BHWC = 0
BHWC2BCHW = 1


class OP():
    def work(self, tensor):
        """
            The virtual function to define the process in child class

            Arg:    tensor  - The np.ndarray object. The tensor you want to deal with
        """
        raise NotImplementedError("You should define your own function in the class!")

    def __call__(self, tensor):
        """
            This function define the proceeding of the operation
            There are different choice toward the tensor parameter
            1. torch.Tensor and rank is CHW
            2. np.ndarray and rank is CHW
            3. torch.Tensor and rank is TCHW
            4. np.ndarray and rank is TCHW

            Arg:    tensor  - The tensor you want to operate
            Ret:    The operated tensor
        """
        isTensor = type(tensor) == torch.Tensor
        if isTensor:
            tensor_type = tensor.type()
            tensor = tensor.cpu().data.numpy()
        if len(tensor.shape) == 3:
            tensor = self.work(tensor)
        elif len(tensor.shape) == 4:
            tensor = np.asarray([self.work(_) for _ in tensor])
        else:
            raise Exception("We dont support the rank format {}".format(tensor.shape),
                            "If the rank of the tensor shape is only 2, you can call 'GrayStack()'")
        if isTensor:
            tensor = torch.from_numpy(tensor)
            tensor = tensor.type(tensor_type)
        return tensor


class Resize(OP):
    def __init__(self, output_size):
        """
            Resize the tensor to the desired size
            This function only support for nearest-neighbor interpolation
            Since this mechanism can also deal with categorical data

            Arg:    output_size - The tuple (H, W)
        """
        self.output_size = output_size

    def work(self, tensor):
        """
            Resize the tensor
            If the tensor is not in the range of [-1, 1], we will do the normalization automatically

            Arg:    tensor  - The np.ndarray object. The tensor you want to deal with
            Ret:    The resized tensor
        """
        # Normalize the tensor if needed
        mean, std = -1, -1
        min_v = np.min(tensor)
        max_v = np.max(tensor)
        if not (max_v <= 1 and min_v >= -1):
            mean = 0.5 * max_v + 0.5 * min_v
            std  = 0.5 * max_v - 0.5 * min_v
            # print(max_v, min_v, mean, std)
            tensor = (tensor - mean) / std

        # Work
        tensor = skimage.transform.resize(tensor, self.output_size, mode = 'constant', order = 0)

        # De-normalize the tensor
        if mean != -1 and std != -1:
            tensor = tensor * std + mean
        return tensor


class Normalize(OP):
    def __init__(self, mean = [127.5, 127.5, 127.5], std = [127.5, 127.5, 127.5]):
        """
            Normalize the tensor with given mean and standard deviation
            * Notice: If you didn't give mean and std, the result will locate in [-1, 1]

            Args:
                mean        - The mean of the result tensor
                std         - The standard deviation
        """
        self.mean = mean
        self.std  = std

    def work(self, tensor):
        """
            Normalize the tensor

            Arg:    tensor  - The np.ndarray object. The tensor you want to deal with
            Ret:    The normalized tensor
        """
        if tensor.shape[0] != len(self.mean):
            raise Exception("The rank format should be BCHW, but the shape is {}".format(tensor.shape))
        result = []
        for t, m, s in zip(tensor, self.mean, self.std):
            result.append((t - m) / s)
        tensor = np.asarray(result)

        # Check if the normalization can really work
        if np.min(tensor) < -1 or np.max(tensor) > 1:
            raise Exception("Normalize can only work with float tensor",
                            "Try to call 'ToFloat()' before normalization")
        return tensor


class UnNormalize(OP):
    def __init__(self, mean = [127.5, 127.5, 127.5], std = [127.5, 127.5, 127.5]):
        """
            Unnormalize the tensor with given mean and standard deviation
            * Notice: If you didn't give mean and std, the function will assume that the original distribution locates in [-1, 1]

            Args:
                mean    - The mean of the result tensor
                std     - The standard deviation
        """
        self.mean = mean
        self.std = std

    def work(self, tensor):
        """
            Un-normalize the tensor

            Arg:    tensor  - The np.ndarray object. The tensor you want to deal with
            Ret:    The un-normalized tensor
        """
        if tensor.shape[0] != len(self.mean):
            raise Exception("The rank format should be BCHW, but the shape is {}".format(tensor.shape))
        result = []
        for t, m, s in zip(tensor, self.mean, self.std):
            result.append(t * s + m)
        tensor = np.asarray(result)
        return tensor


class ToGray(OP):
    def __init__(self):
        pass

    def work(self, tensor):
        """
            Make the tensor into gray-scale

            Arg:    tensor  - The np.ndarray object. The tensor you want to deal with
            Ret:    The gray-scale tensor, and the rank of the tensor is B1HW
        """
        if tensor.shape[0] == 3:
            result = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
            result = np.expand_dims(result, axis = 0)
        elif tensor.shape[0] != 4:
            result = 0.299 * tensor[:, 0] + 0.587 * tensor[:, 1] + 0.114 * tensor[:, 2]
            result = np.expand_dims(result, axis = 1)
        else:
            raise Exception("The rank format should be BCHW, but the shape is {}".format(tensor.shape))
        return result


class Transpose():
    def __init__(self, direction = BHWC2BCHW):
        """
            Transfer the rank of tensor into target one

            Arg:    direction   - The direction you want to do the transpose
        """
        self.direction = direction

    def __call__(self, tensor):
        """
            Arg:    tensor - The torch.Tensor object. The tensor you want to deal with
        """
        if self.direction == BHWC2BCHW:
            tensor = tensor.transpose(-1, -2).transpose(-2, -3)
        else:
            tensor = tensor.transpose(-3, -2).transpose(-2, -1)
        return tensor


class ToTensor():
    def __init__(self):
        pass

    def __call__(self, tensor):
        """
            Arg:    tensor - The torch.Tensor or other type. The tensor you want to deal with
        """
        if type(tensor) == np.ndarray:
            tensor = torch.from_numpy(tensor)
        return tensor

class ToFloat():
    def __init__(self):
        pass

    def __call__(self, tensor):
        """
            Arg:    tensor - The torch.Tensor object. The tensor you want to deal with
        """
        return tensor.float()