'''
@author: JJZHK
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: dataset.py
@time: 2019-04-01 15:44
@desc: 
'''
import ELib.utils.basicdataset as eub
import torch.utils.data as data
from PIL import Image
import random
import collections
import glob
import numpy as np
import os
import math


class FashionMnistPytorchData(data.Dataset):
    def __init__(self, root=r'/data/input/fashionmnist', train=True, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        reader = eub.FashionMnistDataSet(root=self.root)

        (self.train_data, self.train_labels),(self.test_data, self.test_labels) = reader.read()
    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img.reshape(28, 28), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


class MNISTDataSetForPytorch(data.Dataset):
    def __init__(self, root="/data/input/mnist.npz", train=True, radio=0.9, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        reader = eub.MnistDataSet(root=self.root, radio=radio)

        (self.train_data, self.train_labels),(self.test_data, self.test_labels) = reader.read()

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


class Cifar10DataSetForPytorch(data.Dataset):
    def __init__(self, root="/data/input/cifar10/", train=True,transform=None, target_transform=None, target_label=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        reader = eub.Cifar10DataSet(self.root, special_label=target_label)
        (self.train_data, self.train_label), (self.test_data, self.test_label) = reader.read(channel_first=False)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_label[index]
        else:
            img, target = self.test_data[index], self.test_label[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


class BaseDataSet(data.Dataset):
    def __init__(self):
        self.save_file = False
        self.files = None
        self.split_files = None

    def generateIndexList(self, a, size):
        """
            Generate the list of index which will be picked
            This function will be used as train-test-split

            Arg:    a       - The list of images
                    size    - Int, the length of list you want to create
            Ret:    The index list
        """
        result = set()
        while len(result) != size:
            result.add(random.randint(0, len(a) - 1))
        return list(result)


class ImageDataSet(BaseDataSet):
    def __init__(self, root = None, file_name = '.remain.pkl', sample_method = 0, transform = None,
                 split_ratio = 0.0, save_file = False):
        """
            The constructor of ImageDataset

            Arg:    root            - The list object. The image set
                    file_name       - The str. The name of record file.
                    sample_method   - sunnerData.UNDER_SAMPLING or sunnerData.OVER_SAMPLING. Use down sampling or over sampling to deal with data unbalance problem.
                                      (default is sunnerData.OVER_SAMPLING)
                    transform       - transform.Compose object. You can declare some pre-process toward the image
                    split_ratio     - Float. The proportion to split the data. Usually used to split the testing data
                    save_file       - Bool. If storing the record file or not. Default is False
        """
        super().__init__()
        # Record the parameter
        self.root = root
        self.file_name = file_name
        self.sample_method = sample_method
        self.transform = transform
        self.split_ratio = split_ratio
        self.save_file = save_file
        self.img_num = -1

        self.getFiles()

        # Change root obj as the index format
        self.root = range(len(self.root))

        # Adjust the image number
        self.getImgNum()

        # Split the files if split_ratio is more than 0.0
        self.split()

        # Print the domain information
        self.print()

    def getFiles(self):
        """
            Construct the files object for the assigned root
            We accept the user to mix folder with image
            This function can extract whole image in the folder
            The element in the files will all become image

            *******************************************************
            * This function only work if the files object is None *
            *******************************************************
        """
        if not self.files:
            self.files = {}
            for domain_idx, domain in enumerate(self.root):
                images = []
                for img in domain:
                    if os.path.exists(img):
                        if os.path.isdir(img):
                            images += self.readContain(img)
                        else:
                            images.append(img)
                    else:
                        raise Exception("The path {} is not exist".format(img))
                self.files[domain_idx] = sorted(images)

    def readContain(self, folder_name):
        """
            Read the containing in the particular folder

            ==================================================================
            You should customize this function if your data is not considered
            ==================================================================

            Arg:    folder_name - The path of folder
            Ret:    The list of containing
        """
        # Check the common type in the folder
        common_type = collections.Counter()
        for name in os.listdir(folder_name):
            common_type[name.split('.')[-1]] += 1
        common_type = common_type.most_common()[0][0]

        # Deal with the type
        if common_type == 'jpg':
            name_list = glob.glob(os.path.join(folder_name, '*.jpg'))
        elif common_type == 'png':
            name_list = glob.glob(os.path.join(folder_name, '*.png'))
        elif common_type == 'mp4':
            name_list = glob.glob(os.path.join(folder_name, '*.mp4'))
        else:
            raise Exception("Unknown type {}, You should customize in read.py".format(common_type))
        return name_list

    def getImgNum(self):
        """
            Obtain the image number in the loader for the specific sample method
            The function will check if the folder has been extracted
        """
        if self.img_num == -1:
            # Check if the folder has been extracted
            for domain in self.root:
                for img in self.files[domain]:
                    if os.path.isdir(img):
                        raise Exception("You should extend the image in the folder {} first!" % img)

            # Statistic the image number
            for domain in self.root:
                if domain == 0:
                    self.img_num = len(self.files[domain])
                else:
                    if self.sample_method == 1:
                        self.img_num = max(self.img_num, len(self.files[domain]))
                    elif self.sample_method == 0:
                        self.img_num = min(self.img_num, len(self.files[domain]))
        return self.img_num

    def split(self):
        """
            Split the files object into split_files object
            The original files object will shrink

            We also consider the case of pair image
            Thus we will check if the number of image in each domain is the same
            If it does, then we only generate the list once
        """
        # Check if the number of image in different domain is the same
        if not self.files:
            self.getFiles()
        pairImage = True
        for domain in range(len(self.root) - 1):
            if len(self.files[domain]) != len(self.files[domain + 1]):
                pairImage = False

        # Split the files
        self.split_files = {}
        if pairImage:
            split_img_num = math.floor(len(self.files[0]) * self.split_ratio)
            choice_index_list = self.generateIndexList(range(len(self.files[0])), size = split_img_num)
        for domain in range(len(self.root)):
            # determine the index list
            if not pairImage:
                split_img_num = math.floor(len(self.files[domain]) * self.split_ratio)
                choice_index_list = self.generateIndexList(range(len(self.files[domain])), size = split_img_num)
            # remove the corresponding term and add into new list
            split_img_list = []
            remain_img_list = self.files[domain].copy()
            for j in choice_index_list:
                split_img_list.append(self.files[domain][j])
            for j in choice_index_list:
                self.files[domain].remove(remain_img_list[j])
            self.split_files[domain] = sorted(split_img_list)

    def print(self):
        pass

    def readItem(self, item_name):
        """
            Read the file for the given item name

            ==================================================================
            You should customize this function if your data is not considered
            ==================================================================

            Arg:    item_name   - The path of the file
            Ret:    The item you read
        """
        file_type = item_name.split('.')[-1]
        if file_type == "png" or file_type == 'jpg':
            file_obj = np.asarray(Image.open(item_name))

            if len(file_obj.shape) == 3:
                # Ignore the 4th dim (RGB only)
                file_obj = file_obj[:, :, :3]
            elif len(file_obj.shape) == 2:
                # Make the rank of gray-scale image as 3
                file_obj = np.expand_dims(file_obj, axis = -1)
        return file_obj

    def __len__(self):
        return self.img_num

    def __getitem__(self, index):
        return_list = []
        for domain in self.root:
            img_path = self.files[domain][index]
            img = self.readItem(img_path)
            if self.transform:
                img = self.transform(img)
            return_list.append(img)
        return return_list