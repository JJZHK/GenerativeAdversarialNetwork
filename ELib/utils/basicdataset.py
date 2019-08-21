# coding=utf-8
import numpy as np
import pickle
import random
import os
import xml.etree.ElementTree as ET
import cv2
import ELib.utils.config as cfg
import tqdm
import pycocotools.coco as coco
import zipfile
import gzip
import PIL.Image as Image


class BasicDataSet(object):
    def __init__(self, root, train_ratio=1):
        self._root_path = root
        self.ratio=train_ratio

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def read(self, onehot=False, channel_first=True):
        (x_train, targets_train), (x_test, targets_test) = self._readData(channel_first)

        x_total = np.concatenate((x_train, x_test))
        y_total = np.concatenate((targets_train, targets_test))

        index_list = list(range(0, x_total.shape[0]))
        random.shuffle(index_list)

        train_record_count = int(len(index_list) * self.ratio) if self.ratio > 0 else len(x_train)

        index_train = index_list[0:train_record_count]
        index_test  = index_list[train_record_count:len(index_list)]

        x_train = x_total[index_train]
        x_test = x_total[index_test]
        targets_train = y_total[index_train]
        targets_test = y_total[index_test]

        self.TRAIN_RECORDS = x_train.shape[0]
        self.TEST_RECORDS = x_test.shape[0]

        if onehot:
            y_train = np.zeros((targets_train.shape[0], 10), dtype = np.uint8)
            y_test = np.zeros((targets_test.shape[0], 10), dtype = np.uint8)
            y_train[np.arange(targets_train.shape[0]), targets_train] = 1
            y_test[np.arange(targets_test.shape[0]), targets_test] = 1

            return (x_train, y_train), (x_test, y_test)
        else:
            return (x_train, np.reshape(targets_train, newshape=(targets_train.shape[0], 1))), (x_test, np.reshape(targets_test, newshape=(targets_test.shape[0], 1)))


    def _readData(self, channel_first):
        pass


class Cifar10DataSet(BasicDataSet):
    NUM_OUTPUTS = 10
    IMAGE_CHANNEL = 3
    IMAGE_SIZE = 32
    TRAIN_RECORDS = 50000
    TEST_RECORDS = 10000
    LABELS = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

    def __init__(self, root="/data/input/cifar10/", train_ratio=0.9, special_label=None):
        super(Cifar10DataSet, self).__init__(root=root, train_ratio=train_ratio)
        self.special_label = special_label

    def _readData(self, channel_first):
        data_train = []
        targets_train = []
        for i in range(5):
            raw = self.unpickle(os.path.join(self._root_path, 'data_batch_' + str(i + 1)))
            data_train += [raw[b'data']]
            targets_train += raw[b'labels']

        data_train = np.concatenate(data_train)
        targets_train = np.array(targets_train)

        data_test = self.unpickle(os.path.join(self._root_path,'test_batch'))
        targets_test = np.array(data_test[b'labels'])
        data_test = np.array(data_test[b'data'])

        x_train = np.reshape(data_train, (-1, self.IMAGE_CHANNEL, self.IMAGE_SIZE, self.IMAGE_SIZE))
        x_test  = np.reshape(data_test,  (-1, self.IMAGE_CHANNEL, self.IMAGE_SIZE, self.IMAGE_SIZE))

        if channel_first == False:
            x_train = x_train.transpose(0, 2, 3, 1)
            x_test = x_test.transpose(0, 2, 3, 1)

        if self.special_label is not None:
            x_train = x_train[np.where(targets_train==self.special_label)[0]]
            targets_train = targets_train[targets_train[:]==self.special_label]
            x_test = x_test[np.where(targets_test==self.special_label)[0]]
            targets_test = targets_test[targets_test[:]==self.special_label]

        return (x_train, targets_train), (x_test, targets_test)


class MnistDataSet(BasicDataSet):
    NUM_OUTPUTS = 10
    IMAGE_SIZE = 28
    IMAGE_CHANNEL = 1

    def __init__(self, root="/data/input/mnist.npz", radio=1):
        super(MnistDataSet, self).__init__(root=root, train_ratio=radio)

    def _readData(self, channel_first):
        f = np.load(self._root_path)
        x_train, targets_train = f['x_train'], f['y_train']
        x_test, targets_test = f['x_test'], f['y_test']
        f.close()

        return (x_train, targets_train), (x_test, targets_test)


class FashionMnistDataSet(BasicDataSet):
    NUM_OUTPUTS = 10
    IMAGE_SIZE = 28
    IMAGE_CHANNEL = 1

    def __init__(self, root="/data/input/fashionmnist"):
        super(FashionMnistDataSet, self).__init__(root=root, train_ratio=0)

    def _readData(self, channel_first):
        with gzip.open(os.path.join(self._root_path, 'train-labels-idx1-ubyte.gz')) as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(os.path.join(self._root_path, 'train-images-idx3-ubyte.gz')) as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 1, 28, 28)

        with gzip.open(os.path.join(self._root_path, 't10k-labels-idx1-ubyte.gz')) as tlbpath:
            tlabels = np.frombuffer(tlbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(os.path.join(self._root_path, 't10k-images-idx3-ubyte.gz')) as timgpath:
            timages = np.frombuffer(timgpath.read(), dtype=np.uint8, offset=16).reshape(len(tlabels), 1, 28, 28)

        return (images, labels), (timages, tlabels)


class STLDataSet(BasicDataSet):
    NUM_OUTPUTS = 10
    IMAGE_CHANNEL = 3
    IMAGE_SIZE = 96
    TRAIN_RECORDS = 5000
    TEST_RECORDS = 8000
    LABELS = ["airplane","bird","car","cat","deer","dog","horse","monkey","ship","truck"]

    def __init__(self, root="/data/input/STL10/"):
        super(STLDataSet, self).__init__(root=root, train_ratio=0.95)
        self.train_image_file = os.path.join(self._root_path, "train_X.bin")
        self.train_label_file = os.path.join(self._root_path, "train_y.bin")
        self.test_image_file = os.path.join(self._root_path, "test_X.bin")
        self.test_label_file = os.path.join(self._root_path, "test_y.bin")

    def _readData(self, channel_first):

        with open(self.train_image_file, 'rb') as f:
            everything = np.fromfile(f, dtype=np.uint8)
            x_train = np.reshape(everything, (-1, self.IMAGE_CHANNEL, self.IMAGE_SIZE, self.IMAGE_SIZE))

        with open(self.train_label_file, 'rb') as f:
            targets_train = np.fromfile(f, dtype=np.uint8)

        with open(self.test_image_file, 'rb') as f:
            everything = np.fromfile(f, dtype=np.uint8)
            x_test = np.reshape(everything, (-1, self.IMAGE_CHANNEL, self.IMAGE_SIZE, self.IMAGE_SIZE))

        with open(self.test_label_file, 'rb') as f:
            targets_test = np.fromfile(f, dtype=np.uint8)

        targets_train = targets_train-1
        targets_test = targets_test - 1
        if channel_first == False:
            x_train = x_train.transpose(0, 2, 3, 1)
            x_test = x_test.transpose(0, 2, 3, 1)

        return (x_train, targets_train), (x_test, targets_test)


class DataSetBase(object):
    def __init__(self, root,imageset, isSegmentation):
        self._root_path = root
        self._imageset = imageset
        self._img_list = []
        self._isSegmentation = isSegmentation
        self.__prepare__()

    def __len__(self):
        return len(self._img_list)

    def __getitem__(self, index):
        image = self.getImage(index)
        label = self.getLabel(index)
        if self._isSegmentation:
            mask = self.getMask(index)
            return image, label, mask

        return image, label

    def __prepare__(self):
        self.prepare()

    def __getItemInfo__(self, index):

        return self._img_list[index]

    def __getItemInfoByImageId__(self, imgid):
        pass

    def __getIndexByImageId__(self, imgid):
        index = 0
        for item in self._img_list:
            if item["img_id"] == imgid:
                return index

            index += 1

        return -1
    def prepare(self):
        pass

    def getImage(self, index):
        pass

    def getLabel(self, index):
        pass

    def getMask(self, index):
        pass


class VOCDataSet(DataSetBase):
    def __init__(self, config, imageset, isSegmentation):
        root = config.BASE.DATA_ROOT
        self.dataConfig = cfg.DataConfig(config)
        self.COLOR = self.dataConfig.getColors()
        super(VOCDataSet, self).__init__(root=root, imageset=imageset, isSegmentation=isSegmentation)

    def prepare(self):
        prefix = "seg" if self._isSegmentation else "det"
        image_list_file = os.path.join(self._root_path, "MainSet", "%s_%s.txt" % (prefix, self._imageset))
        lines = [x.strip() for x in open(image_list_file, 'r').readlines()]

        bar = tqdm.tqdm(lines)
        for line in bar:
            bar.set_description("Processing %s" % self._imageset)
            l = self.__getItemInfoByImageId__(line)
            self._img_list.append(l)

    def getImage(self, index):
        fname = os.path.join(self._root_path, "JPEGImages", "%s.jpg" % self._img_list[index]["img_id"])
        self._img_list[index]["path"] = fname
        return cv2.imread(fname)

    def getLabel(self, index):
        return self._img_list[index]["boxes"]

    def getMask(self, index):
        info = self._img_list[index]
        file_index = info["img_id"]
        lbl_path = os.path.join(self._root_path, 'SegmentationDecode', file_index + '.png')
        label_mask = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)

        return label_mask

    '''
    {
        img_id : "***",
        width : "***",
        height : "***",
        boxes : [[classs_id, xmin, ymin, xmax, ymax],[classs_id, xmin, ymin, xmax, ymax]
        detail : [{"name":***,"pose":***, "truncated":***,"difficult":***,},{....},{....}]
        ...]
    }
    '''
    def __getItemInfoByImageId__(self, image_id):
        image_info = {}
        image_info["img_id"] = image_id
        labels = []
        detail = []
        anno_file = os.path.join(self._root_path, "Annotations", "%s.xml" % image_id)
        root = ET.parse(anno_file).getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        image_info["width"] = w
        image_info["height"] = h

        for obj in root.iter('object'):
            class_name = obj.find('name').text
            xmlbox = obj.find('bndbox')
            left   = int(xmlbox.find('xmin').text)
            right  = int(xmlbox.find('xmax').text)
            top    = int(xmlbox.find('ymin').text)
            bottom = int(xmlbox.find('ymax').text)
            class_id = self.dataConfig.getNumByClass(class_name)
            item = [class_id, left, top, right, bottom]
            labels.append(item)

            pose = obj.find("pose").text
            truncated = int(obj.find("truncated").text)
            difficult = int(obj.find("difficult").text)
            detail.append({"name": class_name, "pose":pose, "truncated" : truncated, "difficult" : difficult})

        image_info["boxes"] = labels
        image_info["detail"] = detail

        return image_info


class FaPiaoDataSet(DataSetBase):
    def __init__(self, config, imageset, isSegmentation):
        root = config.BASE.DATA_ROOT
        self.dataConfig = cfg.DataConfig(config)
        self.COLOR = self.dataConfig.getColors()
        super(FaPiaoDataSet, self).__init__(root=root, imageset=imageset, isSegmentation=isSegmentation)

    def prepare(self):
        prefix = "seg" if self._isSegmentation else "det"
        image_list_file = os.path.join(self._root_path, "MainSet", "%s_%s.txt" % (prefix, self._imageset))
        lines = [x.strip() for x in open(image_list_file, 'r').readlines()]
        bar = tqdm.tqdm(lines)
        for line in bar:
            bar.set_description("Processing %s" % self._imageset)
            l = self.__getItemInfoByImageId__(line)
            self._img_list.append(l)

    def getImage(self, index):
        fname = os.path.join(self._root_path, "JPEGImages", "%s.jpg" % self._img_list[index]["img_id"])
        self._img_list[index]["path"] = fname
        return cv2.imread(fname)

    def getLabel(self, index):
        return self._img_list[index]["boxes"]

    def getMask(self, index):
        info = self._img_list[index]
        file_index = info["img_id"]
        lbl_path = os.path.join(self._root_path, 'SegmentationDecode', file_index + '.png')
        label_mask = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)

        return label_mask

    '''
    {
        img_id : "***",
        width : "***",
        height : "***",
        boxes : [[classs_id, xmin, ymin, xmax, ymax],[classs_id, xmin, ymin, xmax, ymax]
        detail : [{"name":***,"pose":***, "truncated":***,"difficult":***,},{....},{....}]
        ...]
    }
    '''
    def __getItemInfoByImageId__(self, image_id):
        image_info = {}
        image_info["img_id"] = image_id
        labels = []
        detail = []
        anno_file = os.path.join(self._root_path, "Annotations", "%s.xml" % image_id)
        root = ET.parse(anno_file).getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        image_info["width"] = w
        image_info["height"] = h

        for obj in root.iter('object'):
            class_name = obj.find('name').text
            xmlbox = obj.find('bndbox')
            left   = int(xmlbox.find('xmin').text)
            right  = int(xmlbox.find('xmax').text)
            top    = int(xmlbox.find('ymin').text)
            bottom = int(xmlbox.find('ymax').text)
            class_id = self.dataConfig.getNumByClass(class_name)
            item = [class_id, left, top, right, bottom]
            labels.append(item)

            pose = obj.find("pose").text
            truncated = int(obj.find("truncated").text)
            difficult = int(obj.find("difficult").text)
            detail.append({"name": class_name, "pose":pose, "truncated" : truncated, "difficult" : difficult})

        image_info["boxes"] = labels
        image_info["detail"] = detail

        return image_info


class COCODataSet(DataSetBase):
    def __init__(self, config, imageset, isSegmentation):
        self.dataConfig = cfg.DataConfig(config)
        self.COLOR = self.dataConfig.getColors()
        self.year = 2017
        super(COCODataSet, self).__init__(root=config.BASE.DATA_ROOT, imageset=imageset, isSegmentation=isSegmentation)
        prefix = "%s%d" % (self._imageset, self.year)
        self.zip = zipfile.ZipFile(os.path.join(self._root_path, "%s.zip" % (prefix) ))

    def prepare(self):
        prefix = "%s%d" % (self._imageset, self.year)
        annFile = os.path.join(self._root_path, "annotations", "instances_%s.json" % prefix)
        print(annFile)
        self.coco = coco.COCO(annFile)
        class_ids = sorted(self.coco.getCatIds()) # all classes

        lines = []
        for id in class_ids:
            lines.extend(list(self.coco.getImgIds(catIds=[id])))
        # Remove duplicates
        lines = list(set(lines))

        bar = tqdm.tqdm(lines)
        for line in bar:
            bar.set_description("Processing %s" % self._imageset)
            l = self.__getItemInfoByImageId__(line)
            self._img_list.append(l)

    def getImage(self, index):
        info = self._img_list[index]
        img_info = self.coco.imgs[info["img_id"]]
        self._img_list[index]["path"] = ""
        prefix = "%s%d" % (self._imageset, self.year)
        r = self.zip.read('%s/%s' % (prefix, img_info['file_name']))
        arr = cv2.imdecode(np.frombuffer(r, np.uint8), 1)

        return arr

    def getLabel(self, index):
        return self._img_list[index]["boxes"]

    def getMask(self, index):
        info = self._img_list[index]
        img_id = info['img_id']
        height = info['height']
        width = info['width']
        annIds = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        MASK = np.zeros((height, width), dtype=np.uint16)

        for i, ann in enumerate(anns):
            mask = self.coco.annToMask(ann)
            idxs = np.where(mask > 0)
            MASK[idxs] = ann['category_id']

        return MASK

    '''
    {
        img_id : "***",
        width : "***",
        height : "***",
        boxes : [[classs_id, xmin, ymin, xmax, ymax],[classs_id, xmin, ymin, xmax, ymax]
        ...]
    }
    '''
    def __getItemInfoByImageId__(self, image_id):
        image_info = self.coco.imgs[image_id]
        img_info_json = {}
        img_info_json["img_id"] = image_id
        img_info_json["width"] = image_info["width"]
        img_info_json["height"] = image_info["height"]
        class_coco = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
                      31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                      55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                      82, 84, 85, 86, 87, 88, 89, 90]
        annIds = self.coco.getAnnIds(imgIds=image_id, iscrowd=None)
        ann_info = self.coco.loadAnns(annIds)
        item = []
        for ann in ann_info:
            class_id = [class_coco.index(int(ann["category_id"]))]
            bbox = ann["bbox"]
            bbox[2] = bbox[0] + bbox[2]
            bbox[3] = bbox[1] + bbox[3]
            item.append(class_id + bbox)
        img_info_json["boxes"] = item
        return img_info_json


class TestImageDataSet(DataSetBase):
    def __init__(self, cfg, isSegmentation=False):
        self.cfg = cfg
        super(TestImageDataSet, self).__init__(root=self.cfg.BASE.TEST_DATA_ROOT, imageset='test', isSegmentation=isSegmentation)

    def prepare(self):
        image_list_file = os.path.join(self._root_path, "info.txt")
        lines = [x.strip() for x in open(image_list_file, 'r').readlines()]

        bar = tqdm.tqdm(lines)
        for line in bar:
            bar.set_description("Processing %s" % self._imageset)
            self._img_list.append({"img_id" : line})

    def getImage(self, index):
        fname = os.path.join(self._root_path, "Images", "%s.jpg" % self._img_list[index]["img_id"])
        image = cv2.imread(fname)
        self._img_list[index]["path"] = fname
        self._img_list[index]["height"] = image.shape[0]
        self._img_list[index]["width"] = image.shape[1]
        return image

    def getLabel(self, index):
        return None

    def getMask(self, index):
        return None


# if __name__ == '__main__':
#     import ELib.utils.drawutils as eud
#     import ELib.utils.config as euc
#
#     phase='voc'
#     config = euc.Config(cfgfile='test.yml', loadtype='s').CFGData
#
#     if phase == 'voc':
#         dataset = VOCDataSet(config=config,imageset='train', isSegmentation=True)
#         for i, (image, labels, mask) in enumerate(dataset):
#             boxes = []
#             for label in labels:
#                 class_name = euc.DataConfig(config).getClassByNum(int(label[0]))
#                 boxes.append([(label[1], label[2]),(label[3], label[4]), class_name, "", 1.00])
#             eud.draw_box_by_cv2(image, boxes, "", "test_%d_.jpg" % i, config, mask, "mask_%d.png" % i)
#             if i > 12:
#                 break
#     else:
#         dataset = COCODataSet(imageset='train', isSegmentation=True)
#         for i, (image, labels, mask) in enumerate(dataset):
#             boxes = []
#             for label in labels:
#                 class_name = euc.COCO_Data_Config().getClassByNum(int(label[0]))
#                 boxes.append([(label[1], label[2]),(label[3], label[4]), class_name, "", 1.00])
#             imagename = "test_%d_.jpg" % i
#             # cv2.imwrite("mask_%d.png" % i, mask)
#             eud.draw_box_by_cv2(image, boxes, "", "test_%d_.jpg" % i, config, mask, "mask_%d.png" % i)
#             if i > 12:
#                 break