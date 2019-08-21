'''
@author: JJZHK
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: Config.py
@time: 2018/12/11 10:18
@desc: 
'''
import os
import yaml
import json


class AttrDict:
    def __init__(self, content):
        self._content = content
        self._mysetattr_()

    def _mysetattr_(self):
        self._setattr_(self, self._content)

    def _setattr_(self,obj, content):
        for item in content:
            if isinstance(content[item], dict):
                testobj = AttrDict(content[item])
                obj.__dict__[item] = testobj
                self._setattr_(testobj, content[item])
            elif isinstance(content[item], list):
                if len(content[item]) > 0 and isinstance(content[item][0], dict):
                    obj.__dict__[item] = [AttrDict(x) for x in content[item]]
                else:
                    obj.__dict__[item] = content[item]
            else:
                obj.__dict__[item] = content[item]

    def merge(self, model):
        self._merge(self, model)

    def _merge(self, target, source):
        source_attr = source.__dict__

        for key in source_attr.keys():
            if key == "_content":
                continue
            if isinstance(source_attr[key], AttrDict):
                if hasattr(target, key):
                    self._merge(target.__dict__[key], source.__dict__[key])
                else:
                    target.__dict__[key] = source.__dict__[key]
            else:
                target.__dict__[key] = source.__dict__[key]


class Config:
    def __init__(self, cfgfile, commonfile=None, cfgroot="cfgs/", loadtype='f'):
        '''
        :param commonfile: 公共部分的文件地址或字符串
        :param cfgfile: 需要加载的yml文件名或字符串
        :param cfgroot: yml目录地址
        :param loadtype: 加载方式：f-yml文件加载，s-字符串加载
        '''
        self.commonfile = commonfile
        self.loadtype = loadtype
        self.cfgroot = cfgroot
        self.CFGData = AttrDict({})
        if self.loadtype == 'f':
            with open(os.path.join(cfgroot, cfgfile), 'r') as file:
                data = yaml.load(file, Loader=yaml.FullLoader)
        elif self.loadtype == 's':
            with open(os.path.join(cfgfile), 'r') as file:
                data = yaml.load(file,Loader=yaml.FullLoader)
        else:
            raise Exception("Load type is error.")

        data = json.loads(json.dumps(data))

        if self.commonfile is not None:
            self._initcfg_()

        self._mergecfg_(self.CFGData, AttrDict(data))

        self._updatecfg_()

    def _initcfg_(self):
        if self.loadtype == 'f':
            with open(os.path.join(self.cfgroot, self.commonfile), 'r') as file:
                data = yaml.load(file, Loader=yaml.FullLoader)
        elif self.loadtype == 's':
            data = yaml.load(self.commonfile, Loader=yaml.FullLoader)
        else:
            raise Exception("Load type is error.")

        data = json.loads(json.dumps(data))
        self.CFGData = AttrDict(data)
        self._initspecial_()

    def _mergecfg_(self,target, data):
        target.merge(data)

    def _updatecfg_(self):
        pass

    def _initspecial_(self):
        pass


class DinosaurConfig(Config):
    def __init__(self,  cfgfile, commonfile=None, cfgroot="cfgs/", loadtype='f'):
        super(DinosaurConfig, self).__init__( cfgfile,commonfile, cfgroot, loadtype)

    def _initspecial_(self):
        pass

    def _updatecfg_(self):
        pass


class DataConfig:
    def __init__(self, cfg):
        self.cfg = cfg
        self.colors = self.cfg.BASE.COLORS
        self.classes = self.cfg.BASE.CLASSNAMES
        self.classes_num = len(self.classes)
        self.dataset = None
        self.name = self.cfg.BASE.DATA_TYPE

    def getClasses(self):
        return self.classes

    def getColors(self):
        return self.colors

    def getNumByClass(self, cls):
        classes = self.getClasses()
        return classes.index(cls)

    def getClassByNum(self, num):
        classes = self.getClasses()
        return classes[num]

