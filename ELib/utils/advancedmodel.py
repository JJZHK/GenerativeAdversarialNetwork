'''
@author: JJZHK
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: basicmodel.py
@time: 2018/12/17 14:46
@desc: 
'''
class BasicModel:
    def __init__(self):
        pass

    def __call__(self, **kwargs):
        return self.call(**kwargs)

    def call(self,**kwargs):
        pass

    def getBoxes(self,image, info, eval=False, **kwargs):
        '''
        :param kwargs:
        image - imagelist
        info - information list
        :return:
        '''
        pass