'''
@author: JJZHK
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: aaa.py
@time: 2018/10/25 14:11
@desc:
'''
import numpy as np
import PIL.ImageDraw as pil_imagedraw
import PIL.ImageFont as pil_imagefont
import ELib.utils.imageutil as eui
import ELib.utils.config as euc
import cv2
import os

'''
image : PIL.Image类型对象
'''
def draw_box(image, boxes, outputs, imagename, cfg, mask=None, maskname=None):
    DATA_CONFIG = euc.DataConfig(cfg=cfg)

    COLOR = DATA_CONFIG.getColors()
    CLASS = DATA_CONFIG.getClasses()

    font = pil_imagefont.truetype(font='Courier.dfont',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300
    if boxes is not None:
        for (x1, y1), (x2,y2), class_name, _, prob in boxes:
            draw = pil_imagedraw.Draw(image)
            left = int(x1)
            top = int(y1)
            right = int(x2)
            bottom = int(y2)
            color = tuple(COLOR[CLASS.index(class_name)])

            label = '{} {:.2f}'.format(class_name, prob)
            label_size = draw.textsize(label, font)

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i], outline=color)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=color)
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

    if mask is not None:
        piltocv2 = eui.PILToCV2()
        image = piltocv2(image)
        mask = cv2.cvtColor(maskToImg(mask, CLASS, COLOR), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(outputs, maskname), mask)
        image = cv2.addWeighted(image, 0.6, mask, 0.4, 0)
        cv2topil = eui.CV2ToPIL()
        image = cv2topil(image)

    image.save(os.path.join(outputs, imagename))

    return image

def draw_box_by_cv2(image, boxes, outputs, imagename, cfg, mask=None, maskname=None):
    cv2topil = eui.CV2ToPIL()
    img = cv2topil(image)
    img = draw_box(img, boxes, outputs, imagename, cfg, mask, maskname)
    piltocv2 = eui.PILToCV2()
    return piltocv2(img)

def maskToImg(mask, CLASS, COLOR):
    r = mask.copy()
    g = mask.copy()
    b = mask.copy()
    for ll in range(0, len(CLASS)):
        r[mask == ll] = COLOR[ll][0]
        g[mask == ll] = COLOR[ll][1]
        b[mask == ll] = COLOR[ll][2]
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    return rgb