'''
@author: JJZHK
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: util.py
@time: 2018/11/21 17:20
@desc: 
'''
import numpy as np
import os
import json


class Functional:
    @staticmethod
    def clamp(arr, min=0, max=9999):
        if isinstance(arr, np.int) or isinstance(arr, np.float32):
            return max(min, min(arr, max))

        arr_1 = np.reshape(arr, newshape=(1, -1))
        arr_1 = np.maximum(min, np.minimum(arr_1, max))
        return np.reshape(arr_1, newshape=arr.shape)

class EvalUtil:
    @staticmethod
    def save_detection_boxes_tofile_forVOC(preds, classes, outputs):
        if not os.path.exists(outputs):
            os.mkdir(outputs)

        allclasses = classes
        for classname in allclasses:
            boxes = preds[classname]
            if os.path.exists(os.path.join(outputs, "%s.txt" % classname)):
                os.remove(os.path.join(outputs, "%s.txt" % classname))

            with open(os.path.join(outputs, "%s.txt" % classname), "wt") as file:
                for box in boxes:
                    file.write("%s %f %f %f %f %f\n" % (box[0], box[1], box[2],box[3],box[4],box[5]))
                file.flush()
                file.close()

    @staticmethod
    def save_detection_boxes_tofile_forCOCO(preds, outputs):
        if not os.path.exists(outputs):
            os.mkdir(outputs)

        if os.path.exists(os.path.join(outputs, "%s.json" % "cocoevaltest")):
            os.remove(os.path.join(outputs, "%s.json" % "cocoevaltest"))

        with open(os.path.join(outputs, "%s.json" % "cocoevaltest"), "wt") as file:
            json.dump(preds, file)

    @staticmethod
    def cal_mAP(ground_truth, pred_file_folder, classes, ovthresh=0.5, use_07_metric=True):
        if not os.path.exists(pred_file_folder):
            raise Exception("Predict folder does not exist")

        aps = []
        info = {}

        for i, cls in enumerate(classes):
            filename = os.path.join(pred_file_folder, "%s.txt" % cls)
            rec, prec, ap = EvalUtil.voc_eval(ground_truth, filename, cls,ovthresh=ovthresh, use_07_metric=use_07_metric)
            aps += [ap]
            info[cls] = ap

        return np.mean(aps), info

    @staticmethod
    def voc_eval(ground_truth, filename, classname, ovthresh, use_07_metric):
        imagenames = ground_truth.keys()
        class_recs = {}
        npos = 0
        for imagename in imagenames:
            R = [obj for obj in ground_truth[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}

        with open(filename, 'r') as f:
            lines = f.readlines()
        if any(lines) == 1:

            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
            confidence = np.array([float(x[1]) for x in splitlines])
            BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d in range(nd):
                R = class_recs[image_ids[d]]
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)
                if BBGT.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin, 0.)
                    ih = np.maximum(iymax - iymin, 0.)
                    inters = iw * ih
                    uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                           (BBGT[:, 2] - BBGT[:, 0]) *
                           (BBGT[:, 3] - BBGT[:, 1]) - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax]:
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.

            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap =EvalUtil.voc_ap(rec, prec, use_07_metric)
        else:
            rec = -1.
            prec = -1.
            ap = -1.

        return rec, prec, ap

    @staticmethod
    def voc_ap(rec, prec, use_07_metric=True):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:True).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap