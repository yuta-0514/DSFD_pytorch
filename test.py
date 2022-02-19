import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import cv2
import time
import numpy as np
from PIL import Image

from models.factory import build_net
from torch.autograd import Variable
from utils.augmentations import to_chw_bgr


parser = argparse.ArgumentParser(description='dsfd demo')
parser.add_argument('--network',
                    default='vgg', type=str,
                    choices=['vgg', 'resnet50', 'resnet101', 'resnet152'],
                    help='model for training')
parser.add_argument('--weights',
                    type=str,
                    default='weights/dsfd_face.pth', help='trained model')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--conf', type=float, default=0.95)
parser.add_argument('--iou', type=float, default=0.5)
args = parser.parse_args()


IMAGE_DIR = '/mnt/MAFA/images/'


class MakeDataset():
    def __init__(self):
        self.split_file = '/mnt/MAFA/test_anno.txt'
        self.image_dir = IMAGE_DIR
        self.data_dict = dict()

        with open(self.split_file, 'r') as fp:
            lines = [line.rstrip('\n') for line in fp]

            i = 0
            while i < len(lines):
                print('%6d / %6d' % (i, len(lines)))
                img_name = lines[i]
                num_face = int(lines[i + 1])

                if num_face != 0:
                    rect_list = list()
                    for j in range(num_face):
                        r = [float(x) for x in lines[i + 2 + j].split()[0:4]]
                        rect = [r[0], r[1], r[0] + r[2], r[1] + r[3]]
                        rect_list.append(rect)
                    self.data_dict[img_name] = rect_list
                    i = i + num_face + 2
                else:
                    i = i + 1 + 2


def IoU(boxA, boxB):
    area_A = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    area_B = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    xx1 = np.maximum(boxA[0], boxB[0])
    yy1 = np.maximum(boxA[1], boxB[1])
    xx2 = np.minimum(boxA[2], boxB[2])
    yy2 = np.minimum(boxA[3], boxB[3])
    w_inter = np.maximum(0, xx2 - xx1 + 1)
    h_inter = np.maximum(0, yy2 - yy1 + 1)
    area_inter = w_inter * h_inter

    return area_inter / (area_A + area_B - area_inter)


def nms_(dets, thresh):
    """
    Courtesy of Ross Girshick
    [https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py]
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep).astype(np.int)


def detect(net, image, conf_th):
    w, h = image.shape[1], image.shape[0]

    bboxes = np.empty(shape=(0, 5))

    with torch.no_grad():
        scaled_img = cv2.resize(image, dsize=(0, 0), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
        scaled_img = np.swapaxes(scaled_img, 1, 2)
        scaled_img = np.swapaxes(scaled_img, 1, 0)
        scaled_img = scaled_img[[2, 1, 0], :, :]
        scaled_img = scaled_img.astype('float32')
        scaled_img -= np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype('float32')
        scaled_img = scaled_img[[2, 1, 0], :, :]
        x = torch.from_numpy(scaled_img).unsqueeze(0).to("cuda")
        y = net(x)

        detections = y.data
        scale = torch.Tensor([w, h, w, h])

        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] > conf_th:
                score = detections[0, i, j, 0]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                bbox = (pt[0], pt[1], pt[2], pt[3], score)
                bboxes = np.vstack((bboxes, bbox))
                j += 1

    keep = nms_(bboxes, 0.1)
    bboxes = bboxes[keep]

    return bboxes


def main():
    device = args.device
    conf_threshold = args.conf
    iou_threshold = args.iou

    WD = MakeDataset()

    net = build_net('test', 2, args.network)
    net.to("cuda")
    net.load_state_dict(torch.load(args.weights, map_location="cuda"))
    net.eval()

    scale_list = [0.5, 1]

    N = len(WD.data_dict.keys())
    total_iou = 0.0
    total_recall = 0.0
    total_precision = 0.0
    total_f1score = 0.0
    total_time = 0.0

    for image_index, image_name in enumerate(WD.data_dict.keys(), 1):
        print('%5d / %5d : %s' % (image_index, N, image_name))
        image = cv2.imread(os.path.join(IMAGE_DIR, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = WD.data_dict[image_name]

        img_time = time.time()
        print(image_name)
        pred_boxes = detect(net, image, args.conf)
        img_time = time.time() - img_time
        
        true_num = len(boxes)
        positive_num = len(pred_boxes)
        img_iou = 0.0
        img_recall = 0.0
        img_precision = 0.0
        img_f1score = 0.0

        pred_dict = dict()

        for box in boxes:
            max_iou = 0
            for i, pred_box in enumerate(pred_boxes):
                if i not in pred_dict.keys():
                    pred_dict[i] = 0
                iou = IoU(box, pred_box)
                if iou > max_iou:
                    max_iou = iou
                if iou > pred_dict[i]:
                    pred_dict[i] = iou
            img_iou += max_iou
        
        if true_num * positive_num > 0:
            true_positive = 0.0
            for i in pred_dict.keys():
                if pred_dict[i] > iou_threshold:
                    true_positive += 1.0
            img_recall = true_positive / true_num
            img_precision = true_positive / positive_num
            if img_recall * img_precision == 0:
                img_f1score = 0.0
            else:
                img_f1score = (2*img_recall*img_precision) / (img_recall+img_precision)
            img_iou = img_iou / true_num
        
            print('- | TP = %02d | TN =    |' % (true_positive))
            print('  | FP = %02d | FN = %02d |' % (positive_num - true_positive, true_num - true_positive))

        total_iou += img_iou
        total_recall += img_recall
        total_precision += img_precision
        total_f1score += img_f1score
        total_time += img_time

        print('- Avg.            IoU =', total_iou / image_index)
        print('- Avg.         Recall =', total_recall / image_index)
        print('- Avg.      Precision =', total_precision / image_index)
        print('- Avg.       F1-score =', total_f1score / image_index)
        print('- Avg. Inference Time =', total_time / image_index)

        torch.cuda.empty_cache() #GPUのメモリ不足を防ぐ


if __name__ == '__main__':
    main()