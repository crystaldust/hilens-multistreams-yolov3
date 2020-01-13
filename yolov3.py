# -*- coding: utf-8 -*-
# !/usr/bin/python3

import cv2
import time
import math
import numpy as np
aipp_flag = True
net_h = 352
net_w = 640
input_height = 352
input_width = 640

class_names = ['apple', 'bus', 'train', 'bear', 'zebra', 'giraffe']
class_num   = len(class_names)

stride_list = [8, 16, 32]
anchors_1   = np.array([[10,13],  [16,30],   [33,23]]) / stride_list[0]
anchors_2   = np.array([[30,61],  [62,45],   [59,119]]) / stride_list[1]
anchors_3   = np.array([[116,90], [156,198], [163,326]]) / stride_list[2]
anchor_list = [anchors_1, anchors_2, anchors_3]

conf_threshold = 0.3
iou_threshold  = 0.4

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

model = None

def initModel(_model):
    global model
    model = _model


def predict(image, input_resized):
    # Since the model is now not fully supported, we have to run the feature extraction part of the model
    # Then process the features with numpy

    # Feature extraction
    global model
    img_preprocess, img_w, img_h, new_w, new_h, shift_x_ratio, shift_y_ratio = preprocess(image, aipp_flag)
    outputs = model.infer([input_resized.flatten()])

    # Post proccess the features
    res = get_result(outputs, img_w, img_h, new_w, new_h, shift_x_ratio, shift_y_ratio)
    return res
 
def preprocess(image, aipp_flag=True):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = image.shape    
    
    scale = min(float(net_w)/float(img_w), float(net_h)/float(img_h))
    new_w = int(img_w*scale)
    new_h = int(img_h*scale)
    
    shift_x = (net_w - new_w) // 2
    shift_y = (net_h - new_h) // 2
    shift_x_ratio = (net_w - new_w) / 2.0 / net_w
    shift_y_ratio = (net_h - new_h) / 2.0 / net_h
    
    image = cv2.resize(image, (new_w, new_h))
    
    if aipp_flag:
        new_image = np.zeros((net_h, net_w, 3), np.uint8)
    else:
        new_image = np.zeros((net_h, net_w, 3), np.float32)
    new_image.fill(128) 
    new_image[shift_y : new_h+shift_y, shift_x : new_w+shift_x, :] = image

    if not aipp_flag:
        new_image /= 255.
    return new_image, img_w, img_h, new_w, new_h, shift_x_ratio, shift_y_ratio


def overlap(x1, x2, x3, x4):
    left = max(x1, x3)
    right = min(x2, x4)
    return right - left

def cal_iou(box, truth):
    w = overlap(box[0], box[2], truth[0], truth[2])
    h = overlap(box[1], box[3], truth[1], truth[3])
    if w <= 0 or h <= 0:
        return 0
    inter_area = w * h
    union_area = (box[2] - box[0]) * (box[3] - box[1]) + (truth[2] - truth[0]) * (truth[3] - truth[1]) - inter_area
    return inter_area * 1.0 / union_area

def apply_nms(all_boxes, thres):
    res = []
    
    for cls in range(class_num):        
        cls_bboxes   = all_boxes[cls]
        sorted_boxes = sorted(cls_bboxes, key=lambda d: d[5])[::-1]
        
        p = dict()
        for i in range(len(sorted_boxes)):
            if i in p:
                continue

            truth = sorted_boxes[i]
            for j in range(i+1, len(sorted_boxes)):
                if j in p:
                    continue
                box = sorted_boxes[j]
                iou = cal_iou(box, truth)
                if iou >= thres:
                    p[j] = 1

        for i in range(len(sorted_boxes)):
            if i not in p:
                res.append(sorted_boxes[i])
    return res

def decode_bbox(conv_output, anchors, img_w, img_h, x_scale, y_scale, shift_x_ratio, shift_y_ratio):

    def _sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s
    
    _, h, w = conv_output.shape    
    pred    = conv_output.transpose((1,2,0)).reshape((h * w, 3, 5+class_num))
    
    pred[..., 4:] = _sigmoid(pred[..., 4:])
    pred[..., 0]  = (_sigmoid(pred[..., 0]) + np.tile(range(w), (3, h)).transpose((1,0))) / w
    pred[..., 1]  = (_sigmoid(pred[..., 1]) + np.tile(np.repeat(range(h), w), (3, 1)).transpose((1,0))) / h
    pred[..., 2]  = np.exp(pred[..., 2]) * anchors[:, 0:1].transpose((1,0)) / w
    pred[..., 3]  = np.exp(pred[..., 3]) * anchors[:, 1:2].transpose((1,0)) / h
                  
    bbox          = np.zeros((h * w, 3, 4))
    bbox[..., 0]  = np.maximum((pred[..., 0] - pred[..., 2] / 2.0 - shift_x_ratio) * x_scale * img_w, 0)     #x_min
    bbox[..., 1]  = np.maximum((pred[..., 1] - pred[..., 3] / 2.0 - shift_y_ratio) * y_scale * img_h, 0)     #y_min
    bbox[..., 2]  = np.minimum((pred[..., 0] + pred[..., 2] / 2.0 - shift_x_ratio) * x_scale * img_w, img_w) #x_max
    bbox[..., 3]  = np.minimum((pred[..., 1] + pred[..., 3] / 2.0 - shift_y_ratio) * y_scale * img_h, img_h) #y_max
    
    pred[..., :4] = bbox
    pred          = pred.reshape((-1, 5+class_num))
    pred[:, 4]    = pred[:, 4] * pred[:, 5:].max(1) 
    pred          = pred[pred[:, 4] >= conf_threshold]
    pred[:, 5]    = np.argmax(pred[:, 5:], axis=-1)
    
    all_boxes = [[] for ix in range(class_num)]
    for ix in range(pred.shape[0]):
        box = [int(pred[ix, iy]) for iy in range(4)]
        box.append(int(pred[ix, 5]))
        box.append(pred[ix, 4])
        all_boxes[box[4]-1].append(box)

    return all_boxes

def get_result(model_outputs, img_w, img_h, new_w, new_h, shift_x_ratio, shift_y_ratio):
    
    time_start = time.time()
    
    num_channel = 3 * (class_num + 5)
    x_scale     = net_w / float(new_w)
    y_scale     = net_h / float(new_h)    
    all_boxes   = [[] for ix in range(class_num)]
    for ix in range(3):
        pred      = model_outputs[2-ix].reshape((num_channel, net_h // stride_list[ix], net_w // stride_list[ix]))
        anchors   = anchor_list[ix]
        boxes     = decode_bbox(pred, anchors, img_w, img_h, x_scale, y_scale, shift_x_ratio, shift_y_ratio)        
        all_boxes = [all_boxes[iy] + boxes[iy] for iy in range(class_num)]
    
    res = apply_nms(all_boxes, iou_threshold)
    
    time_end = time.time()
    #print ('postprocess time : ', time_end - time_start)
    
    return res
    
def draw_box_on_img(img_data, res):
    text_thickness = 2
    line_type = 1
    thickness = 2
    for bbox in res:
        x_min = int(bbox[0])
        y_min = int(bbox[1])
        x_max = int(bbox[2])
        y_max = int(bbox[3])
        label = int(bbox[4])
        score = bbox[5]
        cv2.rectangle(img_data, (x_min, y_min), (x_max, y_max), colors[label%6], thickness)

        s = '%s %.1f' % (class_names[label], score*100)
        cv2.putText(img_data, s, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), text_thickness, line_type)

    return img_data    
