# -*- coding: utf-8 -*-
# !/usr/bin/python3

import hilens
import cv2
import numpy as np
import threading
import os
import requests
from time import sleep
from yolov3 import predict, initModel, draw_box_on_img
from distort import lens_distortion_adjustment

model   = None
camera1 = None
camera2 = None
upload_uri1 = None
upload_uri2 = None
display_hdmi = None

def init():
    global model, camera1, camera2, upload_uri1, upload_uri2, display_hdmi

    hilens.init("hello")
    model_path = hilens.get_model_dir() + "./convert-ta.om"
    model      = hilens.Model(model_path)
    display_hdmi = hilens.Display(hilens.HDMI)


    hilens.set_log_level(hilens.DEBUG)

    skill_cfg = hilens.get_skill_config()
    if skill_cfg is None or 'IPC_address1' not in skill_cfg or 'IPC_address2' not in skill_cfg:
        hilens.fatal('Missing IPC_addresses! skill_cfg: {}'.format(skill_cfg))
        hilens.terminate()
        exit(1)

    if 'upload_uri1' not in skill_cfg or 'upload_uri2' not in skill_cfg:
        hilens.fatal('Missing upload URIs! skill_cfg: {}'.format(skill_cfg))
        hilens.terminate()
        exit(1)

    camera1 = hilens.VideoCapture(skill_cfg['IPC_address1'])
    camera2 = hilens.VideoCapture(skill_cfg['IPC_address2'])

    upload_uri1 = skill_cfg['upload_uri1']
    upload_uri2 = skill_cfg['upload_uri2']

    initModel(model)

input_height = 352
input_width  = 640

def capVideo(cap, flag, upload_uri):
    global input_height
    global input_width

    while True:
        start = cv2.getTickCount()
        frame = cap.read()
        input_bgr = cv2.cvtColor(frame,cv2.COLOR_YUV2BGR_NV21)
        input_bgr = lens_distortion_adjustment(input_bgr)

        h, w, _ = input_bgr.shape
        cropped_input_bgr = input_bgr[h//5:h//5*4, w//5:w//5*4, :]

        input_resized = cv2.resize(cropped_input_bgr, (input_width, input_height))
        res = predict(cropped_input_bgr, input_resized)
        finish = cv2.getTickCount()
        duration = (finish - start) / cv2.getTickFrequency()
        fps = int(1/duration)

        output_nv21 = None
        if len(res) != 0:
            deltaY = h // 5
            deltaX = w // 5
            for item in res:
                item[0] += deltaX
                item[2] += deltaX

                item[1] += deltaY
                item[3] += deltaY
            img_data = draw_box_on_img(input_bgr, res)
            output_nv21 = hilens.cvt_color(img_data, hilens.BGR2YUV_NV21)
        else:
            output_nv21 = frame

        display_hdmi.show(output_nv21)

def runThreads():
    cam_thread1 = threading.Thread(target=capVideo, args=(camera1, 1, upload_uri1))
    cam_thread1.start()

    cam_thread2 = threading.Thread(target=capVideo, args=(camera2, 2, upload_uri2))
    cam_thread2.start()

    cam_thread1.join()
    cam_thread2.join()


if __name__ == '__main__':
    init()
    runThreads()
    hilens.terminate()
