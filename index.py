# -*- coding: utf-8 -*-
# !/usr/bin/python3

import hilens
import cv2
import numpy as np
import threading
import os
import json
import requests
from requests.exceptions import ConnectionError, Timeout
from time import sleep
from yolov3 import predict, initModel, draw_box_on_img
from distort import lens_distortion_adjustment

model   = None
camera1 = None
camera2 = None
upload_uri1 = None
upload_uri2 = None
display_hdmi = None

# trick: lens_mtxs[0] is for camera1
lens_mtxs = [
    np.array([[1532.3835774495203, 0.0, 963.2349158394692], [0.0, 1529.9521976155718, 534.408028051908], [0.0, 0.0, 1.0]]),
    np.array([[1.58206604e+03, 0.0, 9.87620699e+02], [0.0, 1.57889682e+03, 5.43361354e+02],[0.0, 0.0, 1.0]])
]

# trick: lens_dists[0] is for camera1
lens_dists = [
    np.array([[-0.2713632975134281, -0.14711397562895906, 0.0031891790347745616, -0.0013621657772392918, 0.20355047701488785]]),
    np.array([[-0.29047252, -0.1558767, 0.0004676, -0.0008854, 0.26806017]])
]


def init():
    global model, camera1, camera2, upload_uri1, upload_uri2, display_hdmi

    hilens.init("hello")
    model_path = hilens.get_model_dir() + "./convert-4label.om"
    model      = hilens.Model(model_path)
    display_hdmi = hilens.Display(hilens.HDMI)


    hilens.set_log_level(hilens.DEBUG)

    skill_cfg = hilens.get_skill_config()
    if skill_cfg is None or 'IPC_address1' not in skill_cfg or 'upload_uri1' not in skill_cfg:
        hilens.fatal('Missing IPC1 skill configs! skill_cfg: {}'.format(skill_cfg))
        hilens.terminate()
        exit(1)
    try:
        camera1 = hilens.VideoCapture(skill_cfg['IPC_address1'])
    except Exception as e:
        hilens.fatal("Failed to create camera1 with {}, e: {}", skill_cfg['IPC_address1'], e)
    upload_uri1 = skill_cfg['upload_uri1']

    if  'IPC_address2' not in skill_cfg or 'upload_uri2' not in skill_cfg:
        hilens.warning('Missing IPC2 skill configs! Camera2 will not work. skill_cfg: {}'.format(skill_cfg))
    else:
        try:
            camera2 = hilens.VideoCapture(skill_cfg['IPC_address2'])
        except Exception as e:
            hilens.fatal("Failed to create camera2 with {}, e: {}", skill_cfg['IPC_address2'], e)
        upload_uri2 = skill_cfg['upload_uri2']

    initModel(model)

input_height = 352
input_width  = 640

def sendData(uri, res):
    for item in res:
        item[5] = '%.3f' % item[5] # Convert posibility from float to str
    try:
        requests.post(uri, json.dumps(res).encode('UTF-8'))
    except ConnectionError as e:
        hilens.error('Failed to send data, connection error:{}'.format(e))
    except Timeout as e:
        hilens.error('Failed to send data, connection timeout:{}'.format(e))

def capVideo(cap, flag, upload_uri):
    global input_height
    global input_width

    stable = False
    num_stable_frames = 0
    stable_frames_threshold = 5

    lens_mtx    = lens_mtxs[flag]
    lens_dist   = lens_dists[flag]

    while True:
        start = cv2.getTickCount()
        frame = cap.read()
        input_bgr = cv2.cvtColor(frame,cv2.COLOR_YUV2BGR_NV21)
        input_bgr = lens_distortion_adjustment(input_bgr, lens_mtx=lens_mtx, lens_dist=lens_dist)

        h, w, _ = input_bgr.shape
        cropped_input_bgr = input_bgr[h//5:h//5*4, w//5:w//5*4, :]

        input_resized = cv2.resize(cropped_input_bgr, (input_width, input_height))
        res = predict(cropped_input_bgr, input_resized)
        finish = cv2.getTickCount()
        duration = (finish - start) / cv2.getTickFrequency()
        fps = int(1/duration)

        output_nv21 = None
        if len(res) != 0:
            print('camera{} got res: {}'.format(flag, res))
            deltaY = h // 5
            deltaX = w // 5
            for item in res:
                item[0] += deltaX
                item[2] += deltaX

                item[1] += deltaY
                item[3] += deltaY
            num_stable_frames += 1
            stable = num_stable_frames >= stable_frames_threshold

            c = (0, 0, 255)
            if stable:
                c = (255,0,0)

            img_data = draw_box_on_img(input_bgr, res, c)
            output_nv21 = hilens.cvt_color(img_data, hilens.BGR2YUV_NV21)
        else:
            output_nv21 = frame
            stable = False
            num_stable_frames = 0

        if flag == 1:
            display_hdmi.show(output_nv21)

        if stable:
                print('stable, send data')
                sendData(upload_uri, res)
                sleep(3)
                stable = False
                num_stable_frames = 0

def runThreads():
    cam_thread1 = threading.Thread(target=capVideo, args=(camera1, 0, upload_uri1))
    cam_thread1.start()
    cam_thread1.join()

    global camera2
    if camera2 != None:
        cam_thread2 = threading.Thread(target=capVideo, args=(camera2, 1, upload_uri2))
        cam_thread2.start()
        cam_thread2.join()

if __name__ == '__main__':
    init()
    runThreads()
    hilens.terminate()
