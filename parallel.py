# -*- coding: utf-8 -*-
# !/usr/bin/python3
# SkillFramework 0.2.2 python demo

# 测试情况：
# 只启动1个线程，主线程与子线程共享全局变量，速度不受影响
# 启动2个线程，主线程与2个子线程共享全局变量，帧率明显下降
# 启动2个线程时，如果一个线程不做解码，则速度没有影响

import hilens
import cv2
import numpy as np
import threading
import os
from time import sleep
from yolov3 import predict, initModel

os.environ["SKILL_ID"] = "ff8080826f675342016f948f917702dd"

model   = None
camera1 = None
camera2 = None

def init():
    global model, camera1, camera2

    hilens.init("hello")
    model_path = hilens.get_model_dir() + "./convert-ta.om"
    model      = hilens.Model(model_path)

    hilens.set_log_level(hilens.DEBUG)

    skill_cfg = hilens.get_skill_config()
    if skill_cfg is None or 'IPC_address1' not in skill_cfg or 'IPC_address2' not in skill_cfg:
        hilens.fatal('Missing IPC_addresses! skill_cfg: ' + skill_cfg)
        hilens.terminate()
        exit(1)

    #camera1 = hilens.VideoCapture(skill_cfg['IPC_address1'])
    #camera2 = hilens.VideoCapture(skill_cfg['IPC_address2'])

    #DEBUGGING
    #camera1 = hilens.VideoCapture("rtsp://192.168.0.157:554/stream2")
    #camera2 = hilens.VideoCapture("rtsp://192.168.0.165:554/stream2")

    initModel(model)

input_height = 352
input_width  = 640

def capVideo(cap, flag):
    global input_height
    global input_width

    while True:
        start = cv2.getTickCount()
        frame = cap.read()
        input_bgr = cv2.cvtColor(frame,cv2.COLOR_YUV2BGR_NV21)
        input_resized = cv2.resize(input_bgr, (input_width, input_height))
        res = predict(input_bgr, input_resized)
        finish = cv2.getTickCount()
        duration = (finish - start) / cv2.getTickFrequency()
        fps = int(1/duration)
        if len(res) != 0:
            hilens.debug('cap.flag: {}, fps: {}, res:{}'.format(flag, fps, res))

def runThreads():
    cam_thread1 = threading.Thread(target=capVideo, args=(camera1, 1))
    cam_thread1.start()

    cam_thread2 = threading.Thread(target=capVideo, args=(camera2, 2))
    cam_thread2.start()

    cam_thread1.join()
    cam_thread2.join()


if __name__ == '__main__':
    init()
    runThreads()
    hilens.terminate()
