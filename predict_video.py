#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jeerrzy
# @File     : predict_video.py
# @Time     : 2023/12/3 16:48
# @Project  : ultralytics_yolo_kp


from model import *


testVideoPath = './test.mp4'
outVideoPath = './test_result.mp4'
cacheJsonPath = './test.json'


if __name__ == "__main__":
    model = YoloKPDetector()
    model.track(testVideoPath, cacheJsonPath)
    visualizeVideo(
        srcVideoPath=testVideoPath,
        outVideoPath=outVideoPath,
        cacheJsonPath=cacheJsonPath,
        scale=0.25
    )
