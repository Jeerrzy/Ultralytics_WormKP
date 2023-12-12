#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jeerrzy
# @File     : test.py
# @Time     : 2023/12/3 16:33
# @Project  : ultralytics_yolo_kp


import os

import numpy as np
import matplotlib.pyplot as plt
from model import *


def predictImage(testImagePath):
    model = YoloKPDetector()
    imageData = cv2.imread(testImagePath)
    bboxes, kps = model.detect(imageData)
    visualizeImage(imageData, bboxes, kps, scale=0.25, show=True)


def predictVideo(testVideoPath, outVideoPath='./runCache.mp4', cacheJsonPath='./runCache.json', resultJsonPath='./result.json'):
    model = YoloKPDetector()
    model.track(testVideoPath, cacheJsonPath)
    yolo_optimize(cacheJsonPath, cacheJsonPath)
    getTwistNumber(cacheJsonPath, refile=True, savePath=resultJsonPath, threshold=160)
    visualizeVideo(
        srcVideoPath=testVideoPath,
        outVideoPath=outVideoPath,
        cacheJsonPath=cacheJsonPath,
        scale=0.25
    )



def getTwistInfo(cacheJsonPath, savePath):
    getTwistNumber(cacheJsonPath, refile=False, savePath=savePath)



def twistDetectVideo(testVideoPath, outVideoPath='./runCache.mp4', cacheJsonPath='./runCache.json', pltSaveDir='./run'):
    model = YoloKPDetector()
    model.track(testVideoPath, cacheJsonPath)
    twistInfo = getTwistNumber(cacheJsonPath)
    if not os.path.exists(pltSaveDir):
        os.makedirs(pltSaveDir)
    for key in twistInfo.keys():
        id_info = twistInfo[key]
        faList = np.array(id_info['angleList'])
        visualizeAngleList(faList[:, 0], faList[:, 1], id_info['peaks'],
                           show=False, savePath=os.path.join(pltSaveDir, str(key)+'.png'))
    visualizeVideo(
        srcVideoPath=testVideoPath,
        outVideoPath=outVideoPath,
        cacheJsonPath=cacheJsonPath,
        scale=0.25
    )
    print('Get Twist Number Result:')
    for _id in twistInfo.keys():
        number = twistInfo[_id]['number']
        print(f'{_id}: {number}')


if __name__ == "__main__":
    th_info_array = []
    x_ = np.array(list(range(0, 200, 5)))
    for th in x_:
        info = getTwistNumber(
            cacheJsonPath='./D9/WL-D9.json',
            refile=False,
            savePath=None,
            threshold=th
        )
        keyinfos = [info[key]['number'] for key in info.keys()]
        th_info_array.append(keyinfos)
    arr = np.array(th_info_array)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for col in range(arr.shape[1]):
        ax.plot(x_, arr[:, col])
    plt.show()
