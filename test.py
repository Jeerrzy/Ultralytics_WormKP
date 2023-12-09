#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jeerrzy
# @File     : test.py
# @Time     : 2023/12/3 16:33
# @Project  : ultralytics_yolo_kp


import os
from model import *


def predictImage(testImagePath):
    model = YoloKPDetector()
    imageData = cv2.imread(testImagePath)
    bboxes, kps = model.detect(imageData)
    visualizeImage(imageData, bboxes, kps, scale=0.25, show=True)


def predictVideo(testVideoPath, outVideoPath='./runCache.mp4', cacheJsonPath='./runCache.json'):
    model = YoloKPDetector()
    model.track(testVideoPath, cacheJsonPath)
    visualizeVideo(
        srcVideoPath=testVideoPath,
        outVideoPath=outVideoPath,
        cacheJsonPath=cacheJsonPath,
        scale=0.25
    )


def getTwistInfo(cacheJsonPath, savePath):
    getTwistNumber(cacheJsonPath, refile=None, savePath=savePath)


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
    predictVideo('./lz_test.avi')
    # twistDetectVideo(
    #     testVideoPath='./test2.mp4'
    # )
    # info = getTwistNumber(cacheJsonPath='./test.json')

