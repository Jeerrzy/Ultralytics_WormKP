#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jeerrzy
# @File     : business.py
# @Time     : 2023/12/11 16:27
# @Project  : ultralytics_yolo_kp


import os
from test import predictVideo


root = 'D:/Worm_CPU_datasets/CPU_Dataset_20231211/D15'


if __name__ == "__main__":
    baseDirName = os.path.basename(root)
    if not os.path.exists(baseDirName):
        os.makedirs(baseDirName)
    for fileName in os.listdir(root):
        name, _ = os.path.splitext(fileName)
        filePath = os.path.join(root, fileName)
        outVideoPath = os.path.join(baseDirName, fileName)
        cacheJsonPath = os.path.join(baseDirName, name+'.json')
        resultJsonPath = os.path.join(baseDirName, name+'_result.json')
        predictVideo(
            testVideoPath=filePath,
            outVideoPath=outVideoPath,
            cacheJsonPath=cacheJsonPath,
            resultJsonPath=resultJsonPath
        )

