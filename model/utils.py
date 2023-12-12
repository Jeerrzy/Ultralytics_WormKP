#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jeerrzy
# @File     : utils.py
# @Time     : 2023/12/3 15:04
# @Project  : ultralytics_yolo_kp
import os.path

import cv2
import copy
import datetime
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# ---------------------------------------------------- #
DrawFrameIDPos = (100, 150)
DrawFrameIDSize = 5
DrawFrameIDColor = (255, 0, 0)
DrawFrameIDThickness = 3
# ---------------------------------------------------- #
DrawBBoxColor = (0, 255, 0)  # 绘制BBox的BGR颜色
DrawBBoxThickness = 2  # 绘制BBox的线条粗细
# ---------------------------------------------------- #
DrawIdColor = (130, 0, 75)  # 绘制ID的BGR颜色
DrawIdSize = 2  # 绘制ID的大小
DrawIdThickness = 4  # 绘制ID的线条粗细
# ---------------------------------------------------- #
DrawTNColor = (75, 0, 130)  # 绘制扭动次数的BGR颜色
DrawTNSize = 3  # 绘制扭动次数的字体大小
DrawTNThickness = 4  # 绘制扭动次数的线条粗细
# ---------------------------------------------------- #
DrawEPColor = (255, 0, 0)  # 绘制端点的BGR颜色
DrawEPRadius = 10  # 绘制端点的半径
DrawEPThickness = -1  # 绘制端点的线条粗细，负数表示实心圆
# ---------------------------------------------------- #
DrawCPColor = (0, 0, 255)  # 绘制端点的BGR颜色
DrawCPRadius = 10  # 绘制端点的半径
DrawCPThickness = -1  # 绘制端点的线条粗细，负数表示实心圆
# ---------------------------------------------------- #
DrawLineColor = (128, 0, 128)  # 绘制端点的BGR颜色
DrawLineThickness = 2  # 绘制端点的线条粗细，负数表示实心圆
# ---------------------------------------------------- #


def get_current_time_str():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def imageScaleResize(imageData, scale):
    """
    :param imageData: (Numpy.ndarray[H, W, C]) opencv格式的原始图像数据
    :param scale: (Float) 缩放尺度系数
    """
    return cv2.resize(imageData, (int(imageData.shape[1]*scale), int(imageData.shape[0]*scale)))


def visualizeImage(imageData, bboxes, keypoints, scale=1.0, show=False):
    """
    :param imageData: (Numpy.ndarray[H, W, C]) opencv格式的原始图像数据
    :param bboxes: (Numpy.ndarray[N, 4]) N个目标，分别用BBOX(x0, y0, x1, y1)表示
    :param keypoints: (Numpy.ndarray[N, K, 2]) N个目标，分别用K个关键点表示，每个关键点的格式为二维向量[x, y]
    :param scale: (Float) 缩放尺度
    """
    # 绘制外接矩形框
    imageData = copy.deepcopy(imageData)
    for (x0, y0, x1, y1) in bboxes:
        (x0, y0, x1, y1) = map(lambda x: int(x), (x0, y0, x1, y1))
        cv2.rectangle(imageData, (x0, y0), (x1, y1), DrawBBoxColor, DrawBBoxThickness)
    # 绘制关键点
    for ((xe1, ye1), (xc, yc), (xe2, ye2)) in keypoints:
        (xe1, ye1, xc, yc, xe2, ye2) = map(lambda x: int(x), (xe1, ye1, xc, yc, xe2, ye2))
        cv2.circle(imageData, (xe1, ye1), DrawEPRadius, DrawEPColor, DrawEPThickness)
        cv2.circle(imageData, (xe2, ye2), DrawEPRadius, DrawEPColor, DrawEPThickness)
        cv2.circle(imageData, (xc, yc), DrawCPRadius, DrawCPColor, DrawCPThickness)
        cv2.line(imageData, (xe1, ye1), (xc, yc), DrawLineColor, DrawLineThickness, cv2.LINE_AA)
        cv2.line(imageData, (xe2, ye2), (xc, yc), DrawLineColor, DrawLineThickness, cv2.LINE_AA)
    if scale is not None:
        imageData = imageScaleResize(imageData, scale=scale)
    if show:
        cv2.imshow('demo', imageData)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return imageData


def visualizeVideo(srcVideoPath, outVideoPath, cacheJsonPath, fps=None, scale=None):
    """
    :param srcVideoPath: (Str) 字符串格式的输入视频路径
    :param outVideoPath: (Str) 字符串格式的输出视频路径
    :param cacheJsonPath: (Str) 字符串格式的缓存Json缓存文件路径
    :param fps: (int) 输出视频的帧速
    :param scale: (float) 输出视频的缩放尺度
    :return: None
    """
    print(f'visualizing {srcVideoPath} ...')
    with open(cacheJsonPath, 'r') as f:
        rawData = json.load(f)
        videoInfo = rawData['videoInfo']
        frames = rawData['frames']
    src_video = cv2.VideoCapture(srcVideoPath)
    out_fps = videoInfo['fps'] if fps is None else fps
    out_w = videoInfo['width'] if scale is None else int(videoInfo['width'] * scale)
    out_h = videoInfo['height'] if scale is None else int(videoInfo['height'] * scale)
    out_video = cv2.VideoWriter(outVideoPath, cv2.VideoWriter_fourcc(*'XVID'), out_fps, (out_w, out_h))
    frame_idx = 1
    with tqdm(total=videoInfo['frameNum'] - 1) as pbar:
        pbar.set_description('Processing')
        while src_video.isOpened():
            ret, frame = src_video.read()
            if not ret:
                break
            frame_result = frames[frame_idx - 1]
            cv2.putText(frame, str(frame_idx), DrawFrameIDPos, cv2.FONT_HERSHEY_SIMPLEX,
                        DrawFrameIDSize, DrawFrameIDColor, DrawFrameIDThickness)
            for key in frame_result.keys():
                wormObjDict = frame_result[key]
                frame = visualizeFrame(frame, wormObjDict)
            out_video.write(imageScaleResize(frame, scale))
            frame_idx += 1
            pbar.update(1)
    src_video.release()
    out_video.release()
    print(f'visualizing {srcVideoPath} down.')


def visualizeFrame(frame, wormObjDict):
    """
    :param wormObjDict: 模型对象字典
    :param frame: 绘制图片
    """
    (x0, y0, x1, y1) = map(lambda x: int(x), wormObjDict['bbox'])
    cv2.rectangle(frame, (x0, y0), (x1, y1), DrawBBoxColor, DrawBBoxThickness)
    try:
        ((xe1, ye1), (xc, yc), (xe2, ye2)) = wormObjDict['kps']
        (xe1, ye1, xc, yc, xe2, ye2) = map(lambda x: int(x), (xe1, ye1, xc, yc, xe2, ye2))
        cv2.circle(frame, (xe1, ye1), DrawEPRadius, DrawEPColor, DrawEPThickness)
        cv2.circle(frame, (xe2, ye2), DrawEPRadius, DrawEPColor, DrawEPThickness)
        cv2.circle(frame, (xc, yc), DrawCPRadius, DrawCPColor, DrawCPThickness)
        cv2.line(frame, (xe1, ye1), (xc, yc), DrawLineColor, DrawLineThickness, cv2.LINE_AA)
        cv2.line(frame, (xe2, ye2), (xc, yc), DrawLineColor, DrawLineThickness, cv2.LINE_AA)
    except:
        pass
    cv2.putText(frame, str(wormObjDict['id']), (x0, y0), cv2.FONT_HERSHEY_SIMPLEX,
                DrawIdSize, DrawIdColor, DrawIdThickness)
    try:
        cv2.putText(frame, str(wormObjDict['tn']), (x1, y0), cv2.FONT_HERSHEY_SIMPLEX,
                    DrawTNSize, DrawTNColor, DrawTNThickness)
    except:
        pass
    return frame


def visualizeAngleList(frameList, angleList, peaks=None, show=True, savePath=None):
    """
    :param frameList: (List) [f1, f2, ...] 帧数列表
    :param angleList: (List) [angle1, angle2, ...] 角度值列表
    :return: None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.scatter(np.array(frameList), np.array(angleList))
    ax.plot(np.array(frameList), np.array(angleList))
    # if peaks is not None:
    #     for x in peaks:
    #         ax.axvline(x=x, color='r')
    if show:
        plt.show()
    if savePath is not None:
        plt.savefig(savePath)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

