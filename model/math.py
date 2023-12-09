#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jeerrzy
# @File     : math.py
# @Time     : 2023/12/3 19:17
# @Project  : ultralytics_yolo_kp


import math
import json
import numpy as np
from .utils import NpEncoder


def getTwistNumber(cacheJsonPath, refile=True, savePath=None):
    """
    :param cacheJsonPath: (Str) json格式的视频追踪缓存结果的路径
    :return: 包含每个目标扭动行为行为检测的信息
        id2TwistInfo{
            _id: {
                angleList: (List) [[f1, angle1], [f2, angle2], ...] 帧数和对应的角度列表
                peaks: (List) [f1, f2, ...] 出现极大值的帧数列表
                number: (int) 极大值数量，即判定发生扭动的次数
            }
            ...
        }
    """
    id2TwistInfo = {}
    with open(cacheJsonPath, 'r') as f:
        rawData = json.load(f)
        id_list = rawData['idList']
        frames = rawData['frames']
    # 检测扭动
    for _id in id_list:
        frame_angle_list = []
        for frame_idx, frame_result in enumerate(frames):
            if str(_id) in frame_result.keys():
                id_data = frame_result[str(_id)]
                (ep1, cp, ep2) = id_data['kps']
                angle = get_angle(ep1, cp, ep2)
                frame_angle_list.append([frame_idx+1, angle])
        peaks = find_peaks(frame_angle_list)
        twist_info = {
            'angleList': frame_angle_list,
            'peaks': peaks,
            'number': len(peaks)
        }
        id2TwistInfo[_id] = twist_info
    if refile:
        # 更新Json
        id2TwistNumber = {_id: 0 for _id in id_list}
        for frame_idx, frame_result in enumerate(frames):
            for key in frame_result.keys():
                id_data = frame_result[key]
                _id = id_data['id']
                if frame_idx in id2TwistInfo[_id]['peaks']:
                    id2TwistNumber[_id] += 1
                id_data['tn'] = id2TwistNumber[_id]
        with open(cacheJsonPath, 'w') as f:
            json.dump(rawData, f, indent=2, cls=NpEncoder)
    if savePath is not None:
        with open(savePath, 'w') as f:
            json.dump(id2TwistInfo, f, indent=2, cls=NpEncoder)
    return id2TwistInfo


def get_angle(ep1, cp, ep2):
    """
    :param ep1: (List) (x, y)格式的端点坐标
    :param cp: (List) (x, y)格式的中心点点坐标
    :param ep2: (List) (x, y)格式的端点坐标
    :return: 三点以中心点为顶点所成的角度
    """
    ep1, cp, ep2 = np.array(ep1), np.array(cp), np.array(ep2)
    v1, v2 = ep1 - cp, ep2 - cp
    try:
        cos_value = (v1.dot(v2))/(np.linalg.norm(v1) * np.linalg.norm(v2))
        angle_radian = math.acos(cos_value)
        angle_degree = math.degrees(angle_radian)
    except:
        angle_degree = 0
    return angle_degree


def find_peaks(frame_angle_list):
    """
    :param frame_angle_list: (List) [[f1, angle1], [f2, angle2], ...] 帧数和对应的角度列表
    :return: (List) [f1, f2, ...] 出现极大值的帧数列表
    """
    peaks = []
    for idx in range(1, len(frame_angle_list) - 1):
        if frame_angle_list[idx - 1][1] < frame_angle_list[idx][1] and frame_angle_list[idx + 1][1] < frame_angle_list[idx][1]:
            peaks.append(int(frame_angle_list[idx][0]))
    return peaks

