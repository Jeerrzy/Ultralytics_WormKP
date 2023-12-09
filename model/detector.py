#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jeerrzy
# @File     : detector.py
# @Time     : 2023/12/1 13:58
# @Project  : ultralytics_yolo_kp


import cv2
import json
from tqdm import tqdm
from .utils import get_current_time_str, NpEncoder
from ultralytics import YOLO


class YoloKPDetector(object):
    _defaults = {
        "model_path": 'model/weights/yolov8_n_pose_2023_12_1_CPU_worm.pt',
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value
        self.model = YOLO(self.model_path)

    def detect(self, imageData):
        """
        :param imageData: (Numpy.ndarray[H, W, C]) opencv格式的原始图像数据
        :return:
            bboxes: (Numpy.ndarray[N, 4]) N个目标，分别用BBOX(x0, y0, x1, y1)表示
            keypoints: (Numpy.ndarray[N, K, 2]) N个目标，分别用K个关键点表示，每个关键点的格式为二维向量[x, y]
        """
        result = self.model(imageData)[0].cpu().numpy()
        bboxes = result.boxes.xyxy  # 边界框输出的 Boxes 对象
        keypoints = result.keypoints.xy  # 姿态输出的 Keypoints 对象
        return bboxes, keypoints

    def track(self, videoPath, cacheJsonPath):
        """
        :param videoPath: (Str) 字符串格式的输入视频路径
        :param videoPath: (Str) 字符串格式的缓存Json文件路径
        :return: None
        """
        print(f'Track {videoPath} ...')
        start_time = get_current_time_str()
        cap = cv2.VideoCapture(videoPath)
        videoInfo = {
            'frameNum': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(cap.get(cv2.CAP_PROP_FPS))
        }
        frames = []
        frame_idx = 1
        id_list = []
        with tqdm(total=videoInfo['frameNum']) as pbar:
            pbar.set_description('Processing')
            while cap.isOpened():
                # 从视频读取一帧
                success, frame = cap.read()
                if not success:
                    break
                # 在帧上运行YOLOv8追踪，持续追踪帧间的物体
                result = self.model.track(frame, persist=True)[0].cpu().numpy()
                bboxes = result.boxes.xyxy  # 边界框输出的bboxes
                ids = list(map(lambda x: int(x), result.boxes.id))  # 边界框输出的ids
                for _id in ids:
                    if _id not in id_list:
                        id_list.append(_id)
                keypoints = result.keypoints.xy  # 姿态输出的keypoints对象
                assert len(bboxes) == len(ids) == len(keypoints), print('object number match error!')
                frame_result = {
                    int(ids[i]):
                        {
                            'id': int(ids[i]),
                            'bbox': bboxes[i],
                            'kps': keypoints[i]
                        }
                    for i in range(len(ids))
                }
                frames.append(frame_result)
                frame_idx += 1
                pbar.update(1)
        end_time = get_current_time_str()
        result = {
            'time': start_time + ' ~~~~~~ ' + end_time,
            'videoInfo': videoInfo,
            'idList': id_list,
            'frames': frames
        }
        with open(cacheJsonPath, 'w') as f:
            json.dump(result, f, indent=2, cls=NpEncoder)
        print(f'Track {videoPath} down, save result to {cacheJsonPath}.')

