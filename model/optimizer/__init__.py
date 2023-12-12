#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @folder: optimizer
# @author: jerrzzy
# @date: 2023/8/26


from .optimizer import TrajectoryOptimizer


def yolo_optimize(trk_seq_path, opt_result_path):
    """
    Params:
    trk_seq_path - MOT格式的追踪结果文件路径
    opt_result_path - MOT格式的优化结果文件路径
    """
    to = TrajectoryOptimizer()
    to.global_optimization(trk_seq_path=trk_seq_path, opt_result_path=opt_result_path)

