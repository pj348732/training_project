import sys

sys.path.insert(0, '/b/home/pengfei_ji/airflow/dags/sta_17001/scripts/workflow_utils/')
sys.path.insert(0, '/b/home/pengfei_ji/airflow/dags/sta_17001/scripts/weekly_workflow/')
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from common_utils import iter_time_range, time_to_minute, n_day_before, after_one_month
from factor_loader import get_stock_map, FixMeanMultiTaskLoss, DaySkeyMapDataset
from torch.utils.data import DataLoader
from os import listdir
from os.path import isfile, join
import importlib
from factor_dao import FactorDAO
import time


# 1603786
def get_slurm_env(name):
    value = os.getenv(name)
    if value is None:
        if name == 'SLURM_ARRAY_TASK_ID' or name == 'SLURM_PROCID':
            return 0
        else:
            return 1
    else:
        return value


feat_map = {
    'MBD_FINAL': [
        'x1_1_2_0_L0000_normed', 'x1_5_1_1_L0003_L0000_normed', 'x1_5_1_1_L0006_L0003_normed',
        'x1_5_1_1_L0012_L0006_normed', 'x1_5_1_2_L0003_L0000_normed', 'x1_5_1_2_L0006_L0003_normed',
        'x1_5_1_2_L0012_L0006_normed', 'x1_6_1_0_L0096_L0000_normed', 'x1_6_2_0_L0096_L0000_normed',
        'x1_7_1_1_L0000_normed',
        'x1_7_7_1_L0000_normed', 'x1_7_8_1_L0000_normed', 'x1_7_3_1_L0000_normed', 'x1_7_2_1_L0000_normed',
        'x1_7_5_1_L0000_normed', 'x1_7_4_1_L0000_normed', 'x1_7_6_1_L0000_normed', 'x1_8_1_1_L0000_normed',
        'x1_8_1_2_L0000_normed', 'x1_9_1_0_L0000_normed', 'x1_10_1_2_L0000_normed', 'x1_10_1_1_L0000_normed',
        'x1_11_1_2_L0006_L0000_normed', 'x1_11_1_2_L0024_L0000_normed', 'x1_11_1_1_L0003_L0000_normed',
        'x1_11_1_1_L0012_L0000_normed', 'x1_11_1_2_L0012_L0000_normed', 'x1_11_1_1_L0024_L0000_normed',
        'x1_11_1_1_L0006_L0000_normed', 'x1_11_1_2_L0003_L0000_normed', 'x1_12_1_1_L0024_L0000_normed',
        'x1_12_1_2_L0048_L0000_normed', 'x1_12_1_2_L0096_L0000_normed', 'x1_12_1_1_L0048_L0000_normed',
        'x1_12_1_2_L0024_L0000_normed', 'x1_12_1_1_L0096_L0000_normed', 'x2_1_5_2_L0024_L0012_normed',
        'x2_1_3_2_L0012_L0006_normed', 'x2_1_5_1_L0012_L0006_normed', 'x2_1_2_2_L0012_L0006_normed',
        'x2_1_3_1_L0024_L0012_normed', 'x2_1_3_2_L0024_L0012_normed', 'x2_1_3_1_L0003_L0000_normed',
        'x2_1_1_1_L0024_L0012_normed', 'x2_1_5_2_L0048_L0024_normed', 'x2_1_4_1_L0012_L0006_normed',
        'x2_1_2_1_L0003_L0000_normed', 'x2_1_5_1_L0048_L0024_normed', 'x2_1_4_2_L0003_L0000_normed',
        'x2_1_1_2_L0006_L0003_normed', 'x2_1_4_1_L0006_L0003_normed', 'x2_1_3_2_L0003_L0000_normed',
        'x2_1_4_1_L0024_L0012_normed', 'x2_1_3_1_L0012_L0006_normed', 'x2_1_1_2_L0024_L0012_normed',
        'x2_1_1_1_L0012_L0006_normed', 'x2_1_3_2_L0048_L0024_normed', 'x2_1_3_1_L0048_L0024_normed',
        'x2_1_5_2_L0006_L0003_normed', 'x2_1_1_2_L0003_L0000_normed', 'x2_1_4_2_L0048_L0024_normed',
        'x2_1_3_1_L0006_L0003_normed', 'x2_1_1_1_L0006_L0003_normed', 'x2_1_5_1_L0006_L0003_normed',
        'x2_1_5_2_L0012_L0006_normed', 'x2_1_2_2_L0006_L0003_normed', 'x2_1_2_1_L0006_L0003_normed',
        'x2_1_4_2_L0024_L0012_normed', 'x2_1_4_2_L0006_L0003_normed', 'x2_1_1_2_L0096_L0048_normed',
        'x2_1_5_1_L0003_L0000_normed', 'x2_1_1_2_L0048_L0024_normed', 'x2_1_1_1_L0003_L0000_normed',
        'x2_1_5_2_L0003_L0000_normed', 'x2_1_4_1_L0048_L0024_normed', 'x2_1_1_1_L0096_L0048_normed',
        'x2_1_4_1_L0003_L0000_normed', 'x2_1_4_2_L0012_L0006_normed', 'x2_1_5_1_L0024_L0012_normed',
        'x2_1_3_2_L0006_L0003_normed', 'x2_1_2_2_L0003_L0000_normed', 'x2_1_1_1_L0048_L0024_normed',
        'x2_1_2_1_L0012_L0006_normed', 'x2_1_1_2_L0012_L0006_normed', 'x3_2_3_1_L0003_L0000_normed',
        'x3_2_3_2_L0024_L0012_normed', 'x3_2_1_1_L0003_L0000_normed', 'x3_2_2_1_L0012_L0006_normed',
        'x3_2_2_2_L0003_L0000_normed', 'x3_2_1_2_L0003_L0000_normed', 'x3_2_1_2_L0012_L0006_normed',
        'x3_2_3_2_L0006_L0003_normed', 'x3_2_1_1_L0012_L0006_normed', 'x3_2_1_1_L0006_L0003_normed',
        'x3_2_1_2_L0006_L0003_normed', 'x3_2_1_1_L0024_L0012_normed', 'x3_2_3_1_L0006_L0003_normed',
        'x3_2_2_2_L0012_L0006_normed', 'x3_2_2_1_L0024_L0012_normed', 'x3_2_2_1_L0006_L0003_normed',
        'x3_2_2_1_L0003_L0000_normed', 'x3_2_3_1_L0012_L0006_normed', 'x3_2_3_2_L0012_L0006_normed',
        'x3_2_1_2_L0024_L0012_normed', 'x3_2_2_2_L0006_L0003_normed', 'x3_2_3_2_L0003_L0000_normed',
        'x3_2_3_1_L0024_L0012_normed', 'x3_2_2_2_L0024_L0012_normed', 'x3_3_2_1_L0000_normed',
        'x3_3_1_1_L0012_L0006_normed',
        'x3_3_2_1_L0003_L0000_normed', 'x3_3_1_1_L0006_L0003_normed', 'x3_3_2_2_L0000_normed',
        'x3_3_1_2_L0003_L0000_normed',
        'x3_3_2_1_L0048_L0024_normed', 'x3_3_2_2_L0024_L0012_normed', 'x3_3_1_1_L0024_L0012_normed',
        'x3_3_1_1_L0000_normed',
        'x3_3_1_2_L0012_L0006_normed', 'x3_3_2_2_L0012_L0006_normed', 'x3_3_2_1_L0012_L0006_normed',
        'x3_3_2_2_L0003_L0000_normed', 'x3_3_1_2_L0000_normed', 'x3_3_1_2_L0006_L0003_normed',
        'x3_3_1_2_L0024_L0012_normed',
        'x3_3_1_1_L0003_L0000_normed', 'x3_3_1_1_L0048_L0024_normed', 'x3_3_2_1_L0024_L0012_normed',
        'x3_3_2_1_L0006_L0003_normed', 'x3_3_2_2_L0006_L0003_normed', 'x3_3_2_2_L0048_L0024_normed',
        'x3_3_1_2_L0048_L0024_normed', 'x4_2_1_1_L0024_L0012_normed', 'x4_2_1_2_L0024_L0012_normed',
        'x4_2_2_2_L0006_L0003_normed', 'x4_2_3_1_L0003_L0000_normed', 'x4_2_3_2_L0006_L0003_normed',
        'x4_2_3_1_L0024_L0012_normed', 'x4_2_3_2_L0024_L0012_normed', 'x4_2_1_2_L0096_L0048_normed',
        'x4_2_2_2_L0012_L0006_normed', 'x4_2_2_1_L0024_L0012_normed', 'x4_2_1_2_L0048_L0024_normed',
        'x4_2_3_1_L0006_L0003_normed', 'x4_2_1_1_L0096_L0048_normed', 'x4_2_2_1_L0003_L0000_normed',
        'x4_2_2_1_L0006_L0003_normed', 'x4_2_1_1_L0003_L0000_normed', 'x4_2_3_2_L0012_L0006_normed',
        'x4_2_2_2_L0024_L0012_normed', 'x4_2_1_1_L0012_L0006_normed', 'x4_2_2_1_L0012_L0006_normed',
        'x4_2_1_1_L0006_L0003_normed', 'x4_2_1_2_L0006_L0003_normed', 'x4_2_1_2_L0003_L0000_normed',
        'x4_2_2_2_L0003_L0000_normed', 'x4_2_3_2_L0003_L0000_normed', 'x4_2_3_1_L0012_L0006_normed',
        'x4_2_1_2_L0012_L0006_normed', 'x4_2_1_1_L0048_L0024_normed', 'x5_1_1_0_L0000_normed', 'x5_1_2_0_L0000_normed',
        'x5_2_1_2_L0024_L0012_normed', 'x5_2_1_2_L0048_L0024_normed', 'x5_2_1_1_L0024_L0012_normed',
        'x5_2_1_1_L0006_L0003_normed', 'x5_2_1_2_L0012_L0006_normed', 'x5_2_1_2_L0000_normed', 'x5_2_1_1_L0000_normed',
        'x5_2_1_2_L0003_L0000_normed', 'x5_2_1_1_L0048_L0024_normed', 'x5_2_1_2_L0006_L0003_normed',
        'x5_2_1_1_L0012_L0006_normed', 'x5_2_1_1_L0003_L0000_normed', 'x5_3_1_0_L0012_L0006_normed',
        'x5_3_2_0_L0012_L0006_normed', 'x5_3_2_0_L0003_L0000_normed', 'x5_3_2_0_L0048_L0024_normed',
        'x5_3_2_0_L0006_L0003_normed', 'x5_3_1_0_L0024_L0012_normed', 'x5_3_1_0_L0003_L0000_normed',
        'x5_3_1_0_L0096_L0048_normed', 'x5_3_1_0_L0006_L0003_normed', 'x5_3_1_0_L0048_L0024_normed',
        'x5_3_2_0_L0024_L0012_normed', 'x6_2_1_2_L0012_L0006_normed', 'x6_2_1_2_L0006_L0003_normed',
        'x6_2_1_2_L0003_L0000_normed', 'x6_2_1_1_L0003_L0000_normed', 'x6_2_1_1_L0012_L0006_normed',
        'x6_2_1_1_L0006_L0003_normed', 'x8_1_1_0_L0006_L0003_normed', 'x8_1_1_0_L0012_L0006_normed',
        'x8_1_1_0_L0096_L0048_normed', 'x8_1_1_0_L0003_L0000_normed', 'x8_1_1_0_L0048_L0024_normed',
        'x8_1_1_0_L0024_L0012_normed', 'x8_2_1_0_L0024_L0012_normed', 'x8_2_1_0_L0003_L0000_normed',
        'x8_2_1_0_L0096_L0048_normed', 'x8_2_1_0_L0048_L0024_normed', 'x8_2_1_0_L0012_L0006_normed',
        'x8_2_1_0_L0006_L0003_normed', 'x8_3_1_0_L0003_L0000_normed', 'x8_3_1_0_L0048_L0024_normed',
        'x8_3_1_0_L0006_L0003_normed', 'x8_3_1_0_L0096_L0048_normed', 'x8_3_1_0_L0024_L0012_normed',
        'x8_3_1_0_L0012_L0006_normed', 'x8_4_1_0_L0003_L0000_normed', 'x8_4_1_0_L0006_L0003_normed',
        'x8_4_1_0_L0012_L0006_normed', 'x8_4_1_0_L0024_L0012_normed', 'x8_4_1_0_L0048_L0024_normed',
        'x8_4_1_0_L0096_L0048_normed', 'x8_5_1_0_L0003_L0000_normed', 'x8_5_1_0_L0006_L0003_normed',
        'x8_5_1_0_L0012_L0006_normed', 'x8_5_1_0_L0024_L0012_normed', 'x8_5_1_0_L0048_L0024_normed',
        'x8_5_1_0_L0096_L0048_normed', 'x9_1_1_0_L0012_L0006_normed', 'x9_1_1_0_L0048_L0024_normed',
        'x9_1_1_0_L0003_L0000_normed', 'x9_1_1_0_L0096_L0048_normed', 'x9_1_1_0_L0006_L0003_normed',
        'x9_1_1_0_L0024_L0012_normed', 'x11_1_1_2_L0006_L0003_normed', 'x11_1_1_1_L0012_L0006_normed',
        'x11_1_1_1_L0003_L0000_normed', 'x11_1_1_1_L0024_L0012_normed', 'x11_1_1_2_L0003_L0000_normed',
        'x11_1_1_2_L0012_L0006_normed', 'x11_1_1_2_L0024_L0012_normed', 'x11_1_1_1_L0048_L0024_normed',
        'x11_1_1_1_L0006_L0003_normed', 'x11_1_1_2_L0048_L0024_normed', 'x12_1_2_1_L0003_L0000_normed',
        'x12_1_1_1_L0012_L0006_normed', 'x12_1_2_2_L0003_L0000_normed', 'x12_1_2_1_L0024_L0012_normed',
        'x12_1_1_2_L0003_L0000_normed', 'x12_1_2_2_L0006_L0003_normed', 'x12_1_1_2_L0024_L0012_normed',
        'x12_1_1_1_L0006_L0003_normed', 'x12_1_1_1_L0024_L0012_normed', 'x12_1_1_2_L0006_L0003_normed',
        'x12_1_1_2_L0012_L0006_normed', 'x12_1_2_1_L0012_L0006_normed', 'x12_1_2_2_L0012_L0006_normed',
        'x12_1_1_1_L0003_L0000_normed', 'x12_1_2_2_L0024_L0012_normed', 'x12_1_2_1_L0006_L0003_normed',
        'x13_1_1_0_L0000_normed', 'x13_2_1_0_L0000_normed', 'x13_3_1_0_L0000_normed', 'x13_4_1_0_L0000_normed',
        'is_five', 'is_ten', 'is_clock', 'week_id', 'session_id', 'minute_id',
    ],
    'SNAPSHOT_FINAL': [
        'x1_1_2_0_L0000_normed', 'x1_5_1_1_L0003_L0000_normed', 'x1_5_1_1_L0006_L0003_normed',
        'x1_5_1_1_L0012_L0006_normed', 'x1_5_1_2_L0003_L0000_normed', 'x1_5_1_2_L0006_L0003_normed',
        'x1_5_1_2_L0012_L0006_normed', 'x1_6_1_0_L0096_L0000_normed', 'x1_6_2_0_L0096_L0000_normed',
        'x1_7_1_1_L0000_normed',
        'x1_7_2_1_L0000_normed', 'x1_7_3_1_L0000_normed', 'x1_7_5_1_L0000_normed', 'x1_7_6_1_L0000_normed',
        'x1_7_7_1_L0000_normed', 'x1_8_1_1_L0000_normed', 'x1_8_1_2_L0000_normed', 'x1_9_1_0_L0000_normed',
        'x1_10_1_1_L0000_normed', 'x1_10_1_2_L0000_normed', 'x1_11_1_1_L0003_L0000_normed',
        'x1_11_1_1_L0006_L0000_normed',
        'x1_11_1_1_L0012_L0000_normed', 'x1_11_1_1_L0024_L0000_normed', 'x1_11_1_2_L0003_L0000_normed',
        'x1_11_1_2_L0006_L0000_normed', 'x1_11_1_2_L0012_L0000_normed', 'x1_11_1_2_L0024_L0000_normed',
        'x1_12_1_1_L0024_L0000_normed', 'x1_12_1_1_L0048_L0000_normed', 'x1_12_1_1_L0096_L0000_normed',
        'x1_12_1_2_L0024_L0000_normed', 'x1_12_1_2_L0048_L0000_normed', 'x1_12_1_2_L0096_L0000_normed',
        'x1_13_1_1_L0003_L0000_normed', 'x1_13_1_1_L0006_L0003_normed', 'x1_13_1_1_L0012_L0006_normed',
        'x1_13_1_1_L0024_L0012_normed', 'x1_13_1_1_L0048_L0024_normed', 'x1_13_1_1_L0096_L0048_normed',
        'x1_13_1_2_L0003_L0000_normed', 'x1_13_1_2_L0006_L0003_normed', 'x1_13_1_2_L0012_L0006_normed',
        'x1_13_1_2_L0024_L0012_normed', 'x1_13_1_2_L0048_L0024_normed', 'x1_13_1_2_L0096_L0048_normed',
        'x8_1_1_0_L0003_L0000_normed', 'x8_1_1_0_L0006_L0003_normed', 'x8_1_1_0_L0012_L0006_normed',
        'x8_1_1_0_L0024_L0012_normed', 'x8_1_1_0_L0048_L0024_normed', 'x8_1_1_0_L0096_L0048_normed',
        'x8_2_1_0_L0003_L0000_normed', 'x8_2_1_0_L0006_L0003_normed', 'x8_2_1_0_L0012_L0006_normed',
        'x8_2_1_0_L0024_L0012_normed', 'x8_2_1_0_L0048_L0024_normed', 'x8_2_1_0_L0096_L0048_normed',
        'x8_3_1_0_L0003_L0000_normed', 'x8_3_1_0_L0006_L0003_normed', 'x8_3_1_0_L0012_L0006_normed',
        'x8_3_1_0_L0024_L0012_normed', 'x8_3_1_0_L0048_L0024_normed', 'x8_3_1_0_L0096_L0048_normed',
        'x8_4_1_0_L0003_L0000_normed', 'x8_4_1_0_L0006_L0003_normed', 'x8_4_1_0_L0012_L0006_normed',
        'x8_4_1_0_L0024_L0012_normed', 'x8_4_1_0_L0048_L0024_normed', 'x8_4_1_0_L0096_L0048_normed',
        'x8_5_1_0_L0003_L0000_normed', 'x8_5_1_0_L0006_L0003_normed', 'x8_5_1_0_L0012_L0006_normed',
        'x8_5_1_0_L0024_L0012_normed', 'x8_5_1_0_L0048_L0024_normed', 'x8_5_1_0_L0096_L0048_normed',
        'x12_1_1_1_L0003_L0000_normed', 'x12_1_2_1_L0003_L0000_normed', 'x12_1_1_2_L0003_L0000_normed',
        'x12_1_2_2_L0003_L0000_normed', 'x12_1_1_1_L0006_L0003_normed', 'x12_1_2_1_L0006_L0003_normed',
        'x12_1_1_2_L0006_L0003_normed', 'x12_1_2_2_L0006_L0003_normed', 'x12_1_1_1_L0012_L0006_normed',
        'x12_1_2_1_L0012_L0006_normed', 'x12_1_1_2_L0012_L0006_normed', 'x12_1_2_2_L0012_L0006_normed',
        'x12_1_1_1_L0024_L0012_normed', 'x12_1_2_1_L0024_L0012_normed', 'x12_1_1_2_L0024_L0012_normed',
        'x12_1_2_2_L0024_L0012_normed', 'x12_1_1_1_L0048_L0024_normed', 'x12_1_2_1_L0048_L0024_normed',
        'x12_1_1_2_L0048_L0024_normed', 'x12_1_2_2_L0048_L0024_normed', 'x13_1_1_0_L0000_normed',
        'x13_2_1_0_L0000_normed', 'x13_3_1_0_L0000_normed', 'x13_4_1_0_L0000_normed',
        'is_five', 'is_ten', 'is_clock', 'week_id', 'session_id', 'minute_id',
    ]
}


def load_model(model_configs):
    path = model_configs['model_path']
    idx = path.rfind('.')
    module_name = path[:idx]
    class_name = path[(idx + 1):]
    module = importlib.import_module(module_name)
    loaded_class = getattr(module, class_name)
    loaded_instance = loaded_class.load_model_from_config(model_configs)
    return loaded_instance


def make_model(model_configs):
    model = load_model(model_configs)
    for name, param in model.named_parameters():
        if param.requires_grad and param.dim() > 1 and 'seq_encoder' not in name:
            nn.init.xavier_uniform(param)
    return model


class StockReg(nn.Module):

    def __init__(self, model_config):

        super(StockReg, self).__init__()
        self.model_config = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stock_model = make_model(model_config)
        self.stock_model = self.stock_model.to(self.device)
        self.stock_model.load_state_dict(torch.load(model_config['pretrain_load_path']))
        for name, param in self.stock_model.named_parameters():
            if 'expert' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x, alias_feats, stock_ids, only_reg=True):

        pred_y = self.stock_model(x, alias_feats, stock_ids, return_tick=False)
        return pred_y
        # if only_reg:
        #     if self.model_config['side'] == 'buy':
        #         return pred_y[:, 0]
        #     else:
        #         return pred_y[:, 1]
        # else:
        #     return pred_y


class FineTrainer(object):

    def __init__(self, train_configs, model_config):

        self.configs = train_configs
        self.model_config = model_config
        self.model_config['side'] = self.configs['side']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.stock_reg = None
        self.optimizer = None
        self.training_data = None
        self.train_dataloader = None
        self.valid_data = None
        self.valid_dataloader = None
        self.test_data = None
        self.test_dataloader = None
        self.criterion = None
        self.stock_reg = None

        self.model_config['batch_size'] = self.configs['batch_size']
        if not os.path.exists(os.path.join(self.configs['model_save_path'], self.configs['train_job_key'])):
            os.mkdir(os.path.join(self.configs['model_save_path'], self.configs['train_job_key']))
            os.mkdir(os.path.join(self.configs['model_save_path'], self.configs['train_job_key'] + '/models/'))
            os.mkdir(os.path.join(self.configs['model_save_path'], self.configs['train_job_key'] + '/logs/'))

        # TODO: save as list of pairs not order-variant dict
        self.skey2id = get_stock_map(train_configs['stock_type'], train_configs['base_day'])
        self.factor_dao = FactorDAO(self.configs['base_path'])

    def count_parameters(self, only_trainable=True):
        """
        returns the total number of parameters used by `m` (only counting
        shared parameters once); if `only_trainable` is True, then only
        includes parameters with `requires_grad = True`
        """
        parameters = self.stock_reg.parameters()
        if only_trainable:
            parameters = list(p for p in parameters if p.requires_grad)
        unique = dict((p.data_ptr(), p) for p in parameters).values()
        return sum(p.numel() for p in unique)

    def setup_trainer(self):

        self.configs['train_end_day'] = self.configs['base_day']
        self.configs['train_start_day'] = n_day_before(self.configs['train_end_day'],
                                                       int(self.configs['back_period']))

        self.configs['test_start_day'] = self.configs['base_day']
        self.configs['test_end_day'] = after_one_month(self.configs['test_start_day'])
        self.configs['valid_start_day'] = self.configs['base_day']
        self.configs['valid_end_day'] = after_one_month(self.configs['test_start_day'])
        self.model_config['stock_number'] = len(self.skey2id)
        self.model_config['train_time_step'] = self.configs['train_time_step']
        self.model_config['alias_time_step'] = self.configs['alias_time_step']
        self.model_config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('train %d-%d' % (self.configs['train_start_day'], self.configs['train_end_day']))
        print('valid %d-%d' % (self.configs['valid_start_day'], self.configs['valid_end_day']))
        print('test %d-%d' % (self.configs['test_start_day'], self.configs['test_end_day']))
        self.stock_reg = StockReg(self.model_config)

        print(self.stock_reg)
        print('number of parameters %d' % self.count_parameters())
        self.stock_reg = self.stock_reg.to(self.device)

        export_path = os.path.join(self.configs['model_save_path'], self.configs['train_job_key']
                                   + '/models/predictions_{start}_{end}.pkl'
                                   .format(start=self.configs['test_start_day'],
                                           end=self.configs['test_end_day']))
        if os.path.exists(export_path) and (not self.configs['refresh_train']):
            print('already trained...')
            return False

        # TODO: change back to SGD with momentum
        params_to_update = []
        for name, param in self.stock_reg.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
        num_steps = self.configs['max_train_step']

        # self.optimizer = apex.optimizers.FusedAdam(params=params_to_update,
        #                                            lr=self.configs['optimizer']['lr'],
        #                                            weight_decay=self.configs['optimizer']['weight-decay'])

        self.optimizer = torch.optim.AdamW(params_to_update,
                                           lr=self.configs['optimizer']['lr'],
                                           betas=(0.9, 0.999),
                                           weight_decay=self.configs['optimizer']['weight-decay'])

        self.training_data = DaySkeyMapDataset(self.configs, self.model_config, 'train')
        self.train_dataloader = DataLoader(self.training_data,
                                           batch_size=1,
                                           shuffle=True,
                                           pin_memory=True,
                                           prefetch_factor=2,
                                           num_workers=2)

        self.criterion = FixMeanMultiTaskLoss(reg_pos=[0, 1, 2, 3, 4, 5], clf_pos=[6, 7, 8, 9, 10, 11],
                                              multi_clf_pos=[]).cuda()
        return True

    def train(self):

        train_loss_cn = 0.0
        train_mse_loss = 0.0
        self.stock_reg.train()
        start = time.time()
        step_i = 0

        model_time = 0.0
        total_task_losses = [0.0] * (self.model_config['task_number'] + 6)
        epoch = 0
        while True:

            all_train_mse, all_train_mae, all_cn = 0.0, 0.0, 0.0
            # FOR each EPOCHS
            et1 = time.time()
            for batch_data in self.train_dataloader:

                step_i += 1
                t1 = time.time()
                self.optimizer.zero_grad()

                feats = batch_data[0][0].cuda(non_blocking=True)
                alias_feats = batch_data[1][0].cuda(non_blocking=True)
                gts = batch_data[2][0].cuda(non_blocking=True)
                stock_ids = batch_data[3][0].cuda(non_blocking=True)
                preds = self.stock_reg(feats, alias_feats, stock_ids, only_reg=False)

                loss, task_losses = self.criterion(preds, gts)
                for i, task_loss in enumerate(task_losses):
                    total_task_losses[i] += task_loss.item()

                loss.backward()
                self.optimizer.step()
                # self.lr_scheduler.step(self.lr_scheduler.last_epoch + 1)

                t2 = time.time()
                model_time += (t2 - t1)

                train_mse_loss += loss.item()
                train_loss_cn += 1

                all_train_mse += loss.item()
                all_cn += 1

                if step_i % self.configs['log_step'] == 0:
                    elapsed = time.time() - start
                    print("Epoch Step: %d Comb Loss: %f time %f model-time %f lr %f" %
                          (step_i, train_mse_loss / train_loss_cn, elapsed,
                           model_time, self.optimizer.param_groups[0]['lr']))
                    for loss_val in total_task_losses:
                        print(loss_val / train_loss_cn)
                    start = time.time()
                    train_mse_loss = 0.0
                    train_loss_cn = 0.0
                    model_time = 0.0
                    total_task_losses = [0.0] * (self.model_config['task_number'] + 6)

                if step_i > int(self.configs['max_train_step']):
                    break

            et2 = time.time()
            print('time per epoch %f' % (et2 - et1))

            if step_i > int(self.configs['max_train_step']):
                print('finish training')
                break

            # after epoch to do validation
            epoch += 1

        torch.save(self.stock_reg.state_dict(), os.path.join(self.configs['model_save_path'],
                                                             self.configs[
                                                                 'train_job_key']
                                                             + '/models/best_model.ckpt'))

    def train_model(self):

        t1 = time.time()
        self.train()
        t2 = time.time()
        self.export_test()
        t3 = time.time()
        print('fine tune time %f and alpha export time %f' % (t2 - t1, t3 - t2))

    # TODO: normalizer read path need change
    def export_test(self, load_model=None):

        if load_model:
            self.stock_reg = load_model
            self.stock_reg.eval()

        elif self.configs['export_best']:
            print('use the best %s .....' % os.path.join(self.configs['model_save_path'],
                                                         self.configs['train_job_key']
                                                         + '/models/best_model.ckpt'))
            self.stock_reg.load_state_dict(torch.load(os.path.join(self.configs['model_save_path'],
                                                                   self.configs['train_job_key']
                                                                   + '/models/best_model.ckpt')))
            self.stock_reg.eval()

        factor_group = self.configs['factor_group']
        version = self.configs['factor_version']

        total_mae, total_mse, loss_cn = 0.0, 0.0, 0.0
        print('detail test.....')

        buy_prediction_df = []
        sell_prediction_df = []

        self.stock_reg.eval()

        norm_df = self.factor_dao.read_factor_normalizer_by_skey_and_day(factor_group='LabelTm',
                                                                         normalizer_name='auto_norm',
                                                                         day=None, skey=self.configs['target_stock'][0],
                                                                         version='v1')
        for day_i in iter_time_range(self.configs['test_start_day'], self.configs['test_end_day']):

            print(day_i)

            factor_df = self.factor_dao.read_factor_by_skey_and_day(factor_group=factor_group,
                                                                    day=day_i, skey=self.configs['target_stock'][0],
                                                                    version=version)
            raw_path = '/b/sta_feat_eq_cn/sta_feat_1_2_l2/LabelTm/{day}/{skey}.parquet' \
                .format(day=day_i, skey=self.configs['target_stock'][0])
            if (factor_df is not None) and (os.path.exists(raw_path)) and (norm_df is not None):
                raw_df = pd.read_parquet(raw_path)
                sell_std = norm_df.loc[norm_df.date == day_i].iloc[0]['sellRetFuture90_std']
                buy_std = norm_df.loc[norm_df.date == day_i].iloc[0]['buyRetFuture90_std']
                ordering_set = set(raw_df.ordering.unique())
                raw_df = raw_df.set_index('ordering')
                factor_df = factor_df.loc[factor_df.ordering.isin(ordering_set)]

                for batch_o, batch_t, batch_feats, batch_alias, batch_stocks, batch_gts \
                        in self.iter_factors_by_day_skey(factor_df, self.configs['train_factors'],
                                                         self.configs['predict_factors'],
                                                         self.configs['train_time_step'],
                                                         self.configs['alias_time_step']):

                    feats = torch.from_numpy(batch_feats).float().to(self.device)
                    alias_feats = torch.from_numpy(batch_alias).float().to(self.device)
                    stock_ids = torch.from_numpy(batch_stocks).int().to(self.device)
                    gts = torch.from_numpy(batch_gts).float().to(self.device)
                    if feats.shape[0] < 2:
                        continue

                    # make predictions......
                    preds = self.stock_reg(feats, alias_feats, stock_ids)
                    loss_cn += 1

                    # export prediction results
                    pred_vals = preds.detach().cpu().numpy()
                    print(pred_vals.shape)

                    for i in range(len(pred_vals)):
                        pred_ent = dict()
                        pred_ent['time'] = batch_t[i]
                        pred_ent['ordering'] = batch_o[i]
                        pred_ent['minute'] = time_to_minute(batch_t[i])
                        pred_ent['yHatBuy'] = pred_vals[i][0] * buy_std
                        pred_ent['date'] = day_i
                        pred_ent['skey'] = self.configs['target_stock'][0]
                        pred_ent['label'] = raw_df.loc[batch_o[i]]['buyRetFuture90']
                        buy_prediction_df.append(pred_ent)

                        pred_ent = dict()
                        pred_ent['time'] = batch_t[i]
                        pred_ent['ordering'] = batch_o[i]
                        pred_ent['minute'] = time_to_minute(batch_t[i])
                        pred_ent['yHatSell'] = pred_vals[i][1] * sell_std
                        pred_ent['date'] = day_i
                        pred_ent['skey'] = self.configs['target_stock'][0]
                        pred_ent['label'] = raw_df.loc[batch_o[i]]['sellRetFuture90']
                        sell_prediction_df.append(pred_ent)

                        # if self.configs['side'] == 'buy':
                        #     pred_ent['yHatBuy'] = pred_vals[i][0] * buy_std
                        #     pred_ent['label'] = raw_df.loc[batch_o[i]]['buyRetFuture90']
                        #
                        # else:
                        #     pred_ent['yHatSell'] = pred_vals[i][0] * sell_std
                        #     pred_ent['label'] = raw_df.loc[batch_o[i]]['sellRetFuture90']

        buy_prediction_df = pd.DataFrame(buy_prediction_df)
        sell_prediction_df = pd.DataFrame(sell_prediction_df)

        # top3_ret = list()
        #
        # for grp, sub_df in prediction_df.groupby(by=['date']):
        #     labels = sub_df.label.tolist()
        #     if self.configs['side'] == 'buy':
        #         hats = sub_df.yHatBuy.tolist()
        #     else:
        #         hats = sub_df.yHatSell.tolist()
        #
        #     topk = int(len(hats) * 0.05)
        #     top3_ret.append(np.mean(np.asarray(labels)[np.asarray(hats).argsort()[-topk:][::-1]]))
        #
        # print('average top gain %f ' % np.mean(top3_ret))
        buy_export_path = os.path.join(self.configs['model_save_path'], self.configs['train_job_key']
                                       + '/models/predictions_buy_{start}_{end}.pkl'
                                       .format(start=self.configs['test_start_day'],
                                               end=self.configs['test_end_day']))
        buy_prediction_df.to_pickle(buy_export_path)

        sell_export_path = os.path.join(self.configs['model_save_path'], self.configs['train_job_key']
                                        + '/models/predictions_sell_{start}_{end}.pkl'
                                        .format(start=self.configs['test_start_day'],
                                                end=self.configs['test_end_day']))
        sell_prediction_df.to_pickle(sell_export_path)

        print('exported %s' % buy_export_path)

    def iter_factors_by_day_skey(self, factor_df, train_factors, predict_factors,
                                 train_time_step=None, alias_time_step=None,
                                 all_time=False):

        factor_df = factor_df.reset_index()
        batch_t = []
        batch_o = []
        batch_feats = []
        batch_alias = []
        batch_stocks = []
        batch_gts = []
        if all_time:
            start_idx = 0
        else:
            start_idx = max(train_time_step - 1, alias_time_step - 1, 16)
        orders = factor_df['ordering'].tolist()
        times = factor_df['time'].tolist()
        spans = factor_df[train_factors].to_numpy()
        gts = factor_df[predict_factors].to_numpy()
        aliases = factor_df[self.configs['alias_factors']].to_numpy()
        stocks = factor_df[['allZT', 'hasZT', 'isZT', 'allDT',
                            'hasDT', 'isDT', 'isST', 'SW1_codes', 'SW2_codes', 'SW3_codes',
                            'marketValue', 'marketShares']].to_numpy()

        for i in range(start_idx, len(factor_df)):

            o = orders[i]
            t = times[i]
            span = spans[i - train_time_step + 1: i + 1, :]
            alias = aliases[i - alias_time_step + 1: i + 1, :]
            stock_embeds = stocks[i, :]
            gt = gts[i, :]
            if not all_time:
                if 112830 < t < 120000 or 130000 < t < 130130:
                    continue
            span = np.nan_to_num(span, nan=0.0, posinf=0.0, neginf=0.0)
            alias = np.nan_to_num(alias, nan=0.0, posinf=0.0, neginf=0.0)
            stock_embeds = np.nan_to_num(stock_embeds, nan=0.0, posinf=0.0, neginf=0.0)
            if (not np.isnan(span).any()) and (not np.isnan(alias).any()):
                if all_time or (not np.isnan(gt).any()):
                    if alias.shape[0] < alias_time_step:
                        pads = np.zeros((alias_time_step - alias.shape[0], alias.shape[1]))
                        alias = np.concatenate([pads, alias], axis=0)
                    batch_o.append(o)
                    batch_t.append(t)
                    batch_feats.append(span)
                    batch_alias.append(alias)
                    batch_stocks.append(stock_embeds)
                    batch_gts.append(gt)
        if len(batch_feats) > 0:
            yield batch_o, batch_t, np.asarray(batch_feats), np.asarray(batch_alias), np.asarray(
                batch_stocks), np.asarray(batch_gts)


# [1601958] 20200201 ['buyRetFuture25_normed', 'buyRetFuture5_normed', 'buyRetFuture85_normed', 'buyRetFuture25Top',
# 'buyRetFuture5Top', 'buyRetFuture85Top'] plus_model_deep_tcn_NEW_REDUCE_V2_1601958_buy_20200201 buy
# 343869 343881
def batch_run():
    sides = ['both']
    day_str = sys.argv[1]
    day_i = int("".join(day_str[:day_str.find('T')].split('-')))
    print(day_i)
    # TODO: change day range tracking here
    day_ranges = [
        (day_i, after_one_month(day_i))
    ]

    feat_groups = ['MBD_FINAL', 'SNAPSHOT_FINAL']
    # 6000
    stock_groups = ['HIGH', 'MID_HIGH', 'MID_LOW', 'LOW']
    model_names = ['combo']

    dist_tasks = []
    for feat_grp in feat_groups:
        for side in sides:
            for range_i in day_ranges:

                all_finetune_skeys = set()

                for stock_grp in stock_groups:
                    stocks = set(get_stock_map(stock_grp, period=range_i[0]))
                    stocks = list(stocks)
                    all_finetune_skeys = all_finetune_skeys | set(stocks)
                    for skey in stocks:
                        for model_name in model_names:
                            dist_tasks.append([side, range_i, skey, feat_grp, stock_grp, model_name])
                print(range_i, len(all_finetune_skeys))

    unit_tasks = [t for i, t in enumerate(dist_tasks) if i % total_worker == work_id]
    print('allocate %d tasks among %d' % (len(unit_tasks), len(dist_tasks)))
    t1 = time.time()

    for task in unit_tasks:
        try:
            if task[3] == 'MBD_FINAL':
                with open(
                        '/b/home/pengfei_ji/airflow/dags/sta_17001/scripts/weekly_workflow/train_configs/mbd_model.yaml') as f:
                    model_configs = yaml.load(f, Loader=SafeLoader)
            else:
                with open(
                        '/b/home/pengfei_ji/airflow/dags/sta_17001/scripts/weekly_workflow/train_configs/snap_model.yaml') as f:
                    model_configs = yaml.load(f, Loader=SafeLoader)

            with open(
                    '/b/home/pengfei_ji/airflow/dags/sta_17001/scripts/weekly_workflow/train_configs/finetune_config.yaml') as f:
                base_configs = yaml.load(f, Loader=SafeLoader)

            base_configs['target_stock'] = [task[2]]
            base_configs['base_day'] = task[1][0]
            base_configs['train_factors'] = feat_map[task[3]]
            base_configs['stock_type'] = task[4]
            base_configs['side'] = task[0]
            model_configs['model_name'] = 'both'
            base_configs['factor_version'] = 'v1'
            base_configs['predict_factors'] = ['buyRetFuture90_normed', 'sellRetFuture90_normed',
                                               'buyRetFuture30_normed', 'sellRetFuture30_normed',
                                               'buyRetFuture300_normed', 'sellRetFuture300_normed',

                                               'buyRetFuture90Top', 'sellRetFuture90Top',
                                               'buyRetFuture30Top', 'sellRetFuture30Top',
                                               'buyRetFuture300Top', 'sellRetFuture300Top']

            base_configs['train_job_key'] = model_configs['model_name'] + "_{model}_{grp}_{skey}_{side}_{start}".format(
                skey=base_configs['target_stock'][0],
                side=task[0],
                start=task[1][0],
                grp=task[3],
                model=task[5]
            )

            print(base_configs['target_stock'], base_configs['base_day'], base_configs['predict_factors'],
                  base_configs['train_job_key'], base_configs['side'])

            model_configs[
                'pretrain_load_path'] = '/b/home/pengfei_ji/production_models/pretrain_model_prod_{model}_{feat}_{side}_{price}_{start}/models/' \
                .format(side=task[0], price=task[4], start=int(task[1][0]), end=int(task[1][1]), feat=task[3],
                        model=task[5])
            model_files = [model_configs['pretrain_load_path'] + f for f in listdir(model_configs['pretrain_load_path'])
                           if isfile(join(model_configs['pretrain_load_path'], f)) if
                           'best' not in f]
            model_files.sort(key=os.path.getatime)
            model_configs['pretrain_load_path'] = model_files[-1]
            base_configs['train_factors'] = feat_map[task[3]]
            base_configs['alias_factors'] = feat_map[task[3]][:-6]
            model_configs['seq_encoder']['num_input'] = len(base_configs['alias_factors'])
            model_configs['wide_encoder']['feat_dim'] = len(base_configs['train_factors']) - 3
            model_configs['task_number'] = len(base_configs['predict_factors'])
            model_configs['stock_class'] = len(get_stock_map(base_configs['stock_type'], base_configs['base_day']))

            print("load pretrained model %s" % model_configs['pretrain_load_path'])
            print(base_configs['factor_version'])
            base_configs['side'] = task[0]

            sta_basic_trainer = FineTrainer(base_configs, model_configs)
            if sta_basic_trainer.setup_trainer():
                sta_basic_trainer.train_model()
            print('----' * 10)

        except Exception:
            print('%s experimental error....' % task)
            print(traceback.format_exc())
            print('----' * 10)
            continue
    t2 = time.time()
    print(t2 - t1)


# 997 1603712
if __name__ == '__main__':
    import argparse
    import yaml
    from yaml.loader import SafeLoader
    import traceback

    array_id = int(get_slurm_env("SLURM_ARRAY_TASK_ID"))
    array_size = int(get_slurm_env("SLURM_ARRAY_TASK_COUNT"))
    proc_id = int(get_slurm_env("SLURM_PROCID"))
    task_size = int(get_slurm_env("SLURM_NTASKS"))
    work_id = array_id * task_size + proc_id
    total_worker = array_size * task_size
    batch_run()
