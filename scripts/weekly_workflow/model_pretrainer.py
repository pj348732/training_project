import sys

sys.path.insert(0, '/b/home/pengfei_ji/airflow/dags/sta_17001/scripts/workflow_utils/')
sys.path.insert(0, '/b/home/pengfei_ji/airflow/dags/sta_17001/scripts/weekly_workflow/')
import random
from factor_loader import DaySkeyMapDataset, get_stock_map
from common_utils import get_slurm_env, after_one_month
import torch
import pytorch_warmup as warmup
from torch.utils.data import DataLoader
import time
import os
from tqdm import *
import torch.nn as nn
import importlib
from torch.utils.tensorboard import SummaryWriter
import psutil
import apex
from factor_loader import NewAlphaDataset, FixMeanMultiTaskLoss
import numpy as np
from factor_dao import FactorDAO

torch.backends.cudnn.benchmark = True


def set_debug_apis(state):
    torch.autograd.profiler.profile(enabled=state)
    torch.autograd.profiler.emit_nvtx(enabled=state)
    torch.autograd.set_detect_anomaly(mode=state)


# Then in training code before the train loop
set_debug_apis(state=False)

# TODO: also with none
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


def seed_worker(worker_id):
    work_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(work_seed)
    random.seed(work_seed)


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


class PreTrainer(object):

    def __init__(self, train_config, model_config, device_id, model_name):

        self.train_config = train_config
        self.model_config = model_config
        self.device_id = device_id
        self.ic_skeys = get_stock_map(train_config['stock_type'], train_config['valid_start_day'])
        # get the proper time steps to train
        self.factor_dao = FactorDAO(train_config['base_path'])
        self.model_config['device'] = device_id
        self.model_config['batch_size'] = self.train_config['batch_size']
        print(self.device_id)
        torch.cuda.set_device(device_id)
        self.stock_model = make_model(model_config)
        self.stock_model = self.stock_model.cuda(device_id)

        print(self.stock_model)
        print("the number of parameters %d " % self.count_parameters())

        print('load train.....')

        self.pretrain_dataset = NewAlphaDataset(train_config['train_start_day'], train_config['train_end_day'],
                                                train_config['base_path'], train_config['factor_group'],
                                                train_config['train_factors'],
                                                train_config['alias_factors'], train_config['predict_factors'],
                                                stocks=list(self.ic_skeys),
                                                factor_version=train_config['factor_version'],
                                                train_time_step=train_config['train_time_step'],
                                                alias_time_step=train_config['alias_time_step'],
                                                stock_type=train_config['stock_type'])
        self.day_skey_pairs = self.pretrain_dataset.day_skey_pairs
        print('load valid.....')
        self.valid_data = DaySkeyMapDataset(self.train_config, self.model_config, 'valid')
        self.valid_dataloader = self.valid_data
        self.criterion = FixMeanMultiTaskLoss(reg_pos=[0, 1, 2, 3, 4, 5], clf_pos=[6, 7, 8, 9, 10, 11],
                                              multi_clf_pos=[]).cuda(self.device_id)

        params_to_update = []
        for name, param in self.stock_model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

        self.optimizer = apex.optimizers.FusedAdam(params=params_to_update,
                                                   lr=train_config['optimizer']['lr'],
                                                   weight_decay=train_config['optimizer']['weight-decay'])

        # self.optimizer = torch.optim.AdamW(self.stock_model.parameters(),
        #                                    lr=train_config['optimizer']['lr'],
        #                                    betas=(0.9, 0.999),
        #                                    weight_decay=train_config['optimizer']['weight-decay'])
        print(self.train_config['train_job_key'])
        self.train_config['max_train_step'] = 8000 # min(1000000, int(int(450000 / 20000) * len(self.day_skey_pairs)))
        num_steps = self.train_config['max_train_step']
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_steps,
                                                                       eta_min=0.00005)

        print(len(self.day_skey_pairs), self.train_config['max_train_step'], len(self.ic_skeys))

        self.warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)

        self.train_dataloader = DataLoader(self.pretrain_dataset,
                                           batch_size=1,
                                           shuffle=True,
                                           pin_memory=True,
                                           prefetch_factor=2,
                                           num_workers=8)

        if not os.path.exists(os.path.join(self.train_config['model_save_path'], self.train_config['train_job_key'])):
            os.mkdir(os.path.join(self.train_config['model_save_path'], self.train_config['train_job_key']))
            os.mkdir(
                os.path.join(self.train_config['model_save_path'], self.train_config['train_job_key'] + '/models/'))
            os.mkdir(os.path.join(self.train_config['model_save_path'], self.train_config['train_job_key'] + '/logs/'))

        self.board_writer = SummaryWriter(os.path.join(self.train_config['model_save_path'],
                                                       self.train_config['train_job_key']))

    def count_parameters(self, only_trainable=True):
        """
        returns the total number of parameters used by `m` (only counting
        shared parameters once); if `only_trainable` is True, then only
        includes parameters with `requires_grad = True`
        """
        parameters = self.stock_model.parameters()
        if only_trainable:
            parameters = list(p for p in parameters if p.requires_grad)
        unique = dict((p.data_ptr(), p) for p in parameters).values()
        return sum(p.numel() for p in unique)

    def pretrain(self):

        # final_path = os.path.join(self.train_config['model_save_path'],
        #                           self.train_config['train_job_key'] + '/models/') \
        #              + 'model_' + str(self.train_config['max_train_step'] + 1) + '.ckpt'
        # if os.path.exists(final_path):
        #     print('already %s trained' % self.train_config['train_job_key'])
        #     return

        train_loss_cn = 0.0
        train_mse_loss = 0.0

        self.stock_model.train()
        start = time.time()
        step_i = 0
        model_time = 0.0
        data_time = 0.0
        back_time = 0.0

        best_val = 1999
        early_stop_threshold = 50
        stop_cn = 0
        total_task_losses = [0.0] * (self.model_config['task_number'] + 6)
        while True:

            for batch_data in self.train_dataloader:
                if len(batch_data) == 0:
                    continue

                step_i += 1
                t1 = time.time()
                # self.optimizer.zero_grad()
                for param in self.stock_model.parameters():
                    param.grad = None

                feats = batch_data[0][0].cuda(non_blocking=True)
                alias_feats = batch_data[1][0].cuda(non_blocking=True)
                gts = batch_data[2][0].cuda(non_blocking=True)
                stock_ids = batch_data[3][0].cuda(non_blocking=True)

                t2 = time.time()
                data_time += (t2 - t1)
                preds = self.stock_model(feats, alias_feats, stock_ids, return_tick=False)
                loss, task_losses = self.criterion(preds, gts)
                for i, task_loss in enumerate(task_losses):
                    total_task_losses[i] += task_loss.item()
                t3 = time.time()
                model_time += (t3 - t2)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step(self.lr_scheduler.last_epoch + 1)
                self.warmup_scheduler.dampen()
                t4 = time.time()
                train_mse_loss += loss.item()
                train_loss_cn += 1
                back_time += (t4 - t3)

                if step_i % self.train_config['log_step'] == 0:
                    elapsed = time.time() - start

                    print(
                        "Epoch Step: %d Combined Loss: %f time %f model-time %f data-time %f back-time %f "
                        "lr %f" % (step_i, train_mse_loss / train_loss_cn, elapsed,
                                   model_time, data_time, back_time, self.optimizer.param_groups[0]['lr']),
                        flush=True)
                    print('task losses.....')
                    process = psutil.Process(os.getpid())

                    for t_loss in total_task_losses:
                        print(t_loss / train_loss_cn)

                    print("memory usage %f" % (process.memory_info().rss / (1024 * 1024)))
                    self.board_writer.add_scalar("Combine Loss/train", train_mse_loss / train_loss_cn, step_i)
                    start = time.time()
                    train_mse_loss = 0.0
                    train_loss_cn = 0.0
                    model_time = 0.0
                    back_time = 0.0
                    data_time = 0.0
                    total_task_losses = [0.0] * (self.model_config['task_number'] + 6)

                if step_i > int(self.train_config['max_train_step']):
                    break

                if step_i % self.train_config['save_step'] == 0 or step_i == 10000:
                    print('save current model')
                    torch.save(self.stock_model.state_dict(), os.path.join(self.train_config['model_save_path'],
                                                                           self.train_config['train_job_key']
                                                                           + '/models/model_' + str(step_i) + '.ckpt'))

                if step_i % self.train_config['valid_step'] == 0 or step_i == 10000:
                    v_reg_loss = self.valid()
                    print("epoch %d validate reg %f" % (step_i, v_reg_loss))
                    self.board_writer.add_scalar("Reg Loss/valid", v_reg_loss, step_i)
                    self.board_writer.flush()

                    if v_reg_loss <= best_val:
                        print('save new best model at %d' % step_i)
                        torch.save(self.stock_model.state_dict(), os.path.join(self.train_config['model_save_path'],
                                                                               self.train_config[
                                                                                   'train_job_key']
                                                                               + '/models/best_model.ckpt'))
                        best_val = v_reg_loss
                        stop_cn = 0
                    else:
                        stop_cn += 1
                        if stop_cn > early_stop_threshold:
                            print('have not improved the best val %f for %d epochs, stop early' % (best_val, stop_cn))
                            self.stock_model.train()
                            break
                    self.stock_model.train()

            if step_i > int(self.train_config['max_train_step']):
                # SAVE LAST STEPS
                torch.save(self.stock_model.state_dict(), os.path.join(self.train_config['model_save_path'],
                                                                       self.train_config['train_job_key']
                                                                       + '/models/model_' + str(step_i) + '.ckpt'))
                print('finish training')
                break

        self.board_writer.close()

    def valid(self):
        self.stock_model.eval()
        total_comb, loss_cn = 0.0, 0.0
        total_task_losses = [0.0] * (self.model_config['task_number'] + 6)

        with torch.no_grad():
            for batch_data in tqdm(self.valid_dataloader):
                feats = torch.from_numpy(batch_data[0]).float().cuda(non_blocking=True)
                alias_feats = torch.from_numpy(batch_data[1]).float().cuda(non_blocking=True)
                gts = torch.from_numpy(batch_data[2]).float().cuda(non_blocking=True)
                stock_ids = torch.from_numpy(batch_data[3]).cuda(non_blocking=True)

                preds = self.stock_model(feats, alias_feats, stock_ids, return_tick=False)
                loss, task_losses = self.criterion(preds, gts)
                total_comb += loss.item()
                for i, task_loss in enumerate(task_losses):
                    total_task_losses[i] += task_loss.item()

                loss_cn += 1

        if loss_cn > 0:
            for task_loss in total_task_losses:
                print(task_loss / loss_cn)
            return total_comb / loss_cn
        else:
            return 0.0


if __name__ == '__main__':

    import argparse
    import yaml
    from yaml.loader import SafeLoader

    array_id = int(get_slurm_env("SLURM_ARRAY_TASK_ID"))
    array_size = int(get_slurm_env("SLURM_ARRAY_TASK_COUNT"))
    proc_id = int(get_slurm_env("SLURM_PROCID"))
    task_size = int(get_slurm_env("SLURM_NTASKS"))
    work_id = array_id * task_size + proc_id
    total_worker = array_size * task_size
    dist_tasks = []

    # 64
    sides = ['both']
    feat_groups = ['MBD_FINAL', 'SNAPSHOT_FINAL']
    day_str = sys.argv[1]
    day_i = int("".join(day_str[:day_str.find('T')].split('-')))
    # day_i is the split-day
    print(day_i)
    # TODO: change day range tracking here
    day_ranges = [
        (day_i, after_one_month(day_i))
    ]
    price_groups = ['HIGH', 'MID_HIGH', 'MID_LOW', 'LOW']

    model_names = ['combo']
    for side in sides:
        for range_i in day_ranges:
            for price_grp in price_groups:
                for feat_grp in feat_groups:
                    for model_name in model_names:
                        dist_tasks.append([side, range_i, price_grp, feat_grp, model_name])

    unit_tasks = [t for i, t in enumerate(dist_tasks) if i % total_worker == work_id]
    print('allocate %d tasks among %d' % (len(unit_tasks), len(dist_tasks)))

    for task in unit_tasks:
        train_start = time.time()

        if task[3] == 'MBD_FINAL':
            with open(
                    '/b/home/pengfei_ji/airflow/dags/sta_17001/scripts/weekly_workflow/train_configs/mbd_model.yaml') as f:
                model_configs = yaml.load(f, Loader=SafeLoader)
        else:
            with open(
                    '/b/home/pengfei_ji/airflow/dags/sta_17001/scripts/weekly_workflow/train_configs/snap_model.yaml') as f:
                model_configs = yaml.load(f, Loader=SafeLoader)

        with open(
                '/b/home/pengfei_ji/airflow/dags/sta_17001/scripts/weekly_workflow/train_configs/pretrain_config.yaml') as f:
            train_configs = yaml.load(f, Loader=SafeLoader)

        train_configs['train_start_day'] = task[1][0] - 10000
        train_configs['train_end_day'] = task[1][0]
        train_configs['valid_start_day'] = task[1][0]
        train_configs['valid_end_day'] = task[1][1]
        train_configs['stock_type'] = task[2]

        train_configs['factor_version'] = 'v1'
        train_configs['predict_factors'] = ['buyRetFuture90_normed', 'sellRetFuture90_normed',
                                            'buyRetFuture30_normed', 'sellRetFuture30_normed',
                                            'buyRetFuture300_normed', 'sellRetFuture300_normed',

                                            'buyRetFuture90Top', 'sellRetFuture90Top',
                                            'buyRetFuture30Top', 'sellRetFuture30Top',
                                            'buyRetFuture300Top', 'sellRetFuture300Top']

        train_configs['train_job_key'] = 'pretrain_model_prod_{model}_{feat}_{side}_{price}_{start}' \
            .format(side=task[0], price=task[2], start=task[1][0], feat=task[3], model=task[4])
        train_configs['train_factors'] = feat_map[task[3]]
        train_configs['alias_factors'] = feat_map[task[3]][:-6]
        model_configs['seq_encoder']['num_input'] = len(train_configs['alias_factors'])
        model_configs['wide_encoder']['feat_dim'] = len(train_configs['train_factors']) - 3
        model_configs['task_number'] = len(train_configs['predict_factors'])
        model_configs['stock_class'] = len(get_stock_map(train_configs['stock_type'], train_configs['valid_start_day']))
        print(train_configs['train_job_key'], train_configs['predict_factors'], train_configs['factor_version'])
        train_configs['side'] = task[0]
        trainer = PreTrainer(train_configs, model_configs, 0, task[4])
        trainer.pretrain()
        train_end = time.time()
        print('total training time is %f' % (train_end - train_start))
