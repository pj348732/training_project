import sys
import os
import numpy as np
# airflow tasks test daily_workflow compute_label_normalizer 2020-11-02
import pandas as pd

sys.path.insert(0, '/b/home/pengfei_ji/airflow/dags/sta_17001/scripts/workflow_utils/')
from common_utils import get_slurm_env, get_all_stocks_by_day, get_trade_days, get_weekday, time_to_minute, \
    get_session_id
from factor_dao import FactorDAO
from tqdm import *
import re

top_ratio_ce = 0.1



class FactorAgg(object):

    def __init__(self, target_day):
        self.target_day = target_day
        self.factor_dao = FactorDAO('/b/home/pengfei_ji/factor_dbs/')
        self.join_keys = ['date', 'skey', 'time', 'ordering']
        self.trade_days = get_trade_days()
        self.feat_cols = [
            'x1_1_2_0_L0000_normed', 'x1_5_1_1_L0003_L0000_normed', 'x1_5_1_1_L0006_L0003_normed',
            'x1_5_1_1_L0012_L0006_normed', 'x1_5_1_2_L0003_L0000_normed', 'x1_5_1_2_L0006_L0003_normed',
            'x1_5_1_2_L0012_L0006_normed', 'x1_6_1_0_L0096_L0000_normed', 'x1_6_2_0_L0096_L0000_normed',
            'x1_7_1_1_L0000_normed',
            'x1_7_2_1_L0000_normed', 'x1_7_3_1_L0000_normed', 'x1_7_4_1_L0000_normed', 'x1_7_5_1_L0000_normed',
            'x1_7_6_1_L0000_normed', 'x1_7_7_1_L0000_normed', 'x1_7_8_1_L0000_normed', 'x1_8_1_1_L0000_normed',
            'x1_8_1_2_L0000_normed', 'x1_9_1_0_L0000_normed', 'x1_10_1_2_L0000_normed', 'x1_10_1_1_L0000_normed',
            'x1_11_1_2_L0006_L0000_normed', 'x1_11_1_2_L0024_L0000_normed', 'x1_11_1_1_L0003_L0000_normed',
            'x1_11_1_1_L0012_L0000_normed', 'x1_11_1_2_L0012_L0000_normed', 'x1_11_1_1_L0024_L0000_normed',
            'x1_11_1_1_L0006_L0000_normed', 'x1_11_1_2_L0003_L0000_normed', 'x1_12_1_1_L0024_L0000_normed',
            'x1_12_1_2_L0048_L0000_normed', 'x1_12_1_2_L0096_L0000_normed', 'x1_12_1_1_L0048_L0000_normed',
            'x1_12_1_2_L0024_L0000_normed', 'x1_12_1_1_L0096_L0000_normed', 'x2_1_1_1_L0024_L0012_normed',
            'x2_1_1_2_L0006_L0003_normed', 'x2_1_1_2_L0024_L0012_normed', 'x2_1_1_1_L0012_L0006_normed',
            'x2_1_1_2_L0003_L0000_normed', 'x2_1_1_1_L0006_L0003_normed', 'x2_1_1_2_L0096_L0048_normed',
            'x2_1_1_2_L0048_L0024_normed', 'x2_1_1_1_L0003_L0000_normed', 'x2_1_1_1_L0096_L0048_normed',
            'x2_1_1_1_L0048_L0024_normed', 'x2_1_1_2_L0012_L0006_normed', 'x2_1_2_2_L0012_L0006_normed',
            'x2_1_2_1_L0003_L0000_normed', 'x2_1_2_2_L0006_L0003_normed', 'x2_1_2_1_L0006_L0003_normed',
            'x2_1_2_2_L0003_L0000_normed', 'x2_1_2_1_L0012_L0006_normed', 'x2_1_3_2_L0012_L0006_normed',
            'x2_1_3_1_L0024_L0012_normed', 'x2_1_3_2_L0024_L0012_normed', 'x2_1_3_1_L0003_L0000_normed',
            'x2_1_3_2_L0003_L0000_normed', 'x2_1_3_1_L0012_L0006_normed', 'x2_1_3_2_L0048_L0024_normed',
            'x2_1_3_1_L0048_L0024_normed', 'x2_1_3_1_L0006_L0003_normed', 'x2_1_3_2_L0006_L0003_normed',
            'x2_1_4_1_L0012_L0006_normed', 'x2_1_4_2_L0003_L0000_normed', 'x2_1_4_1_L0006_L0003_normed',
            'x2_1_4_1_L0024_L0012_normed', 'x2_1_4_2_L0048_L0024_normed', 'x2_1_4_2_L0024_L0012_normed',
            'x2_1_4_2_L0006_L0003_normed', 'x2_1_4_1_L0048_L0024_normed', 'x2_1_4_1_L0003_L0000_normed',
            'x2_1_4_2_L0012_L0006_normed', 'x2_1_5_2_L0024_L0012_normed', 'x2_1_5_1_L0012_L0006_normed',
            'x2_1_5_2_L0048_L0024_normed', 'x2_1_5_1_L0048_L0024_normed', 'x2_1_5_2_L0006_L0003_normed',
            'x2_1_5_1_L0006_L0003_normed', 'x2_1_5_2_L0012_L0006_normed', 'x2_1_5_1_L0003_L0000_normed',
            'x2_1_5_2_L0003_L0000_normed', 'x2_1_5_1_L0024_L0012_normed', 'x3_2_1_1_L0003_L0000_normed',
            'x3_2_1_2_L0003_L0000_normed', 'x3_2_1_2_L0012_L0006_normed', 'x3_2_1_1_L0012_L0006_normed',
            'x3_2_1_1_L0006_L0003_normed', 'x3_2_1_2_L0006_L0003_normed', 'x3_2_1_1_L0024_L0012_normed',
            'x3_2_1_2_L0024_L0012_normed', 'x3_2_2_1_L0012_L0006_normed', 'x3_2_2_2_L0003_L0000_normed',
            'x3_2_2_2_L0012_L0006_normed', 'x3_2_2_1_L0024_L0012_normed', 'x3_2_2_1_L0006_L0003_normed',
            'x3_2_2_1_L0003_L0000_normed', 'x3_2_2_2_L0006_L0003_normed', 'x3_2_2_2_L0024_L0012_normed',
            'x3_2_3_1_L0003_L0000_normed', 'x3_2_3_2_L0024_L0012_normed', 'x3_2_3_2_L0006_L0003_normed',
            'x3_2_3_1_L0006_L0003_normed', 'x3_2_3_1_L0012_L0006_normed', 'x3_2_3_2_L0012_L0006_normed',
            'x3_2_3_2_L0003_L0000_normed', 'x3_2_3_1_L0024_L0012_normed', 'x3_3_1_1_L0012_L0006_normed',
            'x3_3_1_1_L0006_L0003_normed', 'x3_3_1_2_L0003_L0000_normed', 'x3_3_1_1_L0024_L0012_normed',
            'x3_3_1_1_L0000_normed',
            'x3_3_1_2_L0012_L0006_normed', 'x3_3_1_2_L0000_normed', 'x3_3_1_2_L0006_L0003_normed',
            'x3_3_1_2_L0024_L0012_normed',
            'x3_3_1_1_L0003_L0000_normed', 'x3_3_1_1_L0048_L0024_normed', 'x3_3_1_2_L0048_L0024_normed',
            'x3_3_2_1_L0000_normed',
            'x3_3_2_1_L0003_L0000_normed', 'x3_3_2_2_L0000_normed', 'x3_3_2_1_L0048_L0024_normed',
            'x3_3_2_2_L0024_L0012_normed',
            'x3_3_2_2_L0012_L0006_normed', 'x3_3_2_1_L0012_L0006_normed', 'x3_3_2_2_L0003_L0000_normed',
            'x3_3_2_1_L0024_L0012_normed', 'x3_3_2_1_L0006_L0003_normed', 'x3_3_2_2_L0006_L0003_normed',
            'x3_3_2_2_L0048_L0024_normed', 'x4_2_1_1_L0024_L0012_normed', 'x4_2_1_2_L0024_L0012_normed',
            'x4_2_1_2_L0096_L0048_normed', 'x4_2_1_2_L0048_L0024_normed', 'x4_2_1_1_L0096_L0048_normed',
            'x4_2_1_1_L0003_L0000_normed', 'x4_2_1_1_L0012_L0006_normed', 'x4_2_1_1_L0006_L0003_normed',
            'x4_2_1_2_L0006_L0003_normed', 'x4_2_1_2_L0003_L0000_normed', 'x4_2_1_2_L0012_L0006_normed',
            'x4_2_1_1_L0048_L0024_normed', 'x4_2_2_2_L0006_L0003_normed', 'x4_2_2_2_L0012_L0006_normed',
            'x4_2_2_1_L0024_L0012_normed', 'x4_2_2_1_L0003_L0000_normed', 'x4_2_2_1_L0006_L0003_normed',
            'x4_2_2_2_L0024_L0012_normed', 'x4_2_2_1_L0012_L0006_normed', 'x4_2_2_2_L0003_L0000_normed',
            'x4_2_3_1_L0003_L0000_normed', 'x4_2_3_2_L0006_L0003_normed', 'x4_2_3_1_L0024_L0012_normed',
            'x4_2_3_2_L0024_L0012_normed', 'x4_2_3_1_L0006_L0003_normed', 'x4_2_3_2_L0012_L0006_normed',
            'x4_2_3_2_L0003_L0000_normed', 'x4_2_3_1_L0012_L0006_normed', 'x5_1_1_0_L0000_normed',
            'x5_1_2_0_L0000_normed',
            'x5_2_1_2_L0024_L0012_normed', 'x5_2_1_2_L0048_L0024_normed', 'x5_2_1_1_L0024_L0012_normed',
            'x5_2_1_1_L0006_L0003_normed', 'x5_2_1_2_L0012_L0006_normed', 'x5_2_1_2_L0000_normed',
            'x5_2_1_1_L0000_normed',
            'x5_2_1_2_L0003_L0000_normed', 'x5_2_1_1_L0048_L0024_normed', 'x5_2_1_2_L0006_L0003_normed',
            'x5_2_1_1_L0012_L0006_normed', 'x5_2_1_1_L0003_L0000_normed', 'x5_3_1_0_L0012_L0006_normed',
            'x5_3_1_0_L0024_L0012_normed', 'x5_3_1_0_L0003_L0000_normed', 'x5_3_1_0_L0096_L0048_normed',
            'x5_3_1_0_L0006_L0003_normed', 'x5_3_1_0_L0048_L0024_normed', 'x5_3_2_0_L0012_L0006_normed',
            'x5_3_2_0_L0003_L0000_normed', 'x5_3_2_0_L0048_L0024_normed', 'x5_3_2_0_L0006_L0003_normed',
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
            'x8_4_1_0_L0096_L0048_normed', 'x8_4_1_0_L0192_L0096_normed', 'x8_5_1_0_L0003_L0000_normed',
            'x8_5_1_0_L0006_L0003_normed', 'x8_5_1_0_L0012_L0006_normed', 'x8_5_1_0_L0024_L0012_normed',
            'x8_5_1_0_L0048_L0024_normed', 'x8_5_1_0_L0096_L0048_normed', 'x8_5_1_0_L0192_L0096_normed',
            'x9_1_1_0_L0012_L0006_normed', 'x9_1_1_0_L0048_L0024_normed', 'x9_1_1_0_L0003_L0000_normed',
            'x9_1_1_0_L0096_L0048_normed', 'x9_1_1_0_L0006_L0003_normed', 'x9_1_1_0_L0024_L0012_normed',
            'x11_1_1_2_L0006_L0003_normed', 'x11_1_1_1_L0012_L0006_normed', 'x11_1_1_1_L0003_L0000_normed',
            'x11_1_1_1_L0024_L0012_normed', 'x11_1_1_2_L0003_L0000_normed', 'x11_1_1_2_L0012_L0006_normed',
            'x11_1_1_2_L0024_L0012_normed', 'x11_1_1_1_L0048_L0024_normed', 'x11_1_1_1_L0006_L0003_normed',
            'x11_1_1_2_L0048_L0024_normed', 'x12_1_1_1_L0012_L0006_normed', 'x12_1_1_2_L0003_L0000_normed',
            'x12_1_1_2_L0024_L0012_normed', 'x12_1_1_1_L0006_L0003_normed', 'x12_1_1_1_L0024_L0012_normed',
            'x12_1_1_2_L0006_L0003_normed', 'x12_1_1_2_L0012_L0006_normed', 'x12_1_1_1_L0003_L0000_normed',
            'x12_1_2_1_L0003_L0000_normed', 'x12_1_2_2_L0003_L0000_normed', 'x12_1_2_1_L0024_L0012_normed',
            'x12_1_2_2_L0006_L0003_normed', 'x12_1_2_1_L0012_L0006_normed', 'x12_1_2_2_L0012_L0006_normed',
            'x12_1_2_2_L0024_L0012_normed', 'x12_1_2_1_L0006_L0003_normed', 'x13_1_1_0_L0000_normed',
            'x13_2_1_0_L0000_normed', 'x13_3_1_0_L0000_normed', 'x13_4_1_0_L0000_normed',

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
            'x8_4_1_0_L0192_L0096_normed', 'x8_5_1_0_L0003_L0000_normed', 'x8_5_1_0_L0006_L0003_normed',
            'x8_5_1_0_L0012_L0006_normed', 'x8_5_1_0_L0024_L0012_normed', 'x8_5_1_0_L0048_L0024_normed',
            'x8_5_1_0_L0096_L0048_normed', 'x8_5_1_0_L0192_L0096_normed', 'x12_1_1_1_L0003_L0000_normed',
            'x12_1_1_2_L0003_L0000_normed', 'x12_1_1_1_L0006_L0003_normed', 'x12_1_1_2_L0006_L0003_normed',
            'x12_1_1_1_L0012_L0006_normed', 'x12_1_1_2_L0012_L0006_normed', 'x12_1_1_1_L0024_L0012_normed',
            'x12_1_1_2_L0024_L0012_normed', 'x12_1_1_1_L0048_L0024_normed', 'x12_1_1_2_L0048_L0024_normed',
            'x12_1_2_1_L0003_L0000_normed', 'x12_1_2_2_L0003_L0000_normed', 'x12_1_2_1_L0006_L0003_normed',
            'x12_1_2_2_L0006_L0003_normed', 'x12_1_2_1_L0012_L0006_normed', 'x12_1_2_2_L0012_L0006_normed',
            'x12_1_2_1_L0024_L0012_normed', 'x12_1_2_2_L0024_L0012_normed', 'x12_1_2_1_L0048_L0024_normed',
            'x12_1_2_2_L0048_L0024_normed', 'x13_1_1_0_L0000_normed', 'x13_2_1_0_L0000_normed',
            'x13_3_1_0_L0000_normed', 'x13_4_1_0_L0000_normed',

        ]
        self.feat_cols = list(set(self.feat_cols))

    def agg_factors(self, skey_i):

        # get label df
        task_df = self.factor_dao.read_factor_by_skey_and_day('production_label', skey_i,
                                                              self.target_day, version='v1')
        if task_df is not None:
            task_df['date'] = self.target_day
            task_df['time'] = task_df['time'].apply(lambda x: int(x / 1000))
            # generate time-id features
            task_df['week_id'] = task_df['date'].apply(lambda x: get_weekday(x))
            task_df['minute'] = task_df['time'].apply(lambda x: time_to_minute(x / 1000))
            task_df['minute_id'] = task_df['minute'].apply(lambda x: int(x / 5))
            task_df['session_id'] = task_df['minute'].apply(lambda x: get_session_id(x))
            task_df['is_five'] = task_df['time'].apply(lambda x: 1 if int(x / 100) % 5 == 0 else 0)
            task_df['is_ten'] = task_df['time'].apply(lambda x: 1 if int(x / 100) % 10 == 0 else 0)
            task_df['is_clock'] = task_df['time'].apply(lambda x: 1 if int(x / 100) % 100 == 0 else 0)

            # generate extended labels
            for side in ['sell', 'buy']:
                for tick in [9, 30, 90, 300]:
                    rank_name = '{side}RetFuture{tick}Rank'.format(side=side, tick=tick)
                    val_name = '{side}RetFuture{tick}_normed'.format(side=side, tick=tick)
                    top_name = '{side}RetFuture{tick}Top'.format(side=side, tick=tick)
                    task_df[rank_name] = task_df[val_name].rank() / task_df[val_name].count()
                    task_df[top_name] = task_df[rank_name].apply(lambda x: 1 if x >= (1 - top_ratio_ce) else 0)

            # get feature df
            feat_path = '/b/sta_feat_eq_cn/share_pengfei_marlowe/normalized_feature/{day}/{skey}.parquet'.format(
                day=self.target_day,
                skey=skey_i)
            if not os.path.exists(feat_path):
                print("%d, %d feat file missed" % (self.target_day, skey_i))
                return
            feat_dfs = pd.read_parquet(feat_path)
            feat_cols = {fe: fe + '_normed' for fe in feat_dfs.columns.tolist() if re.match('x\d', fe)}
            feat_dfs.rename(columns=feat_cols, inplace=True)
            feat_dfs = feat_dfs[self.join_keys+self.feat_cols]
            task_df = task_df.merge(feat_dfs, on=self.join_keys)

            day_idx = self.trade_days.index(self.target_day)
            if day_idx == 0:
                print("%d, %d not valid trade day" % (self.target_day, skey_i))
            # TODO: check the correctness of mta
            mta_path = '/b/home/pengfei_ji/factor_dbs/NEW_LEVEL/production_mta/mta_{day}.csv'.format(day=self.target_day)
            if not os.path.exists(mta_path):
                print('mta missed %d %d' % (skey_i, self.target_day))
                return
            mta_df = pd.read_csv(mta_path)
            mta_df = mta_df.loc[mta_df.skey == skey_i].drop(['date'], axis=1)
            mta_df = mta_df[['skey', 'allZT', 'hasZT', 'isZT', 'allDT',
                             'hasDT', 'isDT', 'isST', 'SW1_codes', 'SW2_codes', 'SW3_codes',
                             'marketValue', 'marketShares']]
            for col in ['allZT', 'hasZT', 'isZT', 'allDT',
                        'hasDT', 'isDT', 'isST']:
                mta_df[col] = mta_df[col].apply(lambda x: 1 if x > 0 else 0)
            task_df = task_df.merge(mta_df, on=['skey'])
            self.factor_dao.save_factors(data_df=task_df, factor_group='production_factors',
                                         skey=skey_i, day=self.target_day, version='v1')
        else:
            print("%d, %d label file missed" % (self.target_day, skey_i))
            return


def main():
    day_str = sys.argv[1]
    day_i = int("".join(day_str[:day_str.find('T')].split('-')))
    print(day_i)
    # day_i = 20201102
    array_id = int(get_slurm_env("SLURM_ARRAY_TASK_ID"))
    array_size = int(get_slurm_env("SLURM_ARRAY_TASK_COUNT"))
    proc_id = int(get_slurm_env("SLURM_PROCID"))
    task_size = int(get_slurm_env("SLURM_NTASKS"))
    work_id = array_id * task_size + proc_id
    total_worker = array_size * task_size
    dist_tasks = get_all_stocks_by_day(day_i)
    unit_tasks = [t for i, t in enumerate(dist_tasks) if i % total_worker == work_id]
    print('allocate the number of tasks %d out of %d' % (len(unit_tasks), len(dist_tasks)))
    norm_task = FactorAgg(day_i)
    for unit in tqdm(unit_tasks):
        norm_task.agg_factors(unit)


if __name__ == '__main__':
    main()
