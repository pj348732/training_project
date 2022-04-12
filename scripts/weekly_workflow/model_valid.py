import sys
import numpy as np
import torch
sys.path.insert(0, '../workflow_utils/')
import pandas as pd
import yaml
from yaml.loader import SafeLoader
import importlib

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


feat_df = pd.read_csv('./feat_mbd_norm.csv')
label_df = pd.read_csv('./debug_sta4_mbd.csv')

mta_df = pd.read_csv('./mta_20200901.csv')
mta_df = mta_df.loc[mta_df.skey == 2002815][['allZT', 'hasZT', 'isZT', 'allDT',
                                             'hasDT', 'isDT', 'isST', 'SW1_codes', 'SW2_codes', 'SW3_codes',
                                             'marketValue', 'marketShares']].iloc[0].to_numpy()
print(mta_df.shape)

join_df = feat_df.merge(label_df, on=['secid', 'tst', 'sno', 'ApplSeqNum', 'BizIndex'])

# dao_df = factor_dao.read_factor_by_skey_and_day(factor_group='new_task_factors',
#                                                 day=20220321, skey=2002815, version='v8')
#
# stock_ids = dao_df[['allZT', 'hasZT', 'isZT', 'allDT',
#                     'hasDT', 'isDT', 'isST', 'SW1_codes', 'SW2_codes', 'SW3_codes',
#                     'marketValue', 'marketShares']]
# print(stock_ids.head())

# print(join_df.shape)
# print(join_df.columns.tolist())

feat_names = [c[:c.find('_normed')] if '_normed' in c else c for c in feat_map['MBD_FINAL']]
# print(feat_names)
feats = join_df[feat_names].to_numpy()

alphas = join_df[['yhb_unnorm', 'yhs_unnorm']].to_numpy()

updates = join_df.num_update.tolist()
queues = [-1] * 10000

indices = [744, 745, 1002, 1004, 2048, 2060]
wide_feats = []
seq_feats = []
predictors = []
stock_ids = []

for i, q_pos in enumerate(updates):

    queues[q_pos] = i
    if i in indices:

        wide_feat_i = feats[i]
        pred_i = alphas[i]
        seq_feat_i = []
        for j in range(q_pos-15, q_pos+1):

            seq_feat_i.append(feats[queues[j], :-6])

        seq_feat_i = np.asarray(seq_feat_i)
        wide_feat_i = np.nan_to_num(wide_feat_i, nan=0.0, posinf=0.0, neginf=0.0)
        seq_feat_i = np.nan_to_num(seq_feat_i, nan=0.0, posinf=0.0, neginf=0.0)
        wide_feats.append(wide_feat_i)
        seq_feats.append(seq_feat_i)
        predictors.append(pred_i)
        stock_ids.append(mta_df)

wide_feats = torch.from_numpy(np.asarray(wide_feats)).float()
seq_feats = torch.from_numpy(np.asarray(seq_feats)).float()
predictors = torch.from_numpy(np.asarray(predictors)).float()
stock_ids = torch.from_numpy(np.asarray(stock_ids)).int()
print(wide_feats.shape, seq_feats.shape, predictors.shape, stock_ids.shape)

# indices = [1000, 1005, 1010]
# wide_feats = []
# seq_feats = []
# predictors = []
# stock_ids = []
# for i in indices:
#     wide_feat_i = feats[i]
#     seq_feat_i = feats[i - 15:i + 1, :-6]
#     pred_i = alphas[i]
#     wide_feat_i = np.nan_to_num(wide_feat_i, nan=0.0, posinf=0.0, neginf=0.0)
#     seq_feat_i = np.nan_to_num(seq_feat_i, nan=0.0, posinf=0.0, neginf=0.0)
#     wide_feats.append(wide_feat_i)
#     seq_feats.append(seq_feat_i)
#     predictors.append(pred_i)
#     stock_ids.append(mta_df)

# wide_feats = torch.from_numpy(np.asarray(wide_feats)).float()
# seq_feats = torch.from_numpy(np.asarray(seq_feats)).float()
# predictors = torch.from_numpy(np.asarray(predictors)).float()
# stock_ids = torch.from_numpy(np.asarray(stock_ids)).int()
with open('/b/home/pengfei_ji/airflow/dags/sta_17001/scripts/weekly_workflow/train_configs/mbd_model.yaml') as f:
    ss_model_config = yaml.load(f, Loader=SafeLoader)
ss_model_template = load_model(ss_model_config)
params = torch.load(
    '/b/home/pengfei_ji/production_models/both_combo_MBD_FINAL_2002815_both_20200901/models/best_model.ckpt')
params = {key[key.find('.') + 1:]: params[key] for key in params}

ss_model_template.load_state_dict(params)
ss_model_template.eval()
preds = ss_model_template(wide_feats, seq_feats, stock_ids, return_tick=False)
print('loaded model prediction....')
print(preds.shape)
print(preds[:, :2])
print('given alpha.... ')
print(predictors)

