import os
import sys
import numpy as np
import json
import pandas as pd
import subprocess

sys.path.insert(0, '/b/home/pengfei_ji/airflow/dags/sta_17001/scripts/workflow_utils/')
from common_utils import get_slurm_env, get_all_stocks_by_day, get_trade_days, get_encode_path
from factor_dao import FactorDAO
from tqdm import *
from sta4_file_exporter import encode_file


def export_stock_file(day_i):
    encode_path = get_encode_path(day_i)
    assert (encode_file('/b/home/pengfei_ji/factor_dbs/NEW_LEVEL/production_mta/mta_{day}.csv'.format(day=day_i),
                        encode_path + f"sta4_stock_id_{day_i}.en"))
    subprocess.call(['rsync', '-avzhe', 'ssh -p 22', encode_path + f"sta4_stock_id_{day_i}.en",
                     f"pengfei_ji@10.9.23.3:/home/pengfei_ji/mnt_files/{day_i}/"])


def export_normalizer_file(day_i):
    all_skeys = get_all_stocks_by_day(day_i)
    norm_entrys = []
    factor_dao = FactorDAO('/b/home/pengfei_ji/factor_dbs/')

    for skey_i in tqdm(all_skeys):

        norm_df = factor_dao.read_factor_by_skey_and_day('production_label', skey_i,
                                                         day_i, version='v1')
        if norm_df is not None:
            if len(norm_df) > 0:
                norm_df = norm_df.iloc[0].to_dict()

                sell_dict = dict()
                buy_dict = dict()

                sell_dict['skey'] = skey_i
                sell_dict['side'] = 2
                sell_dict['alpha_type'] = 90
                sell_dict['std'] = norm_df['sellRetFuture90_std']

                buy_dict['skey'] = skey_i
                buy_dict['side'] = 1
                buy_dict['alpha_type'] = 90
                buy_dict['std'] = norm_df['buyRetFuture90_std']

                norm_entrys.append(buy_dict)
                norm_entrys.append(sell_dict)

    norm_df = pd.DataFrame(norm_entrys)
    print(norm_df.shape)
    print(norm_df.columns.tolist())
    tmp_path = '/b/home/pengfei_ji/factor_dbs/NEW_LEVEL/production_mta/' + f'sta4_alpha_normalizer_{day_i}.csv'
    norm_df.to_csv(tmp_path)

    encode_path = get_encode_path(day_i)

    assert (encode_file(tmp_path,
                        encode_path + f"sta4_alpha_normalizer_{day_i}.en"))
    subprocess.call(['rsync', '-avzhe', 'ssh -p 22', encode_path + f"sta4_alpha_normalizer_{day_i}.en",
                     f"pengfei_ji@10.9.23.3:/home/pengfei_ji/mnt_files/{day_i}/"])


def main():
    day_str = sys.argv[1]
    day_i = int("".join(day_str[:day_str.find('T')].split('-')))
    print(day_i)
    export_stock_file(day_i)
    export_normalizer_file(day_i)

    # day_i = 20201102


if __name__ == '__main__':
    main()
