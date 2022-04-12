import os
import sys
import numpy as np
import json

# airflow tasks test daily_workflow compute_label_normalizer 2020-11-02
import pandas as pd

sys.path.insert(0, '/b/home/pengfei_ji/airflow/dags/sta_17001/scripts/workflow_utils/')
from common_utils import get_slurm_env, get_all_stocks_by_day, get_trade_days
from factor_dao import FactorDAO
from tqdm import *

# TODO: stock embed important check

class LabelNorm(object):

    def __init__(self, target_day):
        self.factor_dao = FactorDAO('/b/home/pengfei_ji/factor_dbs/')
        self.target_day = target_day
        self.source_path = '/b/sta_feat_eq_cn/sta_feat_1_2_l2/LabelTm/{day}/{skey}.parquet'
        self.save_path = '/b/home/pengfei_ji/factor_dbs/NEW_LEVEL/normed_LabelTm/v0/{day}/{skey}/normed_LabelTm_{day}_{skey}.pkl'
        self.trade_days = get_trade_days()
        self.current_idx = self.trade_days.index(self.target_day)
        self.prev_days = self.trade_days[self.current_idx - 60: self.current_idx]
        self.headers = {'date', 'skey', 'ordering', 'time'}
        with open('/b/home/pengfei_ji/factor_dbs/SW1_codes.json', 'r') as fp:
            self.SW1_codes = json.load(fp)

        with open('/b/home/pengfei_ji/factor_dbs/SW2_codes.json', 'r') as fp:
            self.SW2_codes = json.load(fp)

        with open('/b/home/pengfei_ji/factor_dbs/SW3_codes.json', 'r') as fp:
            self.SW3_codes = json.load(fp)

    def compute_normalizer(self, skey_i):
        # compute normalizer

        label_dfs = []
        for prev_day in self.prev_days:
            source_path = self.source_path.format(day=prev_day, skey=skey_i)
            if os.path.exists(source_path):
                label_df = pd.read_parquet(source_path)
                label_dfs.append(label_df)

        print(len(label_dfs))
        if len(label_dfs) > 0:

            concat_ticks = pd.concat(label_dfs, ignore_index=True)
            norms = dict()
            norms['skey'] = skey_i
            norms['date'] = self.target_day
            for factor in concat_ticks.columns.tolist():
                if factor not in self.headers:
                    norms["{factor}_std".format(factor=factor)] = concat_ticks[factor].dropna().std()

            current_path = self.source_path.format(day=self.target_day, skey=skey_i)
            if not os.path.exists(current_path):
                print("%d, %d raw label file is missed" % (self.target_day, skey_i))
                return
            current_df = pd.read_parquet(current_path)

            factor_to_norm = set(current_df.columns.tolist()) - self.headers

            for factor in factor_to_norm:
                current_df[factor + '_normed'] = current_df[factor] / norms[factor + '_std'] \
                    if norms[factor + '_std'] != 0 else 0.0
                current_df[factor + '_normed'] = np.sign(current_df[factor + '_normed']) \
                                                 * np.power(np.fabs(current_df[factor + '_normed']), 1 / 2)

            factors = list(self.headers) + [fac + '_normed' for fac in factor_to_norm]
            current_df = current_df[factors]
            current_df['date'] = self.target_day
            current_df['buyRetFuture90_std'] = norms['buyRetFuture90_std']
            current_df['sellRetFuture90_std'] = norms['sellRetFuture90_std']

            self.factor_dao.save_factors(data_df=current_df, factor_group='production_label',
                                         skey=skey_i, day=self.target_day, version='v1')
        else:
            print('%d, %d previous label file is missed' % (self.target_day, skey_i))

    def compute_mta_factors(self):
        mta_path = '/b/com_md_eq_cn/mdbar1d_jq/{day}.parquet'.format(day=self.target_day)
        selected_factors = ['skey', 'allZT', 'hasZT', 'isZT', 'allDT',
                            'hasDT', 'isDT', 'isST']
        embedding_factors = [
            'marketValue', 'marketShares', 'SW1_code', 'SW2_code', 'SW3_code'
        ]

        if os.path.exists(mta_path):
            mta_df = pd.read_parquet(mta_path)

            normed_skeys = list()
            day_skeys = list(set(mta_df['skey'].unique()))

            for skey in tqdm(day_skeys):

                raw_dict = mta_df.loc[mta_df.skey == skey][selected_factors + embedding_factors]
                raw_dict = raw_dict.iloc[0].to_dict()

                factor_to_norm = selected_factors[1:]
                skey_ent = dict()
                skey_ent['skey'] = skey
                skey_ent['date'] = self.target_day
                for factor in factor_to_norm:
                    skey_ent[factor] = raw_dict[factor]

                skey_ent['SW1_codes'] = self.SW1_codes[raw_dict['SW1_code']] if raw_dict[
                                                                                    'SW1_code'] in self.SW1_codes else len(
                    self.SW1_codes)
                skey_ent['SW2_codes'] = self.SW2_codes[raw_dict['SW2_code']] if raw_dict[
                                                                                    'SW2_code'] in self.SW2_codes else len(
                    self.SW2_codes)
                skey_ent['SW3_codes'] = self.SW3_codes[raw_dict['SW3_code']] if raw_dict[
                                                                                    'SW3_code'] in self.SW3_codes else len(
                    self.SW3_codes)

                skey_ent['marketValue'] = max(0, int((np.log10(raw_dict['marketValue']) - 7) * 2)) if raw_dict[
                                                                                                          'marketValue'] > 0 \
                    else 0
                skey_ent['marketShares'] = max(0, int(int((np.log10(raw_dict['marketShares']) - 6) * 2))) \
                    if raw_dict['marketShares'] > 0 else 0

                normed_skeys.append(skey_ent)

            normed_skeys = pd.DataFrame(normed_skeys)
            print(normed_skeys.shape)
            normed_skeys.to_csv('/b/home/pengfei_ji/factor_dbs/NEW_LEVEL/production_mta/mta_{day}.csv'
                                .format(day=self.target_day))


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
    norm_task = LabelNorm(day_i)
    if work_id == 0:
        norm_task.compute_mta_factors()
    for unit in tqdm(unit_tasks):
        norm_task.compute_normalizer(unit)


if __name__ == '__main__':
    main()
