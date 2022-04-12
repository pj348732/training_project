import sys
import os
import pandas as pd
from common_utils import iter_time_range, time_to_minute, n_day_before, after_one_month, to_str_date
import pickle
import json


def get_stock_map(stock_type, period=None):
    with open('/b/home/pengfei_ji/ic_price_group/period_skey2groups.pkl', 'rb') as fp:
        grouped_skeys = pickle.load(fp)
    grouped_skeys = grouped_skeys[period][stock_type]
    print('skey breakdown %s-%d : %d %d' % (stock_type, period, len(grouped_skeys),
                                            len(grouped_skeys)))
    return grouped_skeys


class AlphaGenerator(object):

    def __init__(self, model_name, alpha_name, sides, day_ranges, base_path,
                 alpha_path):

        self.sides = sides
        self.day_ranges = day_ranges
        self.model_name = model_name
        self.alpha_name = alpha_name
        self.base_path = base_path
        self.alpha_path = alpha_path

    def export_alpha(self):

        total_day_cnt = 0
        try:
            if not os.path.exists(self.alpha_path + self.alpha_name):
                os.mkdir(self.alpha_path + self.alpha_name)

            if not os.path.exists(self.alpha_path + self.alpha_name + '/ic/'):
                os.mkdir(self.alpha_path + self.alpha_name + '/ic/')
        except FileExistsError:
            pass

        for day_range in self.day_ranges:

            stock_keys = set()
            for grp in ['HIGH', 'MID_LOW', 'LOW', 'MID_HIGH']:
                stock_keys = stock_keys | get_stock_map(grp, day_range[0])
            start_day = day_range[0]
            end_day = n_day_before(day_range[1], 1)
            all_preds = []
            stock_cnts = set()
            print(len(stock_keys))
            for skey in stock_keys:

                sell_df = None
                buy_df = None

                for side in self.sides:

                    pred_path = self.model_name + "_{skey}_both_{start}" \
                        .format(skey=skey, side=side, start=start_day)
                    export_path = ('/models/predictions_{side}_{start}_{end}.pkl'.format(start=str(day_range[0]),
                                                                                         end=str(day_range[1]),
                                                                                         side=side))
                    pred_path = self.base_path + pred_path + export_path
                    if os.path.exists(pred_path):
                        print(pred_path)
                        try:
                            if side == 'buy':
                                buy_df = pd.read_pickle(pred_path)
                                buy_df['yHatBuy'] = buy_df['yHatBuy']
                                buy_df.rename(columns={
                                    'label': 'yBuy',
                                }, inplace=True)
                            else:
                                sell_df = pd.read_pickle(pred_path)
                                sell_df.rename(columns={
                                    'label': 'ySell',
                                }, inplace=True)
                                sell_df['yHatSell'] = sell_df['yHatSell']
                        except Exception:
                            print('rerun %s %s %d' % (self.model_name, day_range, skey))
                            print('read wrong %s' % pred_path)
                if sell_df is not None and buy_df is not None:
                    pred_df = buy_df.merge(sell_df, how='outer', on=['date', 'skey', 'time', 'ordering', 'minute'])
                    all_preds.append(pred_df)
                    stock_cnts.add(skey)

            print(len(stock_cnts))
            if len(all_preds) > 0:
                all_preds = pd.concat(all_preds, axis=0)
                for day_i in iter_time_range(start_day, end_day):
                    day_df = all_preds.loc[all_preds.date == day_i]
                    if len(day_df) > 0:
                        day_df.sort_values(by=['skey', 'minute', 'time', 'ordering'])
                        day_df = day_df[['skey', 'date', 'ordering', 'time', 'yHatBuy',
                                         'yHatSell', 'yBuy', 'ySell']]
                        if len(day_df.drop_duplicates(subset=['skey', 'date', 'ordering'])) != len(day_df):

                            dup_df = day_df[day_df.duplicated(subset=['skey', 'date', 'ordering'], keep=False)]
                            dup_skeys = set(dup_df['skey'].tolist())

                            for dup_skey in dup_skeys:
                                print('rerun %s %s %d' % (self.model_name, day_range, dup_skey))

                        print(self.alpha_path + '{alpha}/ic/sta{day}.parquet'.format(
                            day=day_i, alpha=self.alpha_name))

                        day_df['yHatBuy'] = day_df['yHatBuy']
                        # day_df['yHatBuy'] = day_df['yHatBuy'].apply(lambda x: max(min(0.1, x), -0.1))

                        day_df['yHatSell'] = day_df['yHatSell']
                        # day_df['yHatSell'] = day_df['yHatSell'].apply(lambda x: max(min(0.1, x), -0.1))
                        day_df = day_df[['skey', 'date', 'ordering', 'time', 'yHatBuy', 'yHatSell',
                                         'yBuy', 'ySell',
                                         ]]
                        print(stock_cnts - set(day_df.skey.unique()))
                        print(day_df.time.min(), day_df.time.max())
                        day_df.to_parquet(self.alpha_path + '{alpha}/ic/sta{day}.parquet'.format(
                            day=day_i, alpha=self.alpha_name))
                        total_day_cnt += 1

        print(total_day_cnt)


def get_slurm_env(name):
    value = os.getenv(name)
    if value is None:
        if name == 'SLURM_ARRAY_TASK_ID' or name == 'SLURM_PROCID':
            return 0
        else:
            return 1
    else:
        return value


def test_run():
    array_id = int(get_slurm_env("SLURM_ARRAY_TASK_ID"))
    array_size = int(get_slurm_env("SLURM_ARRAY_TASK_COUNT"))
    proc_id = int(get_slurm_env("SLURM_PROCID"))
    task_size = int(get_slurm_env("SLURM_NTASKS"))
    work_id = array_id * task_size + proc_id
    total_worker = array_size * task_size

    base_path = '/b/home/pengfei_ji/production_models/'
    alpha_path = '/b/home/pengfei_ji/alpha_exports/'

    sides = ['buy', 'sell']
    dist_tasks = []
    model_names = ['both_combo_MBD_FINAL']
    day_ranges = [

        # (20201101, 20201201),
        (20200901, 20201001),
        # (20201001, 20201101),
        # (20201201, 20210101),
        #
        # (20200101, 20200201),
        # (20200201, 20200301),
        # (20200301, 20200401),
        # (20200401, 20200501),
        # (20200501, 20200601),
        # (20200601, 20200701),
        # (20200701, 20200801),
        # (20200801, 20200901),
        #
        # (20210101, 20210201),
        # (20210201, 20210301),
        # (20210301, 20210401),
        # (20210401, 20210501),
        # (20210501, 20210601),
        # (20210601, 20210701),
        #
        # (20210701, 20210801),
        # (20210801, 20210901),
        # (20210901, 20211001),
        # (20211001, 20211101),
        # (20211101, 20211201),
        # (20211201, 20220101),

    ]
    for name in model_names:
        for day_range in day_ranges:
            dist_tasks.append([name, day_range])

    unit_tasks = [t for i, t in enumerate(dist_tasks) if i % total_worker == work_id]

    print('allocate %d tasks among %d' % (len(unit_tasks), len(dist_tasks)))

    for task in unit_tasks:
        model_name, day_range = task
        ag = AlphaGenerator(model_name=model_name, alpha_name='prod_alpha',
                            sides=sides, day_ranges=[day_range], base_path=base_path,
                            alpha_path=alpha_path)
        ag.export_alpha()


if __name__ == '__main__':
    test_run()
