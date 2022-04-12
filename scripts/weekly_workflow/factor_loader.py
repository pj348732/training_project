import random
import sys
import numpy as np
from torch.utils.data import IterableDataset
import torch
from factor_dao import FactorDAO
import pickle
import torch.nn.functional as F
import torch.nn as nn


class FixMeanMultiTaskLoss(torch.nn.Module):

    def __init__(self, reg_pos, clf_pos, multi_clf_pos):
        super().__init__()
        self.clf_loss = torch.nn.BCEWithLogitsLoss()
        self.reg_loss = SmoothL1()
        self.multi_clf_loss = torch.nn.CrossEntropyLoss()
        self.reg_pos = reg_pos
        self.cls_pos = clf_pos
        self.multi_clf_pos = multi_clf_pos
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        print('FixMeanMultiTaskLoss')

    def forward(self, preds, gts):
        losses = []
        # reg loss
        for i in self.reg_pos:
            losses.append(self.reg_loss(preds[:, i], gts[:, i]))
            pearson = self.cos(preds[:, i] - gts[:, i].mean(dim=0, keepdim=True),
                               gts[:, i] - gts[:, i].mean(dim=0, keepdim=True))
            losses.append(-pearson)

        # binary classify loss
        for i in self.cls_pos:
            losses.append(self.clf_loss(preds[:, i], gts[:, i]))

        # stock classify loss
        for i in self.multi_clf_pos:
            losses.append(self.multi_clf_loss(preds[:, i:], gts[:, i].long()))

        total_loss = sum(losses)
        return total_loss, losses


class SmoothL1(torch.nn.Module):
    def __init__(self):
        super(SmoothL1, self).__init__()
        # self.delta = delta

    def __call__(self, preds, gts):
        smoothl1_loss = F.smooth_l1_loss(gts, preds)
        return smoothl1_loss


group2universe = {
    'LOW': 'ic',
    'HIGH': 'ic',
    'MID_LOW': 'ic',
    'MID_HIGH': 'ic',
    'IF_HIGH': 'if',
    'IF_MID': 'if',
    'IF_LOW': 'if',
    'CSI_1': 'csi',
    'CSI_2': 'csi',
    'CSI_3': 'csi',
    'CSI_4': 'csi',
    'CSI_5': 'csi',
    'REST_1': 'csi',
    'REST_2': 'csi',
    'REST_3': 'csi',
    'REST_4': 'csi',
    'REST_5': 'csi',
}


def get_stock_map(stock_type, period):
    if stock_type in group2universe:
        uni = group2universe[stock_type]
        with open(f'/b/home/pengfei_ji/{uni}_price_group/period_skey2groups.pkl', 'rb') as fp:
            grouped_skeys = pickle.load(fp)
        real_period = int(period / 100) * 100 + 1

        if real_period in grouped_skeys:
            grouped_skeys = grouped_skeys[real_period][stock_type]
            return grouped_skeys
        else:
            print('wrong period....')
            return set()
    else:
        print('wrong stock type....')
        return set()


class NewAlphaDataset(torch.utils.data.Dataset):

    def __init__(self, start_day, end_day, base_path,
                 factor_group, train_factors, alias_factors,
                 predict_factors, stocks, factor_version,
                 train_time_step, alias_time_step, stock_type):

        super(torch.utils.data.Dataset).__init__()

        print('Start initializing AlphaDataset...')

        self.factor_dao = FactorDAO(base_path)

        self.day_skey_pairs = [(day, skey) for day, skey
                               in self.factor_dao.find_day_skey_pairs(factor_group,
                                                                      factor_version,
                                                                      start_day,
                                                                      end_day,
                                                                      stocks)
                               if skey in stocks and start_day <= day < end_day]

        self.start_day = start_day
        self.end_day = end_day
        self.base_path = base_path
        self.factor_group = factor_group
        self.train_factors = train_factors
        self.predict_factors = predict_factors
        self.stocks = set(stocks)
        self.alias_factors = alias_factors
        self.whole_factors = list(set(self.predict_factors
                                      + self.alias_factors + self.train_factors)) + ['minute', 'skey']
        self.factor_version = factor_version
        self.train_time_step = train_time_step
        self.alias_time_step = alias_time_step
        self.time_step = max(self.train_time_step, self.alias_time_step)

        self.time_accu = 0
        self.time_count = 0
        print('load %s data between %d and %d with %d files of %d stocks ' % ('train', start_day, end_day,
                                                                              len(self.day_skey_pairs),
                                                                              len(self.stocks)))

    def __getitem__(self, idx):

        feats = []
        alias_feats = []
        stock_ids = []
        predictors = []
        all_times = []
        for sub_df in self.factor_dao.iter_get_factors_by_day_skey(factor_group=self.factor_group,
                                                                   day_skey_pairs=[self.day_skey_pairs[idx]],
                                                                   version=self.factor_version):

            if len(sub_df.dropna()) / len(sub_df) < 0.1:
                return []

            sub_feats = sub_df[self.train_factors].to_numpy()
            sub_alias_feats = sub_df[self.alias_factors].to_numpy()
            sub_predictors = sub_df[self.predict_factors].to_numpy()
            sub_stocks = sub_df[['allZT', 'hasZT', 'isZT', 'allDT',
                                 'hasDT', 'isDT', 'isST', 'SW1_codes', 'SW2_codes', 'SW3_codes',
                                 'marketValue', 'marketShares']].to_numpy()
            times = sub_df['time'].tolist()

            idxs = np.random.permutation(len(sub_df) - self.time_step + 1)
            for i in idxs:
                t = times[i + self.time_step - 1]
                if 112830 < t < 120000 or 130000 < t < 130130:
                    continue
                # if not (93000 <= t <= 93300):
                #     continue
                feat_i = sub_feats[i + self.time_step - self.train_time_step:i + self.time_step]
                alias_feat_i = sub_alias_feats[i + self.time_step - self.alias_time_step:i + self.time_step]
                predictor_i = sub_predictors[i + self.time_step - 1]
                stock_i = sub_stocks[i + self.time_step - 1]

                feat_i = np.nan_to_num(feat_i, nan=0.0, posinf=0.0, neginf=0.0)
                alias_feat_i = np.nan_to_num(alias_feat_i, nan=0.0, posinf=0.0, neginf=0.0)
                stock_i = np.nan_to_num(stock_i, nan=0.0, posinf=0.0, neginf=0.0)
                all_times.append(t)
                if not (np.isnan(feat_i).any() or np.isnan(predictor_i).any() or np.isnan(
                        alias_feat_i).any()
                        or np.isinf(feat_i).any() or np.isinf(predictor_i).any() or np.isinf(
                            alias_feat_i).any()):
                    feats.append(feat_i)
                    alias_feats.append(alias_feat_i)
                    predictors.append(predictor_i)
                    stock_ids.append(stock_i)

                if len(stock_ids) == 1024:
                    break
        if len(stock_ids) > 32:
            data = [np.asarray(feats).astype(np.float32),
                    np.asarray(alias_feats).astype(np.float32),
                    np.asarray(predictors).astype(np.float32),
                    np.asarray(stock_ids).astype(np.float32)]

            return data

        else:
            return []

    def __len__(self):
        return len(self.day_skey_pairs)


class DaySkeyMapDataset(torch.utils.data.Dataset):

    def __init__(self, train_config, model_config, mode):
        self.train_config = train_config
        self.model_config = model_config
        self.mode = mode
        self.feats, self.alias_feats, self.predictors, self.stock_ids = [], [], [], []

        self.stocks = set(train_config['target_stock']) if 'target_stock' in train_config \
            else set(get_stock_map(train_config['stock_type'], train_config['valid_start_day']))
        self.factor_dao = FactorDAO(train_config['base_path'])
        start_day = train_config[mode + '_start_day']
        end_day = train_config[mode + '_end_day']
        self.day_skey_pairs = [(day, skey) for day, skey
                               in self.factor_dao.find_day_skey_pairs(train_config['factor_group'],
                                                                      train_config['factor_version'],
                                                                      start_day,
                                                                      end_day,
                                                                      self.stocks)
                               if skey in self.stocks and start_day <= day < end_day]

        if len(self.stocks) > 1:
            self.day_skey_pairs = random.sample(self.day_skey_pairs, 60)
        print('load %s data between %d and %d with %d files of %d stocks ' % (mode, start_day, end_day,
                                                                              len(self.day_skey_pairs),
                                                                              len(self.stocks)))
        self.time_step = max(train_config['train_time_step'], train_config['alias_time_step'])
        self.train_time_step = train_config['train_time_step']
        self.alias_time_step = train_config['alias_time_step']
        self.load_in_memory()

    def load_in_memory(self):

        # re-shuffle and clean
        random.shuffle(self.day_skey_pairs)
        self.feats, self.alias_feats, self.predictors, self.stock_ids = [], [], [], []

        for sub_df in self.factor_dao.iter_get_factors_by_day_skey(factor_group=self.train_config['factor_group'],
                                                                   day_skey_pairs=self.day_skey_pairs,
                                                                   version=self.train_config['factor_version']):

            sub_feats = sub_df[self.train_config['train_factors']].to_numpy()
            sub_alias_feats = sub_df[self.train_config['alias_factors']].to_numpy()
            sub_predictors = sub_df[self.train_config['predict_factors']].to_numpy()
            sub_stocks = sub_df[['allZT', 'hasZT', 'isZT', 'allDT',
                                 'hasDT', 'isDT', 'isST', 'SW1_codes', 'SW2_codes', 'SW3_codes',
                                 'marketValue', 'marketShares']].to_numpy()
            times = sub_df['time'].tolist()

            feats = []
            alias_feats = []
            stock_ids = []
            predictors = []

            idxs = np.random.permutation(len(sub_df) - self.time_step + 1)

            for i in idxs:

                t = times[i + self.time_step - 1]
                if 112830 < t < 120000 or 130000 < t < 130130:
                    continue

                feat_i = sub_feats[i + self.time_step - self.train_time_step:i + self.time_step]
                alias_feat_i = sub_alias_feats[i + self.time_step - self.alias_time_step:i + self.time_step]
                predictor_i = sub_predictors[i + self.time_step - 1]
                stock_i = sub_stocks[i + self.time_step - 1]

                feat_i = np.nan_to_num(feat_i, nan=0.0, posinf=0.0, neginf=0.0)
                alias_feat_i = np.nan_to_num(alias_feat_i, nan=0.0, posinf=0.0, neginf=0.0)
                stock_i = np.nan_to_num(stock_i, nan=0.0, posinf=0.0, neginf=0.0)

                if not (np.isnan(feat_i).any() or np.isnan(predictor_i).any() or np.isnan(
                        alias_feat_i).any()
                        or np.isinf(feat_i).any() or np.isinf(predictor_i).any() or np.isinf(
                            alias_feat_i).any()):
                    feats.append(feat_i)
                    alias_feats.append(alias_feat_i)
                    predictors.append(predictor_i)
                    stock_ids.append(stock_i)
            if len(stock_ids) > 32:
                self.feats.append(np.asarray(feats))
                self.alias_feats.append(np.asarray(alias_feats))
                self.predictors.append(np.asarray(predictors))
                self.stock_ids.append(np.asarray(stock_ids))

        print('number of batches %d' % len(self.stock_ids))

    def __iter__(self):
        idxs = np.random.permutation(len(self.feats) - 1)
        for i in idxs:
            yield self.feats[i], self.alias_feats[i], self.predictors[i], self.stock_ids[i]

    def __len__(self):
        return len(self.stock_ids)

    def __getitem__(self, index):
        return np.asarray(self.feats[index]).astype(np.float32), \
               np.asarray(self.alias_feats[index]).astype(np.float32), \
               np.asarray(self.predictors[index]).astype(np.float32), \
               np.asarray(self.stock_ids[index]).astype(np.float32)
