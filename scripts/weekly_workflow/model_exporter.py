import sys
sys.path.insert(0, '/b/home/pengfei_ji/airflow/dags/sta_17001/scripts/workflow_utils/')
import pickle
import importlib
import os
from tqdm import *
from os import listdir
from os.path import isfile, join, isdir
from enum import Enum
import torch
from collections import OrderedDict
from sta4_file_exporter import encode_file, encode_sta4_model
import yaml
from yaml.loader import SafeLoader
from common_utils import get_encode_path
import subprocess


class ModelMode(Enum):
    MBD = 0
    SNAPSHOT = 1


def load_model(model_configs):
    path = model_configs['model_path']
    idx = path.rfind('.')
    module_name = path[:idx]
    class_name = path[(idx + 1):]
    module = importlib.import_module(module_name)
    loaded_class = getattr(module, class_name)
    loaded_instance = loaded_class.load_model_from_config(model_configs)
    return loaded_instance


class ModelExporter(object):

    def __init__(self, target_day):

        self.target_day = target_day
        self.model_store_path = '/b/home/pengfei_ji/production_models/'
        # self.encode_path = f'/b/home/pengfei_ji/mnt_files/{target_day}/'
        self.encode_path = get_encode_path(target_day)
        # load mbd and snapshot model template
        with open('/b/home/pengfei_ji/airflow/dags/sta_17001/scripts/weekly_workflow/train_configs/mbd_model.yaml') as f:
            self.mbd_model_template = yaml.load(f, Loader=SafeLoader)
        self.mbd_model_template = load_model(self.mbd_model_template)
        with open('/b/home/pengfei_ji/airflow/dags/sta_17001/scripts/weekly_workflow/train_configs/snap_model.yaml') as f:
            self.ss_model_config = yaml.load(f, Loader=SafeLoader)
        self.ss_model_config = load_model(self.ss_model_config)

        # construct the skey and group mapping for model parameters loading
        self.skey2grps = dict()
        self.grp2skeys = dict()

        # for uni in ['ic', 'if', 'csi', 'rest']:
        for uni in ['ic']:
            with open(f'/b/home/pengfei_ji/{uni}_price_group/period_skey2groups.pkl', 'rb') as fp:
                ex = pickle.load(fp)
                base_day = int(self.target_day / 100) * 100 + 1
                for grp in ex[base_day]:
                    for skey in ex[base_day][grp]:
                        self.skey2grps[int(skey)] = grp

        for skey in self.skey2grps:
            grp = self.skey2grps[skey]
            if grp not in self.grp2skeys:
                self.grp2skeys[grp] = set()
            self.grp2skeys[grp].add(int(skey))

        print('number of skey with group %d, number of price groups %d' % (len(self.skey2grps), len(self.grp2skeys)))

    def export_alpha_models(self,  model_mode):

        shared_state_dicts = []
        finetuned_state_dicts = []
        target_day = self.target_day

        feat = ""
        side = 'both'
        if model_mode == ModelMode.MBD:
            feat = 'MBD_FINAL'
        elif model_mode == ModelMode.SNAPSHOT:
            feat = 'SNAPSHOT_FINAL'

        for grp in self.grp2skeys:
            # TODO: change the date part
            # get the shared model
            # pretrain_model_prod_combo_SNAPSHOT_FINAL_both_LOW_20200901
            # both_combo_SNAPSHOT_FINAL_2300618_buy_20200901
            share_model_path = self.model_store_path \
                               + f'pretrain_model_prod_combo_{feat}_both_{grp}_{target_day}/models/'

            if not os.path.exists(share_model_path):
                print('miss %s' % share_model_path)
                continue
            model_files = [share_model_path + f for f in listdir(share_model_path)
                           if isfile(join(share_model_path, f)) if
                           'best' not in f]
            model_files.sort(key=os.path.getatime)
            share_model_path = model_files[-1]

            # for pretrain model, only keep the shared part
            shared_param_dict = torch.load(share_model_path)
            shared_part = OrderedDict()
            default_part = OrderedDict()

            for key in shared_param_dict.keys():
                if 'tick_net' in key:
                    shared_part[key] = shared_param_dict[key].cpu()
                else:
                    default_part[key] = shared_param_dict[key].cpu()

            print(grp, share_model_path, ' is loaded....')

            shared_state_dicts.append((list(self.grp2skeys[grp]), shared_part))

            # for finetune model, only keep the diff part

            for skey_i in self.grp2skeys[grp]:
                finetune_path = self.model_store_path + f'both_combo_{feat}_{skey_i}_{side}_{target_day}/models/best_model.ckpt'
                if not os.path.exists(finetune_path):
                    print('miss %s' % finetune_path)
                    finetuned_state_dicts.append(([skey_i], default_part))
                else:
                    finetune_param_dict = torch.load(finetune_path)
                    finetune_part = OrderedDict()

                    for key in finetune_param_dict.keys():
                        if 'tick_net' not in key:
                            finetune_part[key[key.find('.') + 1:]] = finetune_param_dict[key].cpu()

                    finetuned_state_dicts.append(([skey_i], finetune_part))

        print(len(shared_state_dicts), len(finetuned_state_dicts))
        if model_mode == ModelMode.MBD:

            seq_num_input = 257
            output_path = self.encode_path + f"sta4_model_mbd_{target_day}.en"
            print([t[1]['task_outputs.11.weight']
                   for t in finetuned_state_dicts if 2002815 in set(t[0])])
            exit(0)

            assert (encode_sta4_model(
                self.mbd_model_template,
                shared_state_dicts,
                finetuned_state_dicts,
                seq_num_input,
                output_path))
            # subprocess.call(['rsync', '-avzhe', 'ssh -p 22', self.encode_path + f"sta4_model_mbd_{target_day}.en",
            #                  f"pengfei_ji@10.9.23.3:/home/pengfei_ji/mnt_files/{target_day}/"])
        else:
            seq_num_input = 100
            output_path = self.encode_path + f"sta4_model_snapshot_{target_day}.en"

            assert (encode_sta4_model(
                self.ss_model_config,
                shared_state_dicts,
                finetuned_state_dicts,
                seq_num_input,
                output_path))
            # subprocess.call(['rsync', '-avzhe', 'ssh -p 22', self.encode_path + f"sta4_model_snapshot_{target_day}.en",
            #                  f"pengfei_ji@10.9.23.3:/home/pengfei_ji/mnt_files/{target_day}/"])


def main():
    # day_str = sys.argv[1]
    # day_i = int("".join(day_str[:day_str.find('T')].split('-')))
    # print(day_i)
    day_i = 20200901
    me = ModelExporter(target_day=day_i)
    me.export_alpha_models(ModelMode.MBD)
    # me.export_alpha_models(ModelMode.SNAPSHOT)


if __name__ == '__main__':
    main()
