from Crypto import Random
from Crypto.Cipher import AES
import msgpack
import datetime
import collections
import random
import torch
import numpy as np


default_key = b'356afd613a57669313a2be2432f41c50'
BS = 16
pad = lambda s: s + bytes((BS - len(s) % BS) * [BS - len(s) % BS])
unpad = lambda s : s[:-ord(s[len(s)-1:])]


def encrypt(data, key=default_key):
    data = pad(data)
    iv = Random.new().read(AES.block_size)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return iv + cipher.encrypt(data)


def decrypt(data, key=default_key):
    iv = data[:16]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(data[16:]))


def encode_file(input_path, output_path):
    with open(input_path, 'rb') as fin:
        data = fin.read()
        with open(output_path, 'wb') as fout:
            fout.write(encrypt(data))
    return True
        

def decode_file(input_path, output_path):
    with open(input_path, 'rb') as fin:
        data = fin.read()
        with open(output_path, 'wb') as fout:
            fout.write(decrypt(data))
    return True


'''
1) light_mmoe_model - LightMMOE
2) shared_state_dict_list - list of (skey list, state_dict)
3) finetuned_state_dict_list - list of (skey list, state_dict)
4) seq_num_input - 259(mbd) 102(snapshot)
5) output_path - encrypted model filename
'''
def encode_sta4_model(
        light_mmoe_model,
        shared_state_dict_list,
        finetuned_state_dict_list,
        seq_num_input,
        output_path):
    model = light_mmoe_model.cpu().eval()

    def pack_tensor_to_msg(tensor):
        tensor = tensor.float()
        msg_tensor = {}
        msg_tensor['shape'] = list(tensor.shape)
        msg_tensor['data'] = tensor.cpu().detach().numpy().tobytes()
        return msg_tensor

    def pack_state_dict_to_msg(secid_list, state_dict):
        msg_sd = {}
        msg_sd['secids'] = list(secid_list)
        msg_sd['model'] = []
        for name, tensor in state_dict.items():
            msg_tensor = pack_tensor_to_msg(tensor)
            msg_tensor['name'] = name
            msg_sd['model'].append(msg_tensor) 
        return msg_sd

    def pack_state_dict_list_to_msg(state_dict_list):
        msg_list = []
        for secid_list, state_dict in state_dict_list:
            msg_list.append(pack_state_dict_to_msg(secid_list, state_dict))
        return msg_list

    def random_input():
        def random_stock_ids():
            a = torch.zeros(12).int()
            for i in range(0, 7):
                a[i] = random.randint(0, 1)
            a[7] = random.randint(0, 30)
            a[8] = random.randint(0, 105)
            a[9] = random.randint(0, 225)
            a[10] = random.randint(0, 19)
            a[11] = random.randint(0, 19)
            return a.reshape(1, 12)

        in0 = torch.rand(1, 1, seq_num_input + 6)
        in0[:, :, -6] = random.randint(0, 1) # is_five
        in0[:, :, -5] = random.randint(0, 1) # is_ten
        in0[:, :, -4] = random.randint(0, 1) # is_clock
        in0[:, :, -3] = random.randint(0, 4) # week_id
        in0[:, :, -2] = random.randint(0, 3) # session_id
        in0[:, :, -1] = random.randint(0, 47) # minute_id
        in1 = torch.rand(1, 16, seq_num_input)
        in2 = random_stock_ids()
        return in0, in1, in2

    def find_and_construct_state_dict(secid):
        shared_sd = None
        finetuned_sd = None
        for secid_list, sd in shared_state_dict_list:
            if secid in secid_list:
                shared_sd = sd
                break
        for secid_list, sd in finetuned_state_dict_list:
            if secid in secid_list:
                finetuned_sd = sd
                break
        if shared_sd is None or finetuned_sd is None:
            raise Exception("cannot find %s state dict" % secid)
        return collections.OrderedDict(list(shared_sd.items()) + list(finetuned_sd.items()))

    msg = {}
    msg['ver'] = 1
    msg['date'] = int(datetime.datetime.now().strftime('%Y%m%d'))
    msg['share'] = pack_state_dict_list_to_msg(shared_state_dict_list)
    msg['finetune'] = pack_state_dict_list_to_msg(finetuned_state_dict_list)

    # generate sample input/output
    msg['sample'] = []
    all_secid_set = set()
    for secid_list, sd in shared_state_dict_list:
        for secid in secid_list:
            all_secid_set.add(secid)
    all_secid_list = list(all_secid_set)
    for i_sample in range(128):
        msg_sample = {}
        secid = random.choice(all_secid_list)
        sd = find_and_construct_state_dict(secid)
        in0, in1, in2 = random_input()
        model.load_state_dict(sd)
        out = model(in0, in1, in2, return_tick=False)
        msg_sample['secid'] = secid
        msg_sample['input'] = [
            pack_tensor_to_msg(in0),
            pack_tensor_to_msg(in1),
            pack_tensor_to_msg(in2)
        ]
        msg_sample['output'] = [
            pack_tensor_to_msg(out)
        ]
        msg['sample'].append(msg_sample)

    msg_data = msgpack.packb(msg, use_bin_type=False)
    enc_msg_data = encrypt(msg_data)
    with open(output_path, 'wb') as fout:
        fout.write(enc_msg_data)
    return True


def split_light_mmoe_state_dict(sd):
    shared = collections.OrderedDict()
    finetuned = collections.OrderedDict()
    for name, tensor in sd.items():
        is_finetuned = any(name.startswith(x) for x in ['expert', 'task_outputs'])
        if is_finetuned:
            finetuned[name] = tensor
        else:
            shared[name] = tensor
    return shared, finetuned


def load_sd_from_model_file(path, secid=1600000):
    enc_msg_data = open(path, 'rb').read()
    msg_data = decrypt(enc_msg_data)
    msg = msgpack.unpackb(msg_data, raw=True)

    def load_model(msg):
        r = collections.OrderedDict()
        for a in msg:
            name = a[b'name'].decode()
            shape = a[b'shape']
            data = a[b'data']
            np_data = np.frombuffer(data, dtype=np.float32).reshape(shape)
            r[name] = torch.from_numpy(np_data)
        return r

    def find_msg_sd(msg_sd_list, secid):
        for msg_sd in msg_sd_list:
            secid_list = msg_sd[b'secids']
            if secid in secid_list:
                return msg_sd

    shared_sd = load_model(find_msg_sd(msg[b'share'], secid)[b'model'])
    finetuned_sd = load_model(find_msg_sd(msg[b'finetune'], secid)[b'model'])
    return collections.OrderedDict(list(shared_sd.items()) + list(finetuned_sd.items()))
    
if __name__ == '__main__':
#def test():
    import sys
    sys.path.append('.')

    from model_zoos.final_alpha_model import LightMMOE
    import yaml
    from yaml.loader import SafeLoader

    bs = 1
    with open('./train_configs/export_config/mbd_model.yaml') as f:
        model_configs = yaml.load(f, Loader=SafeLoader)
    model_configs['device'] = '0'
    model_configs['batch_size'] = bs
    seq_num_input = 259
    light_mmoe = LightMMOE(model_configs).cpu().eval()
    input_x = torch.rand((bs, 1, 265)).cpu()
    alias_x = torch.rand((bs, 16, seq_num_input)).cpu()
    stock_x = torch.ones((bs, 12)).cpu()
    out = light_mmoe(input_x, alias_x, stock_x, return_tick=False)
    print(out)

    sd = light_mmoe.state_dict()
    shared_sd, finetuned_sd = split_light_mmoe_state_dict(sd)
    light_mmoe = LightMMOE(model_configs).cpu().eval()
    sd2 = light_mmoe.state_dict()
    shared_sd2, finetuned_sd2 = split_light_mmoe_state_dict(sd)

    model_path = 'sta4_model_mbd_20220310.en'
    encode_sta4_model(
        light_mmoe,
        [
            ((1600000, 1600006), shared_sd),
            ((2300750, 2300751), shared_sd2)
        ],
        [
            ((1600000, 1600006), finetuned_sd),
            ((2300750, 2300751), finetuned_sd2)
        ],
        seq_num_input,
        model_path)

    sd = load_sd_from_model_file(model_path)
    light_mmoe = LightMMOE(model_configs).cpu().eval()
    out2 = light_mmoe(input_x, alias_x, stock_x, return_tick=False)
    print(out2)
    light_mmoe.load_state_dict(sd)
    out3 = light_mmoe(input_x, alias_x, stock_x, return_tick=False)
    print(out3)
