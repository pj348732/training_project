import torch
import torch.nn as nn
from model_zoos.zoo_base import StockEncoder
from torch.nn.utils import weight_norm
import torch.nn.functional as F


def count_parameters(the_model):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = the_model.parameters()
    parameters = list(p for p in parameters if p.requires_grad)
    unique = dict((p.data_ptr(), p) for p in parameters).values()
    return sum(p.numel() for p in unique)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNModel(nn.Module):
    def __init__(self, num_input, output_size, num_channels, kernel_size, dropout):
        super().__init__()
        self.num_input = num_input
        self.tcn_proj = nn.Linear(num_input, num_input)
        self.tcn_dropout = nn.Dropout(dropout)
        self.tcn_act = nn.Tanh()
        self.tcn = TemporalConvNet(num_input, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        print('parameters of tcn %d' % count_parameters(self))

    def forward(self, x):
        x = self.tcn_dropout(self.tcn_act(self.tcn_proj(x)))
        x = x.permute(0, 2, 1)
        output = self.tcn(x)
        output = self.linear(output[:, :, -1])
        return output


class GLU(nn.Module):
    # Gated Linear Unit
    def __init__(self, input_size):
        super(GLU, self).__init__()

        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)


class FeatureTransformer(nn.Module):

    def __init__(self, input_size, output_size, dropout, device='cpu'):
        super(FeatureTransformer, self).__init__()
        self.fc_layer = nn.Linear(input_size, output_size)
        self.glu_layer = GLU(output_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

    def forward(self, x):
        x = self.dropout(self.fc_layer(x))
        x = torch.add(x, self.glu_layer(x))
        return self.dropout(x * self.scale)


class LightWideEncoder(nn.Module):

    def __init__(self, model_config):
        super(LightWideEncoder, self).__init__()
        self.model_config = model_config
        self.dropout = nn.Dropout(model_config['dropout'])
        input_size = model_config['feat_dim'] + model_config['week_class'] \
                     + model_config['session_class'] + model_config['minute_class']

        self.proj_layer = nn.Linear(input_size,
                                    model_config['proj_dim'])
        self.feat_layer_1 = FeatureTransformer(model_config['proj_dim'], 128, model_config['dropout'],
                                               device=model_config['device'])
        self.feat_layer_2 = FeatureTransformer(128, 64, model_config['dropout'],
                                               device=model_config['device'])
        self.feat_layer_3 = FeatureTransformer(64, model_config['output_dim'], model_config['dropout'],
                                               device=model_config['device'])
        print('parameters of wide %d' % count_parameters(self))

    def forward(self, wide_x, week_x, session_x, minute_x):
        """
        wide_x is all features available
        """
        week_x = F.one_hot(week_x.long(), num_classes=self.model_config['week_class'])
        session_x = F.one_hot(session_x.long(), num_classes=self.model_config['session_class'])
        minute_x = F.one_hot(minute_x.long(), num_classes=self.model_config['minute_class'])
        wide_x = torch.cat([wide_x, week_x, session_x, minute_x], dim=-1)

        wide_x = self.dropout(nn.Sigmoid()(self.proj_layer(wide_x)))
        wide_x = self.feat_layer_1(wide_x)
        wide_x = self.feat_layer_2(wide_x)
        wide_x = self.feat_layer_3(wide_x)
        return wide_x


# maximum no more t
class LightTickNet(nn.Module):

    def __init__(self, model_config):
        super(LightTickNet, self).__init__()

        for conf in ['stock_class', 'stock_dim', 'week_class', 'week_dim', 'session_class',
                     'session_dim', 'minute_class', 'minute_dim', 'device', 'batch_size']:
            model_config['wide_encoder'][conf] = model_config[conf]

        self.wide_encoder = LightWideEncoder(model_config['wide_encoder'])
        self.seq_encoder = TCNModel(num_input=model_config['seq_encoder']['num_input'],
                                    output_size=model_config['seq_encoder']['output_size'],
                                    num_channels=model_config['seq_encoder']['num_channels'],
                                    kernel_size=model_config['seq_encoder']['kernel_size'],
                                    dropout=model_config['seq_encoder']['dropout'])

        comb_size = model_config['wide_encoder']['output_dim'] + model_config['seq_encoder']['output_size']

        self.combine_layer = nn.Linear(comb_size + 32, comb_size)
        self.dropout = nn.Dropout(model_config['dropout'])
        self.model_config = model_config
        self.stock_encoder = StockEncoder(model_config)

    def get_tick_embedding(self, x, alias_feats, stock_ids):
        x = x.squeeze(1)
        seq_x = alias_feats
        # wide_x = torch.cat([x[:, :-3], alias_feats[:, -1, :4]], dim=-1)
        wide_x = x[:, :-3]
        week_x = x[:, -3].int()
        session_x = x[:, -2].int()
        minute_x = x[:, -1].int()

        stock_x = self.stock_encoder(stock_ids)
        wide_x = self.wide_encoder(wide_x, week_x, session_x, minute_x)
        seq_x = self.seq_encoder(seq_x)
        comb_x = self.dropout(self.combine_layer(torch.cat([wide_x, seq_x, stock_x], dim=-1)))
        return comb_x

    @classmethod
    def load_model_from_config(cls, model_config):
        print(model_config)
        return cls(model_config)


class LightMMOE(nn.Module):

    def __init__(self, model_config, tick_mode=False):
        super(LightMMOE, self).__init__()
        self.model_config = model_config
        self.tick_net = LightTickNet(model_config)
        self.tick_mode = tick_mode
        if not tick_mode:
            self.comb_size = model_config['wide_encoder']['output_dim'] + model_config['seq_encoder']['output_size']
            self.expert_kernels = nn.ModuleList([nn.Linear(self.comb_size, model_config['expert_size'], bias=True)
                                                 for i in range(model_config['expert_number'])])
            self.expert_gates = nn.ModuleList([nn.Linear(self.comb_size, model_config['expert_number'], bias=True)
                                               for i in range(model_config['task_number'])])
            self.task_outputs = nn.ModuleList([nn.Linear(model_config['expert_size'], 1, bias=True)
                                               for i in range(model_config['task_number'])])
            self.weight_act = nn.Softmax(dim=1)

    def forward(self, x, alias_feats, stock_ids, return_tick=True):

        tick_x = self.tick_net.get_tick_embedding(x, alias_feats, stock_ids)
        if self.tick_mode:
            return tick_x
        # tick x as the expert input
        expert_outs = []
        for i in range(self.model_config['expert_number']):
            expert_outs.append(nn.ReLU()(self.expert_kernels[i](tick_x)).unsqueeze(-1))
        expert_outs = torch.cat(expert_outs, dim=-1)

        expert_weights = []
        for i in range(self.model_config['task_number']):
            expert_weight = self.weight_act(self.expert_gates[i](tick_x))
            expert_weights.append(expert_weight)

        task_outputs = []
        tick_outputs = []

        for i, expert_weight in enumerate(expert_weights):
            expert_weight = expert_weight.unsqueeze(1)
            expert_weight = expert_outs * expert_weight.repeat(1, self.model_config['expert_size'], 1)
            expert_weight = torch.sum(expert_weight, dim=-1)
            tick_outputs.append(expert_weight)
            task_output = self.task_outputs[i](expert_weight)
            task_outputs.append(task_output)

        task_outputs = torch.cat(task_outputs, dim=-1)
        if not return_tick:
            return task_outputs
        else:
            return task_outputs, tick_x

    @classmethod
    def load_model_from_config(cls, model_config):
        print(model_config)
        return cls(model_config)


if __name__ == '__main__':
    import yaml
    from yaml.loader import SafeLoader

    with open('./model_config.yaml') as f:
        model_configs = yaml.load(f, Loader=SafeLoader)
    light_mmoe = LightMMOE(model_configs)

    input_x = torch.rand((1024, 1, 255))
    alias_x = torch.rand((1024, 16, 278))
    stock_x = torch.ones((1024, 12))
    out = light_mmoe(input_x, alias_x, stock_x)
    print(out.shape)
    exit(0)
