import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import copy
import numpy as np


class StockEncoder(nn.Module):

    def __init__(self, model_config):
        super(StockEncoder, self).__init__()
        self.model_config = model_config

        self.sw1_embed = nn.Embedding(31, 32)
        # 30 105 225
        self.sw2_embed = nn.Embedding(106, 16)
        self.sw3_embed = nn.Embedding(226, 8)
        self.share_embed = nn.Embedding(20, 16)
        self.value_embed = nn.Embedding(20, 16)
        self.embed_proj = nn.Linear(95, 32)
        self.embed_act = nn.LeakyReLU()

    @classmethod
    def load_model_from_config(cls, model_config):
        print(model_config)
        return cls(model_config)

    def forward(self, x):
        dy_x = x[:, :-5].float()
        sw1_x = self.sw1_embed(x[:, -5].int())
        sw2_x = self.sw2_embed(x[:, -4].int())
        sw3_x = self.sw3_embed(x[:, -3].int())
        mv_x = self.value_embed(x[:, -2].int())
        ms_x = self.share_embed(x[:, -1].int())
        stock_x = torch.cat([dy_x, sw1_x, sw2_x, sw3_x, mv_x, ms_x], dim=1)
        return self.embed_act(self.embed_proj(stock_x))


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class FFRLayer(nn.Module):

    def __init__(self, feat_dim, hidden_dim, dropout):
        super(FFRLayer, self).__init__()
        self.ff_layer = PositionWiseFeedForward(feat_dim, hidden_dim, dropout)
        self.dropout = nn.Dropout()

    def forward(self, x):
        return nn.LeakyReLU()(x + self.dropout(self.ff_layer(x)))


class WideEncoder(nn.Module):
    def __init__(self, model_config):
        super(WideEncoder, self).__init__()
        self.model_config = model_config
        self.dropout = nn.Dropout(model_config['dropout'])
        input_size = model_config['feat_dim'] + model_config['week_class'] \
                     + model_config['session_class'] + model_config['minute_class']
        self.proj_layer = nn.Linear(input_size,
                                    model_config['proj_dim'])
        self.ffp_layer1 = FFRLayer(model_config['proj_dim'], model_config['proj_dim'],
                                   model_config['dropout'])
        self.reduce_layer_1 = nn.Linear(model_config['proj_dim'], 128)

        self.ffp_layer2 = FFRLayer(128, 128 * 2, model_config['dropout'])
        self.reduce_layer_2 = nn.Linear(
            128,
            model_config['output_dim'])

        self.ffp_layer3 = FFRLayer(model_config['output_dim'], model_config['output_dim'] * 4, model_config['dropout'])

    def forward(self, wide_x, week_x, session_x, minute_x):
        """
        wide_x is all features available
        """
        week_x = F.one_hot(week_x.long(), num_classes=self.model_config['week_class'])
        session_x = F.one_hot(session_x.long(), num_classes=self.model_config['session_class'])
        minute_x = F.one_hot(minute_x.long(), num_classes=self.model_config['minute_class'])
        wide_x = torch.cat([wide_x, week_x, session_x, minute_x], dim=-1)

        wide_x = self.dropout(nn.Sigmoid()(self.proj_layer(wide_x)))

        wide_x = self.dropout(self.ffp_layer1(wide_x))
        wide_x = nn.ReLU()(self.reduce_layer_1(wide_x))

        wide_x = self.dropout(self.ffp_layer2(wide_x))
        wide_x = nn.ReLU()(self.reduce_layer_2(wide_x))

        wide_x = self.dropout(self.ffp_layer3(wide_x))
        return wide_x


class RanDeepEncoder(nn.Module):
    def __init__(self, model_config):
        super(RanDeepEncoder, self).__init__()

        self.model_config = model_config
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = model_config['device']
        self.multi_head = MultiHeadedAttention(h=model_config['header_num'], d_model=model_config['field_dim'],
                                               dropout=model_config['dropout'])
        self.field_embed = nn.Embedding(model_config['field_number'], model_config['field_dim'])
        self.agg_att = TemporalAttention(feature_dim=model_config['field_dim'], att_dim=model_config['field_dim'])
        self.dropout = nn.Dropout(model_config['dropout'])
        self.res_block = ResBlock(model_config['field_dim'], model_config['dropout'])

        self.week_embed = nn.Embedding(model_config['week_class'], embedding_dim=model_config['week_dim'])
        self.session_embed = nn.Embedding(model_config['session_class'], embedding_dim=model_config['session_dim'])
        self.minute_embed = nn.Embedding(model_config['minute_class'], embedding_dim=model_config['minute_dim'])
        self.deep_proj = nn.Linear(model_config['field_dim'], model_config['field_dim'])

    def forward(self, deep_x, week_x, session_x, minute_x):
        # get embeddings
        batch_size = deep_x.shape[0]
        field_seq = deep_x.shape[1]
        field_ids = torch.from_numpy(np.asarray([[i for i in range(field_seq)] for _ in range(batch_size)])).to(
            self.device)
        field_x = self.field_embed(field_ids)
        minute_x = self.minute_embed(minute_x).unsqueeze(1)
        week_x = self.week_embed(week_x).unsqueeze(1)
        session_x = self.session_embed(session_x).unsqueeze(1)
        deep_log = torch.sign(deep_x) * torch.log(torch.abs(deep_x))
        deep_x = torch.where((deep_x > 2) | (deep_x < -2), deep_log, deep_x)
        deep_x = deep_x.unsqueeze(-1).repeat(1, 1, self.model_config['field_dim'])

        deep_x = field_x * deep_x
        deep_x = torch.cat([deep_x, week_x, session_x, minute_x], dim=1)
        high_order_x = self.dropout(self.multi_head(deep_x, deep_x, deep_x))

        # residual combine
        deep_x = self.deep_proj(deep_x)
        deep_x = nn.ReLU()(deep_x + high_order_x)

        # aggregate
        deep_x = self.dropout(self.agg_att(deep_x))
        return self.res_block(deep_x)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"

    # size(-1) the last dimensions
    # query key value which is already projected
    d_k = query.size(-1)
    # 维数大，matmul为批量乘法
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    # true mask size is <batch_size, 1, ts, mask_values>,
    # <ts, mask_value> mean for current time i, the mask of all positions, 1 is for computation
    if mask is not None:
        # 把t+1后边的值给盖住
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        # h is the number of head and d_k dimensions per self attention
        self.d_k = d_model // h
        self.h = h
        # the first three linear for attention k q v and the last for final result projection
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # transpose把time step维度娜近一点
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # transpos把 head 和 time step翻回来，
        # view 恢复 batch * ts * dim 的形式
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class TemporalAttention(nn.Module):

    def __init__(self, feature_dim, att_dim):
        super(TemporalAttention, self).__init__()
        self.feature_dim = feature_dim
        self.att_dim = att_dim
        self.weight = nn.Parameter(torch.randn(feature_dim, att_dim), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(att_dim, ), requires_grad=True)
        self.context_vec = nn.Parameter(torch.randn(att_dim, 1), requires_grad=True)

    def forward(self, x):
        # x -> [bs, ts, feat]
        eij = nn.Tanh()(torch.matmul(x, self.weight) + self.bias)
        # eij -> [bs, ts, att_dim]
        eij = torch.matmul(eij, self.context_vec).squeeze(-1)
        # eij -> [bs, ts]
        # below is the same
        a = torch.exp(eij)
        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)
        # weighted
        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class ResBlock(nn.Module):

    def __init__(self, in_dims, dropout):
        super(ResBlock, self).__init__()
        self.in_dims = in_dims
        self.dropout = dropout

        self.proj_1 = nn.Linear(in_dims, in_dims)
        self.act_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(dropout)

        self.proj_2 = nn.Linear(in_dims, in_dims)
        self.act_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        deep_x = self.act_1(self.proj_1(self.dropout_1(x)))
        deep_x = self.act_2(self.proj_2(self.dropout_2(deep_x)))
        x = deep_x + x
        x = self.relu(x)
        return x
