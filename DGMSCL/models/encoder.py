import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import torch.nn.functional as F
import numpy as np
from .dilated_conv import DilatedConvEncoder
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier 初始化
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # xavier 初始化

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # 对应 eij 的计算公式
        e = self._prepare_attentional_mechanism_input(Wh)  # 对应 LeakyReLU(eij) 计算公式
        zero_vec = -9e15 * torch.ones_like(e)  # 将没有链接的边设置为负无穷
        attention = torch.where(adj > 0, e, zero_vec)  # 如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留
        attention = F.softmax(attention, dim=1)  # softmax 形状保持不变，得到归一化的注意力
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        h_prime = torch.matmul(attention, Wh)  # 得到由周围节点通过注意力权重进行更新后的表示

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = self.leakyrelu(Wh1 + Wh2.t())
        return e
def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

