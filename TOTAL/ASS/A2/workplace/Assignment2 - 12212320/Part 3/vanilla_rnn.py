from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    def __init__(self, input_length, input_dim, hidden_dim, output_dim, batch_size):
        # h_t = tanh(W_hx x_t + W_hh h_{t-1} + b_h)
        # o_t = W_ph h_t + b_o
        super().__init__()
        self.input_length = input_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.W_hx = nn.Linear(input_dim, hidden_dim)   # x_t -> h_t
        self.W_hh = nn.Linear(hidden_dim, hidden_dim)  # h_{t-1} -> h_t
        self.W_ph = nn.Linear(hidden_dim, output_dim)  # h_t -> o_t
        self.tanh = nn.Tanh()


    def forward(self, x):
        if x.dim() == 2:
            B, T = x.size()
            x = x.unsqueeze(-1).float()  # [B, T, 1]
        else:
            B, T, D = x.size()
            x = x.float()
        h = torch.zeros(1, B, self.hidden_dim, device=x.device)  # [1, B, H]

        # 逐时间步递归
        for t in range(T):
            xt = x[:, t, :]                           # [B, input_dim]
            # W_hx(xt) 返回 [B, H]; h.squeeze(0) 是 [B, H]
            h_t = self.tanh(self.W_hx(xt) + self.W_hh(h.squeeze(0)))  # [B, H]
            h = h_t.unsqueeze(0)  # [1, B, H]

        # 输出最后一步的预测
        out = self.W_ph(h.squeeze(0))  # [B, output_dim]
        return out