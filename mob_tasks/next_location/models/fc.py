import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FullyConnected(nn.Module):
    def __init__(self, d_input, config, total_loc_num, if_residual_layer=True):
        super().__init__()
        self.if_embed_user = config.if_embed_user
        if self.if_embed_user:
            self.emb_user = nn.Embedding(config.total_user_num, d_input)

        self.emb_dropout = nn.Dropout(p=0.1)
        self.if_residual_layer = if_residual_layer
        if self.if_residual_layer:
            self.linear1 = nn.Linear(d_input, d_input * 2)
            self.linear2 = nn.Linear(d_input * 2, d_input)
            self.norm1 = nn.BatchNorm1d(d_input)
            self.fc_dropout1 = nn.Dropout(p=config.fc_dropout)
            self.fc_dropout2 = nn.Dropout(p=config.fc_dropout)

        self.fc_loc = nn.Linear(d_input, total_loc_num)

    def forward(self, out, user) -> Tensor:
        if self.if_embed_user:
            out = out + self.emb_user(user)

        out = self.emb_dropout(out)
        if self.if_residual_layer:
            out = self.norm1(out + self._res_block(out))
        return self.fc_loc(out)

    def _res_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.fc_dropout1(F.relu(self.linear1(x))))
        return self.fc_dropout2(x)
