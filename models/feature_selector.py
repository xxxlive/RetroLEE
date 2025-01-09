import torch.nn as nn
import torch

from models.model_utils import gumbel_softmax


class FocusSelector(nn.Module):
    def __init__(self, in_feature, drop_out):
        super(FocusSelector, self).__init__()
        self.W_ea = nn.Sequential(
            nn.Linear(in_feature, in_feature * 2),
            nn.ReLU(),
            nn.Dropout(p=drop_out),
            nn.Linear(in_feature * 2, 2),
            nn.ReLU()
        )

    def forward(self, x, scopes, hard=True):
        logit = self.W_ea(x)
        res = gumbel_softmax(logits=logit, temperature=1, hard=hard)
        return res


if __name__ == '__main__':
    se = FocusSelector(in_feature=256, drop_out=0.32)
    test = se(torch.randn((32, 256)), None)
    print(test)
