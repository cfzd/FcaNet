import math
import torch
import torch.nn as nn
from .dct_mul_impl import DctMul, DctMulFreqwise, DctMulSubBlock


class FreqAttentionLayer(torch.nn.Module):
    def __init__(self, channel, fea_h, fea_w, reduction = 16):
        super(FreqAttentionLayer, self).__init__()
        self.reduction = reduction

        self.num_split = 8
        mapper_x = [0,0,1,-1,0,3,0,2]
        mapper_y = [0,1,0,-1,3,0,2,0]


        # if channel > 512:
        #     self.num_split = 8
        #     mapper_x = [0,0,1,-1,0,3,0,2]
        #     mapper_y = [0,1,0,-1,3,0,2,0]
        # else:
        #     self.num_split = 1
        #     mapper_x = [0]
        #     mapper_y = [0]
            

        self.dct_layer = DctMulSubBlock(fea_h, fea_w, mapper_x, mapper_y, channel)

        assert len(mapper_x) == self.num_split == len(mapper_y)
 
        
        # self.fcs = nn.ModuleList([
        #     nn.Sequential(
        #     nn.Linear(channel // self.num_split, channel // reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channel // reduction, channel // self.num_split, bias=False),
        #     nn.Sigmoid()) for _ in range(self.num_split)
        # ])
        self.group_conv1x1_fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction * self.num_split, 1, groups = self.num_split,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction * self.num_split, channel, 1, groups = self.num_split,bias=False),
            nn.Sigmoid()
        )

        self.mutual_fc = nn.Sequential(
            nn.Linear(self.num_split, self.num_split, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_split, self.num_split, bias=False),
            nn.Sigmoid())


    def forward(self, x):

        # x = self.reduction_layer(x)
        # n, c, h, w
        # x_float = x.float()
        # xf_linear = self.dct_layer(x)
        # n,c,h,w


        n,c,h,w = x.shape

        c_part = c // self.num_split 

        dct_freq = self.dct_layer(x)
        # gp = torch.nn.functional.adaptive_avg_pool2d(x,1).squeeze()
        # mul = dct_freq/gp
        # print(c)
        # print(torch.max(mul), torch.min(mul))
        # import pdb; pdb.set_trace()
        #n, c

        freq_weight = self.mutual_fc(dct_freq.view(n, self.num_split, c_part).mean(dim = 2))
        # n, c -> n, 8, c/8 -> n,8
        freq_weight = freq_weight.view(n, self.num_split, 1).repeat(1, 1, c_part).view(n,c,1,1)

        dct_freq = dct_freq.view(n, c, 1, 1)
        att = self.group_conv1x1_fc(dct_freq)

        return att * freq_weight * x


class FreqAttentionLayerSE(torch.nn.Module):
    def __init__(self, channel, fea_h, fea_w, reduction = 16):
        super(FreqAttentionLayerSE, self).__init__()
        self.reduction = reduction

        # self.num_split = 8
        mapper_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2]
        mapper_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2]
        # mapper_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 1]
        # mapper_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 4]
        # mapper_x = [0, 0]
        # mapper_y = [0, 1]

        mapper_x1 = [x * (fea_h // 7) for x in mapper_x]
        # mapper_x2 = [x * (fea_w // 7) for x in mapper_x]
        mapper_y1 = [y * (fea_w // 7) for y in mapper_y]
        # mapper_y2 = [y * (fea_h // 7) for y in mapper_y]

        self.fea_h = fea_h
        self.fea_w = fea_w

        self.dct_layer = DctMul(fea_h, fea_w, mapper_x1, mapper_y1, channel)
        self.embbeding = nn.AdaptiveAvgPool2d((fea_h, fea_w))
        # self.dct_layer2 = DctMul(fea_w, fea_h, mapper_x2, mapper_y2, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape

        # if h == self.fea_h and w == self.fea_w:
        #     y = self.dct_layer1(x)
        # elif h == self.fea_w and w == self.fea_h:
        #     y = self.dct_layer2(x)
        # else:
        #     import pdb; pdb.set_trace()
        #     raise Exception
        emb_x = self.embbeding(x)
        y = self.dct_layer(emb_x)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class FreqAttentionLayerSEFineGrain(torch.nn.Module):
    def __init__(self, channel, fea_h, fea_w, reduction = 16):
        super(FreqAttentionLayerSEFineGrain, self).__init__()
        self.reduction = reduction
        
        self.num_split = 8
        mapper_x = [0,0,1,-1,0,3,0,2]
        mapper_y = [0,1,0,-1,3,0,2,0]

        assert self.num_split == len(mapper_x) == len(mapper_y)

        self.dct_layer = DctMulFreqwise(fea_h, fea_w, mapper_x, mapper_y, channel)
        # self.fc = nn.Sequential(
        #     nn.Linear(channel, channel // reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channel // reduction, channel, bias=False),
        #     nn.Sigmoid()
        # )
        self.group_fc = nn.Sequential(
            nn.Conv1d(channel * self.num_split, channel // reduction * self.num_split, 1, bias=False, groups=self.num_split),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction * self.num_split, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape

        y = self.dct_layer(x).unsqueeze(-1)

        y = self.group_fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)

if __name__ == "__main__":
    test_linear_dct_error()
