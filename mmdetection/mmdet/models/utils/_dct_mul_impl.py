import math
import torch
import torch.nn as nn



class DctMul(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(DctMul, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        # num_freq, h, w

    def forwardv1(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        n, c, h, w = x.shape
        x = x.contiguous().view(n * c, 1, h, w)

        result = torch.sum(x * self.weight, dim=[2,3])
        return result

    def forwardv2(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        # print(x.shape)
        # print(self.weight.shape)
        x = x * self.weight

        result = torch.sum(x, dim=[2,3])
        return result
    
    def forward(self,x):
        return self.forwardv2(x)
    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_fileter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_fileter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        
        return dct_fileter
    
class DctMulFreqwise(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, width, height, mapper_x, mapper_y, channel):
        super(DctMulFreqwise, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        # num_freq, h, w

    def forwardv1(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        n, c, h, w = x.shape
        x = x.contiguous().view(n * c, 1, h, w)

        result = torch.sum(x * self.weight, dim=[2,3])
        return result

    def forwardv2(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape
        x = x.repeat(1, self.num_freq, 1, 1) * self.weight

        result = torch.sum(x, dim=[2,3])
        return result
    
    def forward(self,x):
        return self.forwardv2(x)
    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_fileter = torch.zeros(channel * len(mapper_x), tile_size_x, tile_size_y)

        # c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_fileter[i * channel: (i+1)*channel, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        
        return dct_fileter


class DctMulSubBlock(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, width, height, mapper_x, mapper_y, channel):
        super(DctMulSubBlock, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        self.register_buffer('weight', self.get_dct_filter(7, 7, mapper_x, mapper_y, channel))
        # num_freq, h, w

    def forwardv1(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        n, c, h, w = x.shape
        x = x.contiguous().view(n * c, 1, h, w)

        result = torch.sum(x * self.weight, dim=[2,3])
        return result

    def forwardv2(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        n, c, h, w = x.shape
        x = x.reshape(n, c, h//7, 7, w//7, 7).permute(0, 1, 2, 4, 3, 5).reshape(n, c, -1, 7, 7)

        x = x * self.weight

        result = torch.sum(x, dim=[3, 4]).mean(-1)
        return result
    
    def forward(self,x):
        return self.forwardv2(x)
    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_fileter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_fileter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        
        return dct_fileter.unsqueeze(1)


# @profile
def test():
    import sys
    sys.path.insert(0, '../')
    from ecanet.dct_conv import DctConv
    from freqnet.layer import Linear2DDCT
    import torch_dct as tdct

    x = torch.rand(1,2,56,56).cuda()

    layer_mul = DctMul(56, 56, [0,1], [0,1], 2).cuda()
    layer_conv = DctConv( 2, 2, [56, 56], [0,1], [0,1]).cuda()
    layer_linear_all = Linear2DDCT(56,56).cuda()

    y = layer_mul(x) # for gpu warm up

    y = layer_mul(x)
    print('mul:',y)

    x = x.double()
    y2 = layer_conv(x)
    print('conv:',y2)

    x = x.float()
    y3 = layer_linear_all(x)
    print('linear_all:',y3[:,:,0,0])
    print('linear_all:',y3[:,:,1,1])


    for ii in range(2):
        y1 = tdct.dct_2d(x[0,ii].cpu(), norm='ortho')
    
        print(y1[0,0],y1[1,1])

def backward_test():
    x = torch.rand(1,2,56,56).double()
    layer = DctMul(56, 56, [0,1], [0,1])
    optimizer = torch.optim.Adam(layer.parameters(), lr = 0.1)
    for i in range(10):
        y = layer(x)
        print(y)
        loss = torch.sum(torch.abs(y))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    test()
    # backward_test()