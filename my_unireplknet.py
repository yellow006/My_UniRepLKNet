import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

################################################################
# Components of LarK / SmaK Block: SEBlock, FFN, Depthwise Conv / Dilated Reparam Block

class Rearrange(nn.Module):
    def __init__(self, to_nhwc=False):
        super().__init__()
        self.to_nhwc = to_nhwc
    def forward(self, x):
        if self.to_nhwc: return rearrange(x, 'n c h w -> n h w c')
        return rearrange(x, 'n h w c -> n c h w')


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block proposed in SENet (https://arxiv.org/abs/1709.01507)
    We assume the inputs to this layer are (num_samples, channels, height, width)
    """
    def __init__(self, in_channels, latent_dim):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=in_channels, out_channels=latent_dim,
                              kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=latent_dim, out_channels=in_channels,
                            kernel_size=1, stride=1, bias=True)
        self.in_channels = in_channels
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = F.sigmoid(x)
        return inputs * x.view(-1, self.in_channels, 1, 1)


class GRNorm(nn.Module):
    """ GRN (Global Response Normalization) layer
    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
    This implementation is more efficient than the original (https://github.com/facebookresearch/ConvNeXt-V2)
    We assume the inputs to this layer are (num_samples, height, width, channels)
    """
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x


class DRBlock_DWConv(nn.Module):
    """
    input & output shape: num_samples, channels, height, width
    an easy implementation for Dilated Reparam Block
    params: if large_mode == True, it is utilized as DRBlock. Otherwise Depthwise Conv Block.
    """
    def __init__(self, in_channels=3, out_channels=3, kernel_size=7, bias=False, large_mode=True):
        super().__init__()
        self.DRB = nn.ModuleList()
        self.large_mode = large_mode

        if large_mode:
            if kernel_size == 7:
                self.kernel_size = [7, 5, 3, 3]
                self.dilates = [1, 1, 2, 3]
            elif kernel_size == 9:
                self.kernel_size = [9, 5, 5, 3, 3]
                self.dilates = [1, 1, 2, 3, 4]
            elif kernel_size == 11:
                self.kernel_size = [11, 5, 5, 3, 3, 3]
                self.dilates = [1, 1, 2, 3, 4, 5]
            elif kernel_size == 13:
                self.kernel_size = [13, 5, 7, 3, 3, 3]
                self.dilates = [1, 1, 2, 3, 4, 5]
            else:
                raise NotImplementedError("Need implement for large kernel_size: %d"%kernel_size)

            for ker, dil in zip(self.kernel_size, self.dilates):
                self.DRB_sub = nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                            out_channels=out_channels, 
                                            kernel_size=ker, 
                                            stride=1,
                                            padding=(dil * (ker - 1) + 1) // 2,
                                            dilation=dil,
                                            groups=in_channels,
                                            bias=bias),
                                            nn.BatchNorm2d(num_features=out_channels))
                self.DRB.append(self.DRB_sub)
        
        else:
            assert kernel_size in [3, 5]
            self.DRB.append(nn.Conv2d(in_channels=in_channels, 
                                      out_channels=out_channels, 
                                      kernel_size=kernel_size,
                                      padding=kernel_size//2,
                                      dilation=1,
                                      groups=in_channels))
        
    def forward(self, x):
        if self.large_mode:
            output = 0.
            for _, drb_sub in enumerate(self.DRB):
                output += drb_sub(x)
            return output

        return self.DRB[0](x)
    

class FeedForwardNetwork(nn.Module):
    def __init__(self, dim=3, ffn_factor=4):
        super().__init__()
        self.ffnup = nn.Sequential(Rearrange(to_nhwc=True),
                                   nn.Linear(dim, ffn_factor*dim))
        self.nonlinear = nn.Sequential(nn.GELU(),
                                       GRNorm(dim=dim*ffn_factor, use_bias=True))
        self.ffndown = nn.Sequential(nn.Linear(dim*ffn_factor, dim),
                                     Rearrange(to_nhwc=False),
                                     nn.BatchNorm2d(num_features=dim))
    
    def forward(self, x):
        x = self.ffnup(x)
        x = self.nonlinear(x)
        x = self.ffndown(x)
        return x
    

class LayerNorm(nn.Module):
    r""" LayerNorm implementation used in ConvNeXt
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    # try to reshape (n, c, h, w -> n, h, w, c -> n, c, h, w) when channel first.
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last", reshape_last_to_first=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
        self.reshape_last_to_first = reshape_last_to_first

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


################################################################
# Implementation of LarK / SmaK Block
class LSK_Block(nn.Module):
    def __init__(self, 
                 dim, 
                 kernel_size, 
                 ffn_factor=4, 
                 drop_rate=0., 
                 layer_scale_init_value=1e-6, 
                 large_mode=True):
        super().__init__()

        self.batchnorm = nn.BatchNorm2d(num_features=dim)
        self.seblock = SEBlock(dim, dim // 4)
        if large_mode: self.DRBlock = DRBlock_DWConv(in_channels=dim,
                                                     out_channels=dim,
                                                     kernel_size=kernel_size,
                                                     large_mode=True)
        else: self.DRBlock = DRBlock_DWConv(in_channels=dim,
                                            out_channels=dim,
                                            kernel_size=kernel_size,
                                            large_mode=False)
        self.ffn = FeedForwardNetwork(dim=dim, ffn_factor=ffn_factor)
        self.dropout = nn.Dropout(drop_rate)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def compute_non_identity(self, x):
        x = self.batchnorm(self.DRBlock(x))
        x = self.seblock(x)
        x = self.ffn(x)
        x = self.gamma.view(1, -1, 1, 1) * x
        x = self.dropout(x)
        return x
    
    def forward(self, x):
        return x + self.compute_non_identity(x)
    

################################################################
# Implementation of UniRepLKNet
class MyUniRepLKNet(nn.Module):
    def __init__(self,
                 in_channels=3, 
                 classes=10,
                 depth=(2,2,6,2),
                 dims=(40, 80, 160, 320),
                 drop_rate=0.,
                 layer_scale_init_value=1e-6,
                 big_kernel_size=9,
                 small_kernel_size=3):
        super().__init__()
        
        self.downsample = nn.ModuleList()
        self.downsample.append(nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=dims[0]//2, kernel_size=3, stride=2, padding=1),
            LayerNorm(dims[0]//2, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(in_channels=dims[0]//2, out_channels=dims[0], kernel_size=3, stride=2, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")))
        for i in range(3):
            self.downsample.append(nn.Sequential(
                nn.Conv2d(in_channels=dims[i], out_channels=dims[i+1], kernel_size=3, stride=2, padding=1),
                LayerNorm(dims[i+1], eps=1e-6, data_format="channels_first")))
            
        self.stages = nn.ModuleList()
        for i in range(4):
            if i == 0 or i == 2:
                sub_stage = nn.Sequential(*[LSK_Block(kernel_size=small_kernel_size, 
                                                     dim = dims[i],
                                                     drop_rate=drop_rate, 
                                                     layer_scale_init_value=layer_scale_init_value, 
                                                     large_mode=False) for j in range(depth[i])])
            else:
                sub_stage = nn.Sequential(*[LSK_Block(kernel_size=big_kernel_size, 
                                                     dim = dims[i],
                                                     drop_rate=drop_rate, 
                                                     layer_scale_init_value=layer_scale_init_value, 
                                                     large_mode=True) for j in range(depth[i])])
            # else:
            #     sub_stage = nn.Sequential(*[LSK_Block(kernel_size=big_kernel_size if not j % 3 else small_kernel_size,
            #                                          dim = dims[i],
            #                                          drop_rate=drop_rate,
            #                                          layer_scale_init_value=layer_scale_init_value,
            #                                          large_mode = not j % 3) for j in range(depth[i])])
            self.stages.append(sub_stage)

        self.last_dim = dims[-1]
        self.norm = nn.LayerNorm(self.last_dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(self.last_dim, classes))

    def forward(self, x):
        for stage_idx in range(4):
            x = self.downsample[stage_idx](x)
            x = self.stages[stage_idx](x)
        x = self.norm(x.mean([-2, -1]))
        x = self.mlp(x)
        return x

                                                     

