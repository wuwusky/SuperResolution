import torch
import torch.nn as nn
import math
import torch.nn.functional as F



class net_FSRCNN(nn.Module):
    def __init__(self, in_channels):
        super(net_FSRCNN, self).__init__()

        self.stage_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
        )

        self.layers = []
        self.layers.append(nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
        ))

        for _ in range(3):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0),
                nn.PReLU(),
            ))
        
        self.layers.append(nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
        ))
        
        self.stage_2 = nn.Sequential(*self.layers)

        self.stage_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=in_channels, kernel_size=3, stride=4, padding=1, output_padding=3),
        )
    
    def forward(self, x):
        out = self.stage_1(x)
        # print(out.shape)
        out = self.stage_2(out)
        out = self.stage_3(out)
        return out

class net_SPCNN(nn.Module):
    def __init__(self, upscale):
        super(net_SPCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=3, padding=1, bias=True),
            nn.Tanh(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.Tanh()
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels=32*2*2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.PixelShuffle(2),
            nn.Tanh(),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels=32*3*3, kernel_size=1, stride=1, padding=0, bias=True),
            nn.PixelShuffle(3),
            nn.Tanh(),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=3*upscale*upscale, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PixelShuffle(upscale),
            nn.Sigmoid(),
        )
    def forward(self, x):
        out = self.layer1(x)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        out = self.layer5(out)
        # print(out.shape)
        return out

class net_SR_1(nn.Module):
    def __init__(self):
        super(net_SR_1, self).__init__()
        self.num_uint = 128
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.num_uint, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(self.num_uint),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(self.num_uint),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint, kernel_size=1, padding=0, bias=True),
        )
        self.layer3 = nn.Sequential(
            nn.BatchNorm2d(self.num_uint),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint, kernel_size=3, stride=3, padding=1, bias=True),
        )
        self.layer4 = nn.Sequential(
            nn.BatchNorm2d(self.num_uint),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint, kernel_size=1, padding=0, bias=True),
        )
        self.layer5 = nn.Sequential(
            nn.BatchNorm2d(self.num_uint),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint, kernel_size=1, padding=0, bias=True),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint*3*3, kernel_size=1, padding=0, bias=True),
            nn.PixelShuffle(3),
            nn.Tanh(),
        )
        self.layer7 = nn.Sequential(
            nn.BatchNorm2d(self.num_uint),
            nn.Tanh(),
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint, kernel_size=1, padding=0, bias=True),
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint*2*2, kernel_size=1, padding=0, bias=True),
            nn.PixelShuffle(2),
            nn.Tanh(),
        )
        self.layer9 = nn.Sequential(
            nn.BatchNorm2d(self.num_uint),
            nn.Tanh(),
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint, kernel_size=1, padding=0, bias=True),
        )
        self.layer9_out_aux = nn.Sequential(
            nn.Conv2d(in_channels=self.num_uint, out_channels=3, kernel_size=1, padding=0, bias=True),
            nn.Tanh(),
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint*2*2, kernel_size=1, padding=0, bias=True),
            nn.PixelShuffle(2),
            nn.Tanh(),
        )
        self.layer11 = nn.Sequential(
            nn.BatchNorm2d(self.num_uint),
            nn.Tanh(),
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint, kernel_size=1, padding=0, bias=True),
        )
        self.layer11_out_aux = nn.Sequential(
            nn.Conv2d(in_channels=self.num_uint, out_channels=3, kernel_size=1, padding=0, bias=True),
            nn.Tanh(),
        )
        self.layer12 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint*2*2, kernel_size=1, padding=0, bias=True),
            nn.PixelShuffle(2),
            nn.Tanh(),
        )
        self.layer13 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_uint, out_channels=3, kernel_size=1, padding=0, bias=True),
            nn.Tanh(),
        )
    def forward(self, x):
        out_2_d = self.layer1(x)
        out = self.layer2(out_2_d)
        out_3_d = self.layer3(out)
        out = self.layer4(out_3_d)
        out = self.layer5(out) + out_3_d
        out = self.layer6(out)
        out = self.layer7(out) + out_2_d
        out = self.layer8(out)
        out = self.layer9(out)
        out_aux_1 = self.layer9_out_aux(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out_aux_2 = self.layer11_out_aux(out)
        out = self.layer12(out)
        out = self.layer13(out)
        if self.training:
            return out
        else:
            return out

class net_SR_1_sim(nn.Module):
    def __init__(self):
        super(net_SR_1_sim, self).__init__()
        self.num_uint = 32

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.num_uint, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(self.num_uint),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(self.num_uint),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint, kernel_size=3, padding=1, bias=True),
        )
        self.layer3 = nn.Sequential(
            nn.BatchNorm2d(self.num_uint),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint, kernel_size=3, stride=3, padding=1, bias=True),
        )
        self.layer4 = nn.Sequential(
            nn.BatchNorm2d(self.num_uint),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint, kernel_size=3, padding=1, bias=True),
        )
        self.layer5 = nn.Sequential(
            nn.BatchNorm2d(self.num_uint),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint, kernel_size=3, padding=1, bias=True),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint*3*3, kernel_size=1, padding=0, bias=True),
            nn.PixelShuffle(3),
            nn.ReLU(inplace=True),
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_uint, out_channels=8*2*2, kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )
        self.layer7_aux = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8*4*4, kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(4),
            nn.ReLU(inplace=True),
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )
    def forward(self, input):
        out_2_d = self.layer1(input)
        out = self.layer2(out_2_d)
        out_3_d = self.layer3(out)
        out = self.layer4(out_3_d)
        out = self.layer5(out) + out_3_d
        out = self.layer6(out) + out_2_d
        out = self.layer7(out)
        out_aux_1 = self.layer7_aux(out)
        out = self.layer8(out)
        out = self.layer9(out)
        if self.training:
            return out
        else:
            return out

class net_SR_2(nn.Module):
    def __init__(self):
        super(net_SR_2, self).__init__()
        self.num_uint = 128

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.num_uint, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint, kernel_size=3, stride=3, padding=1, bias=False),
            nn.BatchNorm2d(self.num_uint),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(self.num_uint),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint, kernel_size=1, padding=0, bias=False),
        )
        self.layer3 = nn.Sequential(
            nn.BatchNorm2d(self.num_uint),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint, kernel_size=5, stride=5, padding=2, bias=False),
        )
        self.layer4 = nn.Sequential(
            nn.BatchNorm2d(self.num_uint),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint, kernel_size=1, padding=0, bias=False),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint*5*5, kernel_size=1, padding=0, bias=True),
            nn.PixelShuffle(5),
            nn.ReLU(inplace=True),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint*3*3, kernel_size=1, padding=0, bias=True),
            nn.PixelShuffle(3),
            nn.ReLU(inplace=True),
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint*2*2, kernel_size=1, padding=0, bias=True),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_uint, out_channels=self.num_uint*2*2, kernel_size=1, padding=0, bias=True),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )
        self.layer8_out = nn.Sequential(
            nn.Conv2d(in_channels=self.num_uint, out_channels=3, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

        self.layer7_aux = nn.Sequential(
            nn.Conv2d(in_channels=self.num_uint, out_channels=3, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        # self.upsample = nn.UpsamplingBilinear2d(scale_factor=4)
    def forward(self, input):
        out_3_d = self.layer1(input)
        out = self.layer2(out_3_d)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        out = self.layer5(out) + out_3_d
        out = self.layer6(out)
        out = self.layer7(out)
        if self.training:
            out_aux_1 = self.layer7_aux(out)
        out = self.layer8(out)
        out = self.layer8_out(out)
        if self.training:
            return out, out_aux_1
        else:
            return out

def basic_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size//2), bias=bias)

class ResBlock(nn.Module):
    def __init__(self, conv_block, n_features, kernel_size, bias=True, act=nn.ReLU(True), bn=False, res_scale=1):
        super(ResBlock, self).__init__()
        layers = []
        for i in range(2):
            layers.append(basic_conv(n_features, n_features, kernel_size, bias=bias))
            if bn:
                layers.append(nn.BatchNorm2d(n_features))
            if i == 0:
                layers.append(act)
        self.layers = nn.Sequential(*layers)
        self.res_scale = res_scale
    def forward(self, x):
        res = self.layers(x)
        res = torch.mul(res, self.res_scale)
        out = x + res
        return out

class net_SR_3(nn.Module):
    def __init__(self, n_features=64, n_blocks=32):
        super(net_SR_3, self).__init__()
        k_size = 3
        act = nn.ReLU(inplace=True)

        self.stem = nn.Sequential(
            basic_conv(3, n_features, kernel_size=k_size),
        )

        layers = []
        for _ in range(n_blocks):
            layers.append(ResBlock(basic_conv, n_features, kernel_size=k_size, act=act, res_scale=1.0))
        layers.append(basic_conv(n_features, n_features, k_size))
        self.res_layer = nn.Sequential(*layers)

        self.layer_out = nn.Sequential(
            basic_conv(n_features, 3*2*2,3, bias=True),
            nn.PixelShuffle(2),
            basic_conv(3, 3*2*2,3, bias=True),
            nn.PixelShuffle(2),
            # basic_conv(n_features, 3, kernel_size=k_size),
            nn.Sigmoid()
        )
    def forward(self, input):
        out = self.stem(input)
        res = self.res_layer(out)
        # res = F.dropout2d(res, p=0.15, training=self.training, inplace=True)
        out = out + res
        out = self.layer_out(out)
        return out

class ResBlock_W(nn.Module):
    def __init__(self, conv_block, n_features, kernel_size, bias=True, act=nn.ReLU(True), bn=False, res_scale=0.5):
        super(ResBlock_W, self).__init__()
        wn = lambda x:torch.nn.utils.weight_norm(x)
        self.res = nn.Sequential(
            wn(basic_conv(n_features, n_features*6, 1)),
            act,
            wn(basic_conv(n_features*6, int(n_features*0.8), 1)),
            wn(basic_conv(int(n_features*0.8), n_features, 3)),
        )
        self.res_scale = res_scale
    def forward(self, x):
        res = self.res(x)
        # res = F.dropout2d(res, p=0.1, training=self.training)
        res = torch.mul(res, self.res_scale)
        out = res + x
        return out

class ResBlock_W_SE(nn.Module):
    def __init__(self, conv_block, n_features, kernel_size, bias=True, act=nn.ReLU(True), bn=False, res_scale=1):
        super(ResBlock_W_SE, self).__init__()
        wn = lambda x:torch.nn.utils.weight_norm(x)
        self.res = nn.Sequential(
            wn(basic_conv(n_features, n_features*6, 1)),
            act,
            wn(basic_conv(n_features*6, int(n_features*0.8), 1)),
            wn(basic_conv(int(n_features*0.8), n_features, 3)),
        )
        self.res_scale = res_scale

        self.se_block_1 = nn.AdaptiveAvgPool2d(1)
        self.se_block_2 = nn.Sequential(
            nn.Linear(n_features, int(n_features/16)),
            act,
            nn.Linear(int(n_features/16), n_features),
            nn.Sigmoid(),
        )
    def forward(self, x):
        res = self.res(x)
        res_weight = self.se_block_1(res)
        res_weight = res_weight.view(res_weight.size(0), -1)
        res_weight = self.se_block_2(res_weight)
        res_weight = res_weight.view(res_weight.size(0), res_weight.size(1), 1, 1)
        res = torch.mul(res, res_weight)

        # res = F.dropout2d(res, p=0.1, training=self.training)

        res = torch.mul(res, self.res_scale)
        out = res + x
        
        return out

class net_SR_4(nn.Module):
    def __init__(self, n_features=32, n_blocks=8):
        wn = lambda x:torch.nn.utils.weight_norm(x)
        super(net_SR_4, self).__init__()
        k_size = 3
        act = nn.LeakyReLU(inplace=True)
        self.stem = nn.Sequential(
            wn(basic_conv(3, n_features, kernel_size=k_size)),
        )
        self.layer_out_res = nn.Sequential(
            wn(basic_conv(n_features, 3*2*2, k_size)),
            nn.PixelShuffle(2),
            wn(basic_conv(3, 3*2*2, k_size)),
            nn.PixelShuffle(2),
        )
        self.layer_out_in = nn.Sequential(
            wn(basic_conv(3, 3*2*2, k_size)),
            nn.PixelShuffle(2),
            wn(basic_conv(3, 3*2*2, k_size)),
            nn.PixelShuffle(2),
        )
        layers = []
        for _ in range(n_blocks):
            layers.append(ResBlock_W(basic_conv, n_features, kernel_size=k_size, act=act, res_scale=1.0))
        layers.append(wn(basic_conv(n_features, n_features, k_size)))
        self.res_layer = nn.Sequential(*layers)
        self.act = nn.Sigmoid()
    def forward(self, x):
        out_res = self.layer_out_res(self.res_layer(self.stem(x)))
        out_in = self.layer_out_in(x)
        out = out_res + out_in
        # out = self.act(out)
        return out

class net_SR_4_sub(nn.Module):
    def __init__(self, n_features=32, n_blocks=3):
        wn = lambda x:torch.nn.utils.weight_norm(x)
        super(net_SR_4_sub, self).__init__()
        k_size = 3
        act = nn.LeakyReLU(True)
        self.stem = nn.Sequential(
            wn(basic_conv(3, n_features, kernel_size=k_size)),
        )
        self.layer_out_res = nn.Sequential(
            wn(basic_conv(n_features, 3*2*2, k_size)),
            nn.PixelShuffle(2),
            wn(basic_conv(3, 3*2*2, k_size)),
            nn.PixelShuffle(2),
        )
        self.layer_out_in = nn.Sequential(
            wn(basic_conv(3, 3*2*2, k_size)),
            nn.PixelShuffle(2),
            wn(basic_conv(3, 3*2*2, k_size)),
            nn.PixelShuffle(2),
        )
        layers = []
        for _ in range(n_blocks):
            layers.append(ResBlock_W_SE(basic_conv, n_features, kernel_size=k_size, act=act, res_scale=1.0))
        layers.append(wn(basic_conv(n_features, n_features, k_size)))
        self.res_layer = nn.Sequential(*layers)
        self.act = nn.Sigmoid()
    def forward_sub(self, x):
        out_res = self.layer_out_res(self.res_layer(self.stem(x)))
        out_in = self.layer_out_in(x)
        out = out_res + out_in
        out = self.act(out)
        return out
    def forward(self, x_subs):
        out_1 = self.forward_sub(x_subs[0])
        out_2 = self.forward_sub(x_subs[1])
        out_3 = self.forward_sub(x_subs[2])
        out_4 = self.forward_sub(x_subs[3])
        return out_1, out_2, out_3, out_4
  
class net_SR_5(nn.Module):
    def __init__(self, n_features=32, n_blocks=8):
        wn = lambda x:torch.nn.utils.weight_norm(x)
        super(net_SR_5, self).__init__()
        k_size = 3
        act = nn.PReLU()
        self.stem = nn.Sequential(
            wn(basic_conv(3, n_features, kernel_size=k_size)),
        )
        self.layer_out_res = nn.Sequential(
            wn(basic_conv(n_features, 3*2*2, k_size)),
            nn.PixelShuffle(2),
            wn(basic_conv(3, 3*2*2, k_size)),
            nn.PixelShuffle(2),
        )
        self.layer_out_in = nn.Sequential(
            wn(basic_conv(3, 3*2*2, k_size)),
            nn.PixelShuffle(2),
            wn(basic_conv(3, 3*2*2, k_size)),
            nn.PixelShuffle(2),
        )
        layers = []
        for _ in range(n_blocks):
            layers.append(ResBlock_W_SE(basic_conv, n_features, kernel_size=k_size, act=act, res_scale=1.0))
        layers.append(wn(basic_conv(n_features, n_features, k_size)))
        self.res_layer = nn.Sequential(*layers)
        self.act = nn.Sigmoid()
    def forward(self, x):
        out_res = self.layer_out_res(self.res_layer(self.stem(x)))
        out_in = self.layer_out_in(x)
        # out_src = nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        out = out_res + out_in
        # out = out + torch.mul(out_src, 0.25)
        out = self.act(out)
        return out

class atm(nn.Module):
    def __init__(self, n_features=32):
        super(atm, self).__init__()
        k_size = 3
        act = nn.ReLU(inplace=True)
        self.mask = nn.Sequential(
            nn.BatchNorm2d(3),
            act,
            basic_conv(3, n_features*4, kernel_size=k_size),
            nn.BatchNorm2d(n_features*4),
            act,
            basic_conv(n_features*4, n_features, kernel_size=k_size),
            nn.BatchNorm2d(n_features),
            act,
            basic_conv(n_features, 1, kernel_size=1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        out = self.mask(x)
        return out

class net_SR_5_filter(nn.Module):
    def __init__(self, n_features=32, n_blocks=8):
        wn = lambda x:torch.nn.utils.weight_norm(x)
        super(net_SR_5_filter, self).__init__()
        k_size = 3
        act = nn.PReLU()
        self.stem = nn.Sequential(
            wn(basic_conv(3, n_features, kernel_size=k_size)),
        )
        self.layer_out_res = nn.Sequential(
            wn(basic_conv(n_features, 3*2*2, k_size)),
            nn.PixelShuffle(2),
            wn(basic_conv(3, 3*2*2, k_size)),
            nn.PixelShuffle(2),
        )
        self.layer_out_in = nn.Sequential(
            wn(basic_conv(3, 3*2*2, k_size)),
            nn.PixelShuffle(2),
            wn(basic_conv(3, 3*2*2, k_size)),
            nn.PixelShuffle(2),
        )
        layers = []
        for _ in range(n_blocks):
            layers.append(ResBlock_W_SE(basic_conv, n_features, kernel_size=k_size, act=act, res_scale=1.0))
        layers.append(wn(basic_conv(n_features, n_features, k_size)))
        self.res_layer = nn.Sequential(*layers)
        self.atm = atm()

    def forward(self, x):
        out_res = self.layer_out_res(self.res_layer(self.stem(x)))
        out_in = self.layer_out_in(x)
        out = out_res*self.atm(out_res) + out_in*self.atm(out_in)
        return out


class net_SR_5_ex(nn.Module):
    def __init__(self, n_features=32, n_blocks=8):
        wn = lambda x:torch.nn.utils.weight_norm(x)
        super(net_SR_5_ex, self).__init__()
        k_size = 3
        act = nn.PReLU()
        self.stem = nn.Sequential(
            wn(basic_conv(3, n_features, kernel_size=k_size)),
        )
        self.layer_out_res = nn.Sequential(
            wn(basic_conv(n_features, 3*2*2, k_size)),
            nn.PixelShuffle(2),
            wn(basic_conv(3, 3*2*2, k_size)),
            nn.PixelShuffle(2),
        )
        self.layer_out_in = nn.Sequential(
            wn(basic_conv(3, 3*2*2, k_size)),
            nn.PixelShuffle(2),
            wn(basic_conv(3, 3*2*2, k_size)),
            nn.PixelShuffle(2),
        )
        layers = []
        for _ in range(n_blocks):
            layers.append(ResBlock_W_SE(basic_conv, n_features, kernel_size=k_size, act=act, res_scale=1.0))
        layers.append(wn(basic_conv(n_features, n_features, k_size)))
        self.res_layer = nn.Sequential(*layers)
        self.act = nn.Tanh()
    def forward_single(self, x):
        out_res = self.layer_out_res(self.res_layer(self.stem(x)))
        out_in = self.layer_out_in(x)
        # out_src = nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        out = out_res + out_in
        # out = self.act(out)
        # out = out + torch.mul(out_src, 0.25)
        return out
    def forward(self, x_subs):
        out_1 = self.forward_single(x_subs[0])
        out_2 = self.forward_single(x_subs[1])
        out_3 = self.forward_single(x_subs[2])
        out_4 = self.forward_single(x_subs[3])
        return out_1, out_2, out_3, out_4

##---------------------------------------------------------------------------------------------------
class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))
    def forward(self, input):
        return input*self.scale
class AWRU(nn.Module):
    def __init__(self, n_feats, k_size, block_feats, wn, res_scale=1, act=nn.ReLU(True)):
        super(AWRU, self).__init__()
        self.res_scale = Scale(res_scale)
        self.x_scale = Scale(1)
        self.body = nn.Sequential(
            wn(basic_conv(n_feats, block_feats, k_size)),
            act,
            wn(basic_conv(block_feats, n_feats, k_size)),
        )
    def forward(self, x):
        res = self.res_scale(self.body(x)) + self.x_scale(x)
        return res
class AWMS(nn.Module):
    def __init__(self, scale, n_feats, k_size, wn):
        super(AWMS, self).__init__()
        out_feats = scale*scale*3
        self.tail_k3 = wn(basic_conv(n_feats, out_feats, 3))
        self.tail_k5 = wn(basic_conv(n_feats, out_feats, 5))
        self.tail_k7 = wn(basic_conv(n_feats, out_feats, 7))
        self.tail_k9 = wn(basic_conv(n_feats, out_feats, 9))
        self.PixelShuffle = nn.PixelShuffle(scale)
        self.scale_k3 = Scale(0.25)
        self.scale_k5 = Scale(0.25)
        self.scale_k7 = Scale(0.25)
        self.scale_k9 = Scale(0.25)
    def forward(self, x):
        x0 = self.PixelShuffle(self.scale_k3(self.tail_k3(x)))
        x1 = self.PixelShuffle(self.scale_k5(self.tail_k5(x)))
        x2 = self.PixelShuffle(self.scale_k7(self.tail_k7(x)))
        x3 = self.PixelShuffle(self.scale_k9(self.tail_k9(x)))
        return x0 + x1 + x2 + x3

class LFB(nn.Module):
    def __init__(self, n_feats, kernel_size, block_feats, n_awru, wn, res_scale, act=nn.ReLU(True)):
        super(LFB, self).__init__()
        self.n = n_awru
        self.lfl = nn.ModuleList([AWRU(n_feats, kernel_size, block_feats, wn=wn, res_scale=res_scale, act=act)
            for i in range(self.n)])

        self.reduction = wn(nn.Conv2d(n_feats*self.n, n_feats, kernel_size, padding=kernel_size//2))
        self.res_scale = Scale(res_scale)
        self.x_scale = Scale(1)

    def forward(self, x):
        out=[]
        for i in range(self.n):
            x = self.lfl[i](x)
            out.append(x)
        res = self.reduction(torch.cat(out,dim=1))
        return self.res_scale(res) + self.x_scale(x)

class net_SR_6(nn.Module):
    def __init__(self):
        super(net_SR_6, self).__init__()

        # hyper-params
        scale = 4
        n_resblocks = 5
        n_feats = 32
        kernel_size = 3
        res_scale = 1.0
        n_awru = 3
        block_feats = 64
        act = nn.ReLU(True)
        # wn = lambda x: x
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(
            [0.5, 0.5, 0.5])).view([1, 3, 1, 1])
        # define head module
        # head = HEAD(args, n_feats, kernel_size, wn)
        head = []
        head.append(
            wn(nn.Conv2d(3, n_feats, 3, padding=3//2)))
        # define body module
        body = []
        for i in range(n_resblocks):
            body.append(
                LFB(n_feats, kernel_size, block_feats, n_awru, wn=wn, res_scale=res_scale, act=act))
        # define tail module
        out_feats = scale*scale*3
        tail = AWMS(scale, n_feats, kernel_size, wn)
        skip = []
        skip.append(
            wn(nn.Conv2d(3, out_feats, 3, padding=3//2))
        )
        skip.append(nn.PixelShuffle(scale))
        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = tail
        self.skip = nn.Sequential(*skip)
    def forward(self, x):
        # x = x - self.rgb_mean.cuda(0)
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s
        # x = F.sigmoid(x)
        # x = x + self.rgb_mean.cuda(0)
        return x

            



##---------------------------------------------------------------------------------------------------

class DenseBlock(torch.nn.Module):

    def __init__(self, input_size, output_size, bias=True, activation='relu', norm='batch'):

        super(DenseBlock, self).__init__()

        self.fc = torch.nn.Linear(input_size, output_size, bias=bias)



        self.norm = norm

        if self.norm =='batch':

            self.bn = torch.nn.BatchNorm1d(output_size)

        elif self.norm == 'instance':

            self.bn = torch.nn.InstanceNorm1d(output_size)



        self.activation = activation

        if self.activation == 'relu':

            self.act = torch.nn.ReLU(True)

        elif self.activation == 'prelu':

            self.act = torch.nn.PReLU()

        elif self.activation == 'lrelu':

            self.act = torch.nn.LeakyReLU(0.2, True)

        elif self.activation == 'tanh':

            self.act = torch.nn.Tanh()

        elif self.activation == 'sigmoid':

            self.act = torch.nn.Sigmoid()



    def forward(self, x):

        if self.norm is not None:

            out = self.bn(self.fc(x))

        else:

            out = self.fc(x)



        if self.activation is not None:

            return self.act(out)

        else:

            return out

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)
        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)
        if self.activation is not None:
            return self.act(out)
        else:
            return out

class DeconvBlock(torch.nn.Module):

    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):

        super(DeconvBlock, self).__init__()

        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)



        self.norm = norm

        if self.norm == 'batch':

            self.bn = torch.nn.BatchNorm2d(output_size)

        elif self.norm == 'instance':

            self.bn = torch.nn.InstanceNorm2d(output_size)



        self.activation = activation

        if self.activation == 'relu':

            self.act = torch.nn.ReLU(True)

        elif self.activation == 'prelu':

            self.act = torch.nn.PReLU()

        elif self.activation == 'lrelu':

            self.act = torch.nn.LeakyReLU(0.2, True)

        elif self.activation == 'tanh':

            self.act = torch.nn.Tanh()

        elif self.activation == 'sigmoid':

            self.act = torch.nn.Sigmoid()



    def forward(self, x):

        if self.norm is not None:

            out = self.bn(self.deconv(x))

        else:

            out = self.deconv(x)



        if self.activation is not None:

            return self.act(out)

        else:

            return out

class ResnetBlock(torch.nn.Module):

    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm='batch'):

        super(ResnetBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)

        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)



        self.norm = norm

        if self.norm == 'batch':

            self.bn = torch.nn.BatchNorm2d(num_filter)

        elif norm == 'instance':

            self.bn = torch.nn.InstanceNorm2d(num_filter)



        self.activation = activation

        if self.activation == 'relu':

            self.act = torch.nn.ReLU(True)

        elif self.activation == 'prelu':

            self.act = torch.nn.PReLU()

        elif self.activation == 'lrelu':

            self.act = torch.nn.LeakyReLU(0.2, True)

        elif self.activation == 'tanh':

            self.act = torch.nn.Tanh()

        elif self.activation == 'sigmoid':

            self.act = torch.nn.Sigmoid()





    def forward(self, x):

        residual = x

        if self.norm is not None:

            out = self.bn(self.conv1(x))

        else:

            out = self.conv1(x)



        if self.activation is not None:

            out = self.act(out)



        if self.norm is not None:

            out = self.bn(self.conv2(out))

        else:

            out = self.conv2(out)



        out = torch.add(out, residual)

        return out

class UpBlock(torch.nn.Module):

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):

        super(UpBlock, self).__init__()

        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)        



    def forward(self, x):

    	h0 = self.up_conv1(x)

    	l0 = self.up_conv2(h0)

    	h1 = self.up_conv3(l0 - x)

    	return h1 + h0

class UpBlockPix(torch.nn.Module):

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, scale=4, bias=True, activation='prelu', norm=None):

        super(UpBlockPix, self).__init__()

        self.up_conv1 = Upsampler(scale,num_filter)

        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

        self.up_conv3 = Upsampler(scale,num_filter)        



    def forward(self, x):

    	h0 = self.up_conv1(x)

    	l0 = self.up_conv2(h0)

    	h1 = self.up_conv3(l0 - x)

    	return h1 + h0
      
class D_UpBlock(torch.nn.Module):

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):

        super(D_UpBlock, self).__init__()

        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None)

        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)        



    def forward(self, x):

    	x = self.conv(x)

    	h0 = self.up_conv1(x)

    	l0 = self.up_conv2(h0)

    	h1 = self.up_conv3(l0 - x)

    	return h1 + h0
class D_UpBlockPix(torch.nn.Module):

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, scale=4, bias=True, activation='prelu', norm=None):

        super(D_UpBlockPix, self).__init__()

        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None)

        self.up_conv1 = Upsampler(scale,num_filter)

        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

        self.up_conv3 = Upsampler(scale,num_filter)



    def forward(self, x):

    	x = self.conv(x)

    	h0 = self.up_conv1(x)

    	l0 = self.up_conv2(h0)

    	h1 = self.up_conv3(l0 - x)

    	return h1 + h0

class DownBlock(torch.nn.Module):

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):

        super(DownBlock, self).__init__()

        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)



    def forward(self, x):

    	l0 = self.down_conv1(x)

    	h0 = self.down_conv2(l0)

    	l1 = self.down_conv3(h0 - x)

    	return l1 + l0

class DownBlockPix(torch.nn.Module):

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, scale=4,bias=True, activation='prelu', norm=None):

        super(DownBlockPix, self).__init__()

        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

        self.down_conv2 = Upsampler(scale,num_filter)

        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)



    def forward(self, x):

    	l0 = self.down_conv1(x)

    	h0 = self.down_conv2(l0)

    	l1 = self.down_conv3(h0 - x)

    	return l1 + l0

class D_DownBlock(torch.nn.Module):

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):

        super(D_DownBlock, self).__init__()

        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None)

        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)



    def forward(self, x):

    	x = self.conv(x)

    	l0 = self.down_conv1(x)

    	h0 = self.down_conv2(l0)

    	l1 = self.down_conv3(h0 - x)

    	return l1 + l0

class D_DownBlockPix(torch.nn.Module):

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, scale=4, bias=True, activation='prelu', norm=None):

        super(D_DownBlockPix, self).__init__()

        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None)

        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

        self.down_conv2 = Upsampler(scale,num_filter)

        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)



    def forward(self, x):

    	x = self.conv(x)

    	l0 = self.down_conv1(x)

    	h0 = self.down_conv2(l0)

    	l1 = self.down_conv3(h0 - x)

    	return l1 + l0

class Net(nn.Module):
    def __init__(self, num_channels=3, base_filter=8, feat=8, num_stages=2, scale_factor=4):
        super(Net, self).__init__()
        if scale_factor == 2:
        	kernel = 6
        	stride = 2
        	padding = 2
        elif scale_factor == 4:
        	kernel = 8
        	stride = 4
        	padding = 2
        elif scale_factor == 8:
        	kernel = 12
        	stride = 8
        	padding = 2       
        self.num_stages = num_stages

        #Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None)
        #Back-projection stages
        self.up1 = UpBlock(base_filter, kernel, stride, padding)
        self.down1 = DownBlock(base_filter, kernel, stride, padding)
        self.up2 = UpBlock(base_filter, kernel, stride, padding)
        self.down2 = D_DownBlock(base_filter, kernel, stride, padding, 2)
        self.up3 = D_UpBlock(base_filter, kernel, stride, padding, 2)
        self.down3 = D_DownBlock(base_filter, kernel, stride, padding, 3)
        self.up4 = D_UpBlock(base_filter, kernel, stride, padding, 3)
        self.down4 = D_DownBlock(base_filter, kernel, stride, padding, 4)
        self.up5 = D_UpBlock(base_filter, kernel, stride, padding, 4)
        self.down5 = D_DownBlock(base_filter, kernel, stride, padding, 5)
        self.up6 = D_UpBlock(base_filter, kernel, stride, padding, 5)
        self.down6 = D_DownBlock(base_filter, kernel, stride, padding, 6)
        self.up7 = D_UpBlock(base_filter, kernel, stride, padding, 6)
        #Reconstruction
        self.output_conv = ConvBlock(num_stages*base_filter, num_channels, 3, 1, 1, activation=None, norm=None)       
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
           
    def forward(self, x):
        x = self.feat0(x)
        l = self.feat1(x)       
        results = []
        for i in range(self.num_stages):
            h1 = self.up1(l)
            l1 = self.down1(h1)
            h2 = self.up2(l1)           
            concat_h = torch.cat((h2, h1),1)
            l = self.down2(concat_h)        
            concat_l = torch.cat((l, l1),1)
            h = self.up3(concat_l)           
            concat_h = torch.cat((h, concat_h),1)
            l = self.down3(concat_h)          
            concat_l = torch.cat((l, concat_l),1)
            h = self.up4(concat_l)            
            concat_h = torch.cat((h, concat_h),1)
            l = self.down4(concat_h)
            concat_l = torch.cat((l, concat_l),1)
            h = self.up5(concat_l)
            concat_h = torch.cat((h, concat_h),1)
            l = self.down5(concat_h)
            concat_l = torch.cat((l, concat_l),1)
            h = self.up6(concat_l)
            concat_h = torch.cat((h, concat_h),1)
            l = self.down6(concat_h)
            concat_l = torch.cat((l, concat_l),1)
            h = self.up7(concat_l)
            results.append(h)
        results = torch.cat(results,1)
        x = self.output_conv(results)
        return x



