import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import os
from matplotlib import pyplot as plt
import torch.fft
import numpy as np
from model.MAC_Kernel import GenerateKernels, GenerateKernels3, GenerateKernels4

os.environ['CUDA_VISIBLE_DEVICES']="0"

kernels = GenerateKernels()
weights = [
            nn.Parameter(data = torch.FloatTensor(k).unsqueeze(0).unsqueeze(0), requires_grad=False).cuda()
            for ks in kernels for k in ks
        ]
kernels2 = GenerateKernels3()
weights2 = [
            nn.Parameter(data = torch.FloatTensor(k).unsqueeze(0).unsqueeze(0), requires_grad=False).cuda()
            for ks in kernels2 for k in ks
        ]
kernels3 = GenerateKernels4()
weights3 = [
            nn.Parameter(data = torch.FloatTensor(k).unsqueeze(0).unsqueeze(0), requires_grad=False).cuda()
            for ks in kernels3 for k in ks
        ]

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, max(1, in_planes // 16), 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(max(1, in_planes // 16), in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        # out = out + residual
        return out

class MAC(nn.Module):
    def __init__(self, inplanes, outplanes, one, two, three, scales = 4):
        super(MAC, self).__init__()
        if outplanes % scales != 0: 
            raise ValueError('Planes must be divisible by scales')
        self.weights = weights[:]
        self.weights2 = weights2
        self.weights3 = weights3
        self.scales = scales
        self.relu = nn.ReLU(inplace = True)
        self.spx = outplanes // scales
        self.inconv = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, 1, 0),
            nn.BatchNorm2d(outplanes)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.spx, self.spx, one, 1, one // 2, groups = self.spx),
            nn.BatchNorm2d(self.spx),
        )
        self.conv1[0].weight.data = self.weights[one // 2 - 1].repeat(self.spx, 1, 1, 1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.spx, self.spx, two, 1, 2, groups = self.spx, dilation=2),
            nn.BatchNorm2d(self.spx),
        )
        self.conv2[0].weight.data = self.weights[two // 2 - 1].repeat(self.spx, 1, 1, 1)

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.spx, self.spx, three, 1, 1, groups = self.spx),
        )
        self.conv3[0].weight.data = self.weights2[0].repeat(self.spx, 1, 1, 1)

        self.conv4 = nn.Sequential(
            nn.Conv2d(self.spx, self.spx, three, 1, 2, groups = self.spx, dilation=2),
        )
        self.conv4[0].weight.data = self.weights3[0].repeat(self.spx, 1, 1, 1)
        
        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(self.spx)
        )
        self.outconv = nn.Sequential(
            nn.Conv2d(outplanes, outplanes, 3, 1, 1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )
        self.ca = ChannelAttention(outplanes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.inconv(x)
        inputt = x
        xs = torch.chunk(x, self.scales, 1)
        ys = []
        ys.append(xs[0])
        ys.append(self.relu(self.conv1(xs[1])))
        ys.append(self.relu(self.conv2(xs[2] + ys[1])))
        temp = xs[3] + ys[2]
        temp1 = self.conv5(self.conv3(temp) + self.conv4(temp))
        ys.append(self.relu(temp1))
        y = torch.cat(ys, 1)

        y = self.outconv(y)

        output = self.relu(y + inputt)
        return output

class DHPF(nn.Module):
    def __init__(self, energy):
        super(DHPF, self).__init__()
        self.energy = energy
    
    def _determine_cutoff_frequency(self, f_transform, target_ratio):
        total_energy = self._calculate_total_energy(f_transform)
        target_low_freq_energy = total_energy * target_ratio

        for cutoff_frequency in range(1, min(f_transform.shape[0], f_transform.shape[1]) // 2):
            low_freq_energy = self._calculate_low_freq_energy(f_transform, cutoff_frequency)
            if low_freq_energy >= target_low_freq_energy:
                return cutoff_frequency
        return 5 
    
    def _calculate_total_energy(self, f_transform):
        magnitude_spectrum = torch.abs(f_transform)
        total_energy = torch.sum(magnitude_spectrum ** 2)
        return total_energy
    
    def _calculate_low_freq_energy(self, f_transform, cutoff_frequency):
        magnitude_spectrum = torch.abs(f_transform)
        height, width = magnitude_spectrum.shape

        low_freq_energy = torch.sum(magnitude_spectrum[
            height // 2 - cutoff_frequency:height // 2 + cutoff_frequency,
            width // 2 - cutoff_frequency:width // 2 + cutoff_frequency
        ] ** 2)
    
        return low_freq_energy

    def forward(self, x):
        B, C, H, W = x.shape
        f = torch.fft.fft2(x)
        fshift = torch.fft.fftshift(f)
        crow, ccol = H // 2, W // 2
        for i in range(B):
            cutoff_frequency = self._determine_cutoff_frequency(fshift[i, 0], self.energy) 
            fshift[i, :, crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 0
        ishift = torch.fft.ifftshift(fshift)
        ideal_high_pass = torch.abs(torch.fft.ifft2(ishift))
        return ideal_high_pass 

class HDNet(nn.Module):
    def __init__(self, input_channels, block=ResNet):
        super(HDNet, self).__init__()
        param_channels = [16, 32, 64, 128, 256]
        param_blocks = [2, 2, 2, 2]
        energy = [0.1, 0.2, 0.4, 0.8]

        self.pool = nn.MaxPool2d(2, 2)
        self.sigmoid = nn.Sigmoid()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.conv_init = nn.Conv2d(input_channels, param_channels[0], 1, 1)
        self.py_init = self._make_layer2(input_channels, 1, block)

        self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], block)
        self.encoder_1 = self._make_layer(param_channels[0], param_channels[1], block, param_blocks[0])
        self.encoder_2 = self._make_layer(param_channels[1], param_channels[2], block, param_blocks[1])
        self.encoder_3 = self._make_layer(param_channels[2], param_channels[3], block, param_blocks[2])
     
        self.middle_layer = self._make_layer(param_channels[3], param_channels[4], block, param_blocks[3])
        
        self.decoder_3 = self._make_layer2(param_channels[3]+param_channels[4], param_channels[3], block, param_blocks[2])
        self.decoder_2 = self._make_layer2(param_channels[2]+param_channels[3], param_channels[2], block, param_blocks[1])
        self.decoder_1 = self._make_layer2(param_channels[1]+param_channels[2], param_channels[1], block, param_blocks[0])
        self.decoder_0 = self._make_layer2(param_channels[0]+param_channels[1], param_channels[0], block)

        self.py3 = DHPF(energy[3])
        self.py2 = DHPF(energy[2])
        self.py1 = DHPF(energy[1])
        self.py0 = DHPF(energy[0])

        self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
        self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
        self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
        self.output_3 = nn.Conv2d(param_channels[3], 1, 1)

        self.final = nn.Conv2d(4, 1, 3, 1, 1)


    def _make_layer(self, in_channels, out_channels, block, block_num=1):
        layer = []        
        layer.append(MAC(in_channels, out_channels, 3, 3, 3))
        for _ in range(block_num-1):
            layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)
    
    def _make_layer2(self, in_channels, out_channels, block, block_num = 1):
        layer= []
        layer.append(block(in_channels, out_channels))
        for _ in range(block_num-1):
            layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)

    def forward(self, x, warm_flag):
        
        x_e0 = self.encoder_0(self.conv_init(x)) #
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))

        x_m = self.middle_layer(self.pool(x_e3))
        
        x_d3 = self.decoder_3(torch.cat([x_e3, self.up(x_m)], 1))
        x_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_d3)], 1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self.up(x_d2)], 1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self.up(x_d1)], 1))
        
        mask0 = self.output_0(x_d0)
        mask1 = self.output_1(x_d1)
        mask2 = self.output_2(x_d2)
        mask3 = self.output_3(x_d3)
        
        if warm_flag:
            x_py_init = self.py_init(x)
            x_py_v3 = x_py_init * self.sigmoid(self.up_8(mask3)) + x_py_init 
            x_py_v3 = self.py3(x_py_v3)

            x_py_v2 = x_py_v3 * self.sigmoid(self.up_4(mask2)) + x_py_v3 
            x_py_v2 = self.py2(x_py_v2)

            x_py_v1 = x_py_v2 * self.sigmoid(self.up(mask1)) + x_py_v2 
            x_py_v1 = self.py1(x_py_v1)

            x_py_v0 = x_py_v1 * self.sigmoid(mask0) + x_py_v1 
            x_py_v0 = self.sigmoid(self.py0(x_py_v0))

            output = self.final(torch.cat([mask0, self.up(mask1), self.up_4(mask2), self.up_8(mask3)], dim=1))
            output = output * x_py_v0 + output
            return [mask0, mask1, mask2, mask3], output
    
        else:
            output = self.output_0(x_d0)
            output = output
            return [], output