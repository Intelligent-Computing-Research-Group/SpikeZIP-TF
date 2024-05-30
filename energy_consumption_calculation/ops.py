'''
Copyright (C) 2022 Guangyao Chen - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
try:
    from spikingjelly.clock_driven.neuron import MultiStepIFNode, MultiStepLIFNode, IFNode, LIFNode, MultiStepParametricLIFNode, ParametricLIFNode
except:
    from spikingjelly.activation_based.neuron import MultiStepIFNode, MultiStepLIFNode, IFNode, LIFNode, MultiStepParametricLIFNode, ParametricLIFNode
import sys
sys.path.insert(0,"/home/kang_you/SpikeZIP_transformer/")
from spike_quan_layer import IFNeuron,QAttention,QuanConv2d,QuanLinear,MyQuan
from timm.models.vision_transformer import Attention

def spike_rate(inp):
    # 改变判断前层输入是否为脉冲的方法，假定unique元素个数小于“某个整数”（一个shortcut会使得脉冲矩阵的最大值+1）
    # Nspks_max = 30  # for spikformer-4-384, real Nspks_max is 9 (2*4+1=9)；for spikformer-8-512, real Nspks_max is 17 (2*8+1=17)；
    # Nspks_max = 1  # similar to the original version by ChenGY 只有真正全0-1矩阵才作为event-driven，计算AC，否则均计算为MAC
    Nspks_max = 2  # SpikeZIP: 输出是-threshold,0,threshold三值的矩阵，因此这个值需要修改成 2

    if torch.abs(inp).max() == 0:
        spkhistc = None
        spike = True
        spike_rate = 0
        return spike, spike_rate, spkhistc
    inp = inp / torch.abs(inp).max() # SpikeZIP: 将feature除以threshold转化为-1,0,1的三值矩阵，threshold通过inp.max()求得
    num = inp.unique()
    # print(len(num))
    # if len(num) <= Nspks_max+1 and inp.max() <= Nspks_max and inp.min() >= 0: 
    if len(num) <= Nspks_max+1 and inp.max() <= Nspks_max - 1 and inp.min() >= -(Nspks_max - 1): #SpikeZIP: 这里由于有负值，因为最大值应该是>=-1
        # 将[0,1,2...16]这种累积脉冲的矩阵，转化为只含有[0,1]的矩阵，把“整数*浮点数”作为一个AC
        # inp = torch.where(inp<1.0, 0.0, 1.0)
        
        # calculate module.__spkhistc__ （较为耗时，且对多个batch的数据进行统计较麻烦（类似于发放率，逐个batch累加，最后求平均），因此此处仅用一个batch的测试获取此信息）
        # https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
        # spkhistc, _ = np.histogram(np.array(inp.cpu()), bins=np.arange(21))
        spkhistc = None
        
        spike = True
        spike_rate = (torch.abs(inp).sum() / inp.numel()).item()  # 此种计算方法已计入了时长T的影响，因为inp包含T这一维度。（注意：分母也包含了T这一维度）
        # neg_rate = ((inp<0).sum() / inp.numel()).item()
        # pos_rate = ((inp>0).sum() / inp.numel()).item()
        # print("spike_rate",spike_rate,"neg_rate",neg_rate,"pos_rate",pos_rate)
        # print(len(num), spike_rate, num)
        # print(len(num), spike_rate, num, inp.unique())
        # print(len(num), spike_rate, num, spkhistc)
    else: 
        spkhistc = None
        
        spike = False
        spike_rate = 1
        # print(len(num), spike_rate, num.max())
        # print(len(num), spike_rate, num.max(), spkhistc)
    
    return spike, spike_rate, spkhistc

    # # original version by ChenGY
    # # T = inp.shape[1]
    # num = inp.unique()
    # if len(num) <= 2 and inp.max() <= 1 and inp.min() >= 0: 
    #     spike = True
    #     spike_rate = (inp.sum() / inp.numel()).item()
    # else: 
    #     spike = False
    #     spike_rate = 1

    # return spike, spike_rate


def empty_syops_counter_hook(module, input, output):
    module.__syops__ += np.array([0.0, 0.0, 0.0, 0.0])


def upsample_syops_counter_hook(module, input, output):
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__syops__[0] += int(output_elements_count)

    # spike, rate = spike_rate(output[0])
    spike, rate, _ = spike_rate(output) 

    if spike:
        module.__syops__[1] += int(output_elements_count) * rate
    else:
        module.__syops__[2] += int(output_elements_count)

    module.__syops__[3] += rate * 100

def relu_syops_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__syops__[0] += int(active_elements_count)

    # spike, rate = spike_rate(output[0])
    spike, rate, _ = spike_rate(output) 

    if spike:
        module.__syops__[1] += int(active_elements_count) * rate
    else:
        module.__syops__[2] += int(active_elements_count)

    module.__syops__[3] += rate * 100

def IF_syops_counter_hook(module, input, output):
    active_elements_count = input[0].numel()
    module.__syops__[0] += int(active_elements_count)

    # spike, rate = spike_rate(output[0])
    spike, rate, spkhistc = spike_rate(output)
    module.__syops__[1] += int(active_elements_count)
    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc

def LIF_syops_counter_hook(module, input, output):  # output is <class 'torch.Tensor'>
    active_elements_count = input[0].numel()  # input is tuple, input[0].shape = torch.Size([4, 1, 48, 32, 32]) [T, B, C, H, W]
    module.__syops__[0] += int(active_elements_count)  # 输入的元素个数作为操作数，不管是否有脉冲？似乎有问题？？ input[0].numel() = np.prod(input[0].shape)

    # spike, rate = spike_rate(output[0])  # output.shape = torch.Size([4, 1, 48, 32, 32]) [T, B, C, H, W] # 神经元层计算的是本身的发放率。但只用了第一个时间步的脉冲输入来计算发放率，代表性不足？？ 考虑使用spike, rate = spike_rate(output)。猜测作者是误认为output与input一样是tuple
    spike, rate, spkhistc = spike_rate(output)  
    module.__syops__[1] += int(active_elements_count)
    # module.__syops__[2] += int(active_elements_count)
    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc

def cal_linear_sparsity(input, weight_mask):
    input_mask = (input != 0).float()
    output_mask = F.linear(input_mask,weight_mask,bias=None)
    input_full = torch.ones(size=input.shape).to(input.device)
    weight_full = torch.ones(size=weight_mask.shape).to(weight_mask.device)
    output_full = F.linear(input_full,weight_full,bias=None)

    return torch.sum(output_mask)/torch.sum(output_full)

def linear_syops_counter_hook(module, input, output):
    input = input[0]  # input is tuple, input[0].shape = torch.Size([4, 64, 384]) [TB, N, C]  # output.shape = torch.Size([4, 64, 384])
    
    spike, rate, spkhistc = spike_rate(input)  # 计算了前一层的发放率  # input.unique --> [0,1,2]  spike=False 不把该层作为spike-triggered
    print("spike, rate",spike, rate)
    # if spike == True:
    #     if hasattr(module,"weight_mask"):
    #         real_rate = cal_linear_sparsity(input,module.weight_mask)
    #     else:
    #         real_rate = cal_linear_sparsity(input,(module.weight != 0).float())
    # else:
    #     real_rate = 1.0
    # pytorch checks dimensions, so here we don't care much
    batch_size = input.shape[0]
    output_last_dim = output.shape[-1]
    # bias_syops = output_last_dim if module.bias is not None else 0
    bias_syops = output_last_dim*batch_size if module.bias is not None else 0 # need to take batch_size into account, as in conv_syops_counter_hook
    module.__syops__[0] += int(np.prod(input.shape) * output_last_dim + bias_syops)
    if spike:
        module.__syops__[1] += int(np.prod(input.shape) * output_last_dim + bias_syops) * rate
    else:
        module.__syops__[2] += int(np.prod(input.shape) * output_last_dim + bias_syops)

    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc


def pool_syops_counter_hook(module, input, output):
    input = input[0]  # input is tuple, input[0].shape = torch.Size([4, 192, 32, 32]) [TB, C, H, W]  # output.shape = torch.Size([4, 192, 16, 16])
    spike, rate, spkhistc = spike_rate(input)
    module.__syops__[0] += int(np.prod(input.shape)) # 直接加上元素个数，对吗？？

    if spike:
        module.__syops__[1] += int(np.prod(input.shape)) * rate
    else:
        module.__syops__[2] += int(np.prod(input.shape))

    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc

def bn_syops_counter_hook(module, input, output):
    input = input[0]  # input is tuple, input[0].shape = torch.Size([4, 48, 32, 32]) [TB, C, H, W]
    spike, rate, spkhistc = spike_rate(input)
    batch_syops = np.prod(input.shape)
    if module.affine:
        batch_syops *= 2
    module.__syops__[0] += int(batch_syops)

    if spike:
        module.__syops__[1] += int(batch_syops) * rate
    else:
        module.__syops__[2] += int(batch_syops)
    
    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc

def ln_syops_counter_hook(module, input, output):
    input = input[0]  # input is tuple, input[0].shape = torch.Size([4, 48, 32, 32]) [TB, C, H, W]
    spike, rate, spkhistc = spike_rate(input)
    # batch_syops = np.prod(input.shape)
    batch_syops = np.prod(input.shape)
    if module.elementwise_affine:
        batch_syops *= 2
    module.__syops__[0] += int(batch_syops)

    if spike:
        module.__syops__[1] += int(batch_syops) * rate
    else:
        module.__syops__[2] += int(batch_syops)
    
    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc


def cal_conv_sparsity(input, weight_mask, stride,padding,groups):
    input_mask = (input != 0).float()
    output_mask = F.conv2d(input_mask,weight_mask,bias=None,stride=stride,padding=padding,groups=groups)
    input_full = torch.ones(size=input.shape).to(input.device)
    weight_full = torch.ones(size=weight_mask.shape).to(weight_mask.device)
    output_full = F.conv2d(input_full,weight_full,bias=None,stride=stride,padding=padding,groups=groups)
    return torch.sum(output_mask)/torch.sum(output_full)

def conv_syops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]  # input is tuple, input[0].shape = torch.Size([4, 3, 32, 32]) [TB, C, H, W]
    spike, rate, spkhistc = spike_rate(input)

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])  # output.shape = torch.Size([4, 48, 32, 32])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups
    padding = conv_module.padding[0]
    stride = conv_module.stride[0]
    # if spike==True:
    #     if hasattr(conv_module,"weight_mask"):
    #         real_rate = cal_conv_sparsity(input, conv_module.weight_mask, stride,padding,groups)
    #     else:
    #         real_rate = cal_conv_sparsity(input, (conv_module.weight != 0).float(), stride,padding,groups)
    # else:
    #     real_rate = 1.0

    filters_per_channel = out_channels // groups
    conv_per_position_syops = int(np.prod(kernel_dims)) * \
        in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_syops = conv_per_position_syops * active_elements_count

    bias_syops = 0

    if conv_module.bias is not None:

        bias_syops = out_channels * active_elements_count

    # overall_syops = overall_conv_syops + bias_syops
    overall_syops = overall_conv_syops + bias_syops

    conv_module.__syops__[0] += int(overall_syops)

    if spike:
        conv_module.__syops__[1] += int(overall_syops) * rate
    else:
        conv_module.__syops__[2] += int(overall_syops)

    conv_module.__syops__[3] += rate * 100
    conv_module.__spkhistc__ = spkhistc


def rnn_syops(syops, rnn_module, w_ih, w_hh, input_size):
    # matrix matrix mult ih state and internal state
    syops += w_ih.shape[0]*w_ih.shape[1]
    # matrix matrix mult hh state and internal state
    syops += w_hh.shape[0]*w_hh.shape[1]
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        # add both operations
        syops += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        # hadamard of r
        syops += rnn_module.hidden_size
        # adding operations from both states
        syops += rnn_module.hidden_size*3
        # last two hadamard product and add
        syops += rnn_module.hidden_size*3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        # adding operations from both states
        syops += rnn_module.hidden_size*4
        # two hadamard product and add for C state
        syops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        # final hadamard
        syops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
    return syops


def rnn_syops_counter_hook(rnn_module, input, output):
    """
    Takes into account batch goes at first position, contrary
    to pytorch common rule (but actually it doesn't matter).
    If sigmoid and tanh are hard, only a comparison syops should be accurate
    """
    syops = 0
    # input is a tuple containing a sequence to process and (optionally) hidden state
    inp = input[0]
    batch_size = inp.shape[0]
    seq_length = inp.shape[1]
    num_layers = rnn_module.num_layers

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__('weight_ih_l' + str(i))
        w_hh = rnn_module.__getattr__('weight_hh_l' + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        syops = rnn_syops(syops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__('bias_ih_l' + str(i))
            b_hh = rnn_module.__getattr__('bias_hh_l' + str(i))
            syops += b_ih.shape[0] + b_hh.shape[0]

    syops *= batch_size
    syops *= seq_length
    if rnn_module.bidirectional:
        syops *= 2
    rnn_module.__syops__[0] += int(syops)

def rnn_cell_syops_counter_hook(rnn_cell_module, input, output):
    syops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__('weight_ih')
    w_hh = rnn_cell_module.__getattr__('weight_hh')
    input_size = inp.shape[1]
    syops = rnn_syops(syops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__('bias_ih')
        b_hh = rnn_cell_module.__getattr__('bias_hh')
        syops += b_ih.shape[0] + b_hh.shape[0]

    syops *= batch_size
    rnn_cell_module.__syops__[0] += int(syops)


def multihead_attention_counter_hook(multihead_attention_module, input, output):
    syops = 0

    # print(input[0].shape)
    q = k = v = input[0]

    batch_first = True

    if batch_first:
        batch_size = q.shape[0]
        len_idx = 1
    else:
        batch_size = q.shape[1]
        len_idx = 0

    dim_idx = 2

    qdim = q.shape[dim_idx]
    kdim = k.shape[dim_idx]
    vdim = v.shape[dim_idx]

    qlen = q.shape[len_idx]
    klen = k.shape[len_idx]
    vlen = v.shape[len_idx]

    num_heads = multihead_attention_module.num_heads
    assert qdim == multihead_attention_module.num_heads * multihead_attention_module.head_dim

    if hasattr(multihead_attention_module,"kdim") is None:
        assert kdim == qdim
    if hasattr(multihead_attention_module,"vdim") is None:
        assert vdim == qdim

    syops = 0

    # Q scaling
    syops += qlen * qdim

    # Initial projections
    syops += (
        (qlen * qdim * qdim)  # QW
        + (klen * kdim * kdim)  # KW
        + (vlen * vdim * vdim)  # VW
    )

    if hasattr(multihead_attention_module,"in_proj_bias") is not None:
        syops += (qlen + klen + vlen) * qdim

    # attention heads: scale, matmul, softmax, matmul
    qk_head_dim = qdim // num_heads
    v_head_dim = vdim // num_heads

    head_syops = (
        (qlen * klen * qk_head_dim)  # QK^T
        + (qlen * klen)  # softmax
        + (qlen * klen * v_head_dim)  # AV
    )

    syops += num_heads * head_syops

    # final projection, bias is always enabled
    syops += qlen * vdim * (vdim + 1)

    syops *= batch_size
    multihead_attention_module.__syops__[0] += int(syops)
    multihead_attention_module.__syops__[2] += int(syops)


CUSTOM_MODULES_MAPPING = {}

MODULES_MAPPING = {
    # convolutions
    nn.Conv1d: conv_syops_counter_hook,
    nn.Conv2d: conv_syops_counter_hook,
    QuanConv2d: conv_syops_counter_hook,
    nn.Conv3d: conv_syops_counter_hook,
    
    # activations
    nn.ReLU: relu_syops_counter_hook,
    MyQuan: relu_syops_counter_hook,
    nn.PReLU: relu_syops_counter_hook,
    nn.ELU: relu_syops_counter_hook,
    nn.LeakyReLU: relu_syops_counter_hook,
    nn.ReLU6: relu_syops_counter_hook,
    # poolings
    nn.MaxPool1d: pool_syops_counter_hook,
    nn.AvgPool1d: pool_syops_counter_hook,
    nn.AvgPool2d: pool_syops_counter_hook,
    nn.MaxPool2d: pool_syops_counter_hook,
    nn.MaxPool3d: pool_syops_counter_hook,
    nn.AvgPool3d: pool_syops_counter_hook,
    nn.AdaptiveMaxPool1d: pool_syops_counter_hook,
    nn.AdaptiveAvgPool1d: pool_syops_counter_hook,
    nn.AdaptiveMaxPool2d: pool_syops_counter_hook,
    nn.AdaptiveAvgPool2d: pool_syops_counter_hook,
    nn.AdaptiveMaxPool3d: pool_syops_counter_hook,
    nn.AdaptiveAvgPool3d: pool_syops_counter_hook,
    # BNs
    nn.BatchNorm1d: bn_syops_counter_hook,
    nn.BatchNorm2d: bn_syops_counter_hook,
    nn.BatchNorm3d: bn_syops_counter_hook,
    nn.LayerNorm: ln_syops_counter_hook,

    # Neuron IF
    MultiStepIFNode: IF_syops_counter_hook,
    IFNode: IF_syops_counter_hook,
    IFNeuron: IF_syops_counter_hook, # SpikeZIP: our neuron
    # Neuron LIF
    MultiStepLIFNode: LIF_syops_counter_hook,
    LIFNode: LIF_syops_counter_hook,
    # Neuron PLIF
    MultiStepParametricLIFNode: LIF_syops_counter_hook,
    ParametricLIFNode: LIF_syops_counter_hook,

    nn.InstanceNorm1d: bn_syops_counter_hook,
    nn.InstanceNorm2d: bn_syops_counter_hook,
    nn.InstanceNorm3d: bn_syops_counter_hook,
    nn.GroupNorm: bn_syops_counter_hook,
    # FC
    nn.Linear: linear_syops_counter_hook,
    QuanLinear: linear_syops_counter_hook,
    # Upscale
    nn.Upsample: upsample_syops_counter_hook,
    # Deconvolution
    nn.ConvTranspose1d: conv_syops_counter_hook,
    nn.ConvTranspose2d: conv_syops_counter_hook,
    nn.ConvTranspose3d: conv_syops_counter_hook,
    # RNN
    nn.RNN: rnn_syops_counter_hook,
    nn.GRU: rnn_syops_counter_hook,
    nn.LSTM: rnn_syops_counter_hook,
    nn.RNNCell: rnn_cell_syops_counter_hook,
    nn.LSTMCell: rnn_cell_syops_counter_hook,
    nn.GRUCell: rnn_cell_syops_counter_hook,
    nn.MultiheadAttention: multihead_attention_counter_hook,
    Attention: multihead_attention_counter_hook,
    QAttention: multihead_attention_counter_hook
}

if hasattr(nn, 'GELU'):
    MODULES_MAPPING[nn.GELU] = relu_syops_counter_hook
