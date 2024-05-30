import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
import math
from copy import deepcopy
import numpy as np
import scipy

# torch.set_default_dtype(torch.double)
# torch.set_default_tensor_type(torch.DoubleTensor)

class ORIIFNeuron(nn.Module):
    def __init__(self,q_threshold,level,sym=False):
        super(ORIIFNeuron,self).__init__()
        self.q = 0.0
        self.acc_q = 0.0
        self.q_threshold = q_threshold
        self.is_work = False
        self.cur_output = 0.0
        # self.steps = torch.tensor(3.0) 
        self.level = torch.tensor(level)
        self.sym = sym
        self.pos_max = torch.tensor(level - 1)
        self.neg_min = torch.tensor(0)
            
        self.eps = 0

    # def __repr__(self):
    #         return f"IFNeuron(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, neg_min={self.neg_min}, q_threshold={self.q_threshold})"
    
    def reset(self):
        # print("IFNeuron reset")
        self.q = 0.0
        self.cur_output = 0.0
        self.acc_q = 0.0
        self.is_work = False
        self.spike_position = None
        # self.neg_spike_position = None

    def forward(self,input):
        x = input/self.q_threshold
        if (not torch.is_tensor(x)) and x == 0.0 and (not torch.is_tensor(self.cur_output)) and self.cur_output == 0.0:
            self.is_work = False
            return x
        
        if not torch.is_tensor(self.cur_output):
            self.cur_output = torch.zeros(x.shape,dtype=x.dtype).to(x.device)
            self.acc_q = torch.zeros(x.shape,dtype=torch.float32).to(x.device)
            self.q = torch.zeros(x.shape,dtype=torch.float32).to(x.device) + 0.5

        self.is_work = True
        
        self.q = self.q + (x.detach() if torch.is_tensor(x) else x)
        self.acc_q = torch.round(self.acc_q)

        spike_position = (self.q - 1 >= 0)
        # neg_spike_position = (self.q < -self.eps) & (self.acc_q > self.neg_min)

        self.cur_output[:] = 0
        self.cur_output[spike_position] = 1
        # self.cur_output[neg_spike_position] = -1

        self.acc_q = self.acc_q + self.cur_output
        self.q[spike_position] = self.q[spike_position] - 1
        # self.q[neg_spike_position] = self.q[neg_spike_position] + 1

        # print((x == 0).all(), (self.cur_output==0).all())
        if (x == 0).all() and (self.cur_output==0).all():
            self.is_work = False
        
        # print("self.cur_output",self.cur_output)
        
        return self.cur_output*self.q_threshold


class IFNeuron(nn.Module):
    def __init__(self,q_threshold,level,sym=False):
        super(IFNeuron,self).__init__()
        self.q = 0.0
        self.acc_q = 0.0
        self.q_threshold = q_threshold
        self.is_work = False
        self.cur_output = 0.0
        # self.steps = torch.tensor(3.0) 
        self.level = torch.tensor(level)
        self.sym = sym
        if sym:
            self.pos_max = torch.tensor(level//2 - 1)
            self.neg_min = torch.tensor(-level//2)
        else:
            self.pos_max = torch.tensor(level - 1)
            self.neg_min = torch.tensor(0)
            
        self.eps = 0

    # def __repr__(self):
    #         return f"IFNeuron(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, neg_min={self.neg_min}, q_threshold={self.q_threshold})"
    
    def reset(self):
        # print("IFNeuron reset")
        self.q = 0.0
        self.cur_output = 0.0
        self.acc_q = 0.0
        self.is_work = False
        self.spike_position = None
        self.neg_spike_position = None

    def forward(self,input):
        x = input/self.q_threshold
        if (not torch.is_tensor(x)) and x == 0.0 and (not torch.is_tensor(self.cur_output)) and self.cur_output == 0.0:
            self.is_work = False
            return x*self.q_threshold
        
        if not torch.is_tensor(self.cur_output):
            self.cur_output = torch.zeros(x.shape,dtype=x.dtype).to(x.device)
            self.acc_q = torch.zeros(x.shape,dtype=torch.float32).to(x.device)
            self.q = torch.zeros(x.shape,dtype=torch.float32).to(x.device) + 0.5

        self.is_work = True
        
        self.q = self.q + (x.detach() if torch.is_tensor(x) else x)
        self.acc_q = torch.round(self.acc_q)

        spike_position = (self.q - 1 >= 0) & (self.acc_q < self.pos_max)
        neg_spike_position = (self.q < -self.eps) & (self.acc_q > self.neg_min)

        self.cur_output[:] = 0
        self.cur_output[spike_position] = 1
        self.cur_output[neg_spike_position] = -1

        self.acc_q = self.acc_q + self.cur_output
        self.q[spike_position] = self.q[spike_position] - 1
        self.q[neg_spike_position] = self.q[neg_spike_position] + 1

        # print((x == 0).all(), (self.cur_output==0).all())
        if (x == 0).all() and (self.cur_output==0).all():
            self.is_work = False
        
        # print("self.cur_output",self.cur_output)
        
        return self.cur_output*self.q_threshold


class Spiking_LayerNorm(nn.Module):
    def __init__(self,dim):
        super(Spiking_LayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.X = 0.0
        self.Y_pre = None

    def reset(self):
        # print("Spiking_LayerNorm reset")
        self.X = 0.0
        self.Y_pre = None
        
    def forward(self,input):
        self.X = self.X + input
        Y = self.layernorm(self.X)
        if self.Y_pre is not None:
            Y_pre = self.Y_pre.detach().clone()
        else:
            Y_pre = 0.0
        self.Y_pre = Y
        return Y - Y_pre

class spiking_softmax(nn.Module):
    def __init__(self):
        super(spiking_softmax, self).__init__()
        self.X = 0.0
        self.Y_pre = 0.0
    
    def reset(self):
        # print("spiking_softmax reset")
        self.X = 0.0
        self.Y_pre = 0.0        
    
    def forward(self, input):
        self.X = input + self.X
        Y = F.softmax(self.X,dim=-1)
        Y_pre = deepcopy(self.Y_pre)
        self.Y_pre = Y
        return Y - Y_pre

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def floor_pass(x):
    y = x.floor()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def threshold_optimization(data, quantization_level=255, n_trial=300, eps=1e-10):
    '''
    This function collect the activation data and find the optimized clipping
    threshold using KL_div as metric. Since this method is originated from
    post-training quantization which adopted in Tensor-RT, we keep the number of
    bits here.
    Args:
        data(numpy array): activation data
        n_bit(int):
        n_trial(int): the searching steps.
        eps(float): add eps at the average bin step for numberical stability.

    '''

    n_lvl = quantization_level  # quantization levels
    n_half_lvls = (quantization_level)//2
    n_bin_edge = n_lvl * n_trial + 1

    data_max = np.max(np.abs(data))
    hist, bin_edge = np.histogram(data.flatten(),
                                  bins=np.linspace(-data_max,
                                                   data_max,
                                                   num=n_bin_edge))

    mid_idx = int((len(hist)) / 2)
    start_idx = 100
    # log the threshold and corresponding KL-divergence
    kl_result = np.empty([len(range(start_idx, n_trial + 1)), 2])

    for i in range(start_idx, n_trial + 1):
        ref_dist = np.copy(hist[mid_idx - i * n_half_lvls:mid_idx +
                                i * n_half_lvls])
        # merge the outlier
        ref_dist[0] += hist[:mid_idx - i * n_half_lvls].sum()
        ref_dist[-1] += hist[mid_idx + i * n_half_lvls:].sum()
        # perform quantization: bins merge and expansion
        reshape_dim = int(len(ref_dist) / n_lvl)
        ref_dist_reshape = ref_dist.reshape(n_lvl, i)
        # merge bins for quantization
        ref_dist_merged = ref_dist_reshape.sum(axis=1)
        nonzero_mask = (ref_dist_reshape != 0
                        )  # obtain the mask of non-zero bins
        # in each merged large bin, get the average bin_count
        average_bin_count = ref_dist_merged / (nonzero_mask.sum(1) + eps)
        # expand the merged bins back
        expand_bin_count = np.expand_dims(average_bin_count,
                                          axis=1).repeat(i, axis=1)
        candidate_dist = (nonzero_mask * expand_bin_count).flatten()
        kl_div = scipy.stats.entropy(candidate_dist / candidate_dist.sum(),
                                     ref_dist / ref_dist.sum())
        #log threshold and KL-divergence
        current_th = np.abs(
            bin_edge[mid_idx - i * n_half_lvls])  # obtain current threshold
        kl_result[i -
                  start_idx, 0], kl_result[i -
                                           start_idx, 1] = current_th, kl_div

    # based on the logged kl-div result, find the threshold correspond to the smallest kl-div
    th_sel = kl_result[kl_result[:, 1] == kl_result[:, 1].min()][0, 0]
    print(f"Threshold calibration of current layer finished!, calculate threshold {th_sel}")

    return th_sel

# class MyQuan(nn.Module):
#     def __init__(self,level,sym = False,**kwargs):
#         super(MyQuan,self).__init__()
#         # self.level_init = level
#         self.s_init = 0.0
#         self.level = level
#         self.sym = sym
#         if level >= 256:
#             print("level",level)
#             self.pos_max = 'full'
#             self.neg_min = 'full'
#         else:
#             print("level",level)
#             self.pos_max = torch.tensor(level)
#             if sym:
#                 self.pos_max = torch.tensor(float(level//2 - 1))
#                 self.neg_min = torch.tensor(float(-level//2))
#             else:
#                 self.pos_max = torch.tensor(float(level - 1))
#                 self.neg_min = torch.tensor(float(0))
#
#         self.s = nn.Parameter(torch.tensor(1.0))
#         self.batch_init = 20
#         self.init_state = 0
#         self.debug = False
#         self.tfwriter = None
#         self.global_step = 0.0
#         self.name = "myquan"
#
#     def __repr__(self):
#         return f"MyQuan(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, neg_min={self.neg_min}, s={self.s.data})"
#
#
#     def reset(self):
#         self.history_max = torch.tensor(0.0)
#         self.init_state = 0
#         self.is_init = True
#
#     def profiling(self,name,tfwriter,global_step):
#         self.debug = True
#         self.name = name
#         self.tfwriter = tfwriter
#         self.global_step = global_step
#
#     def forward(self, x):
#         # print("self.pos_max",self.pos_max)
#         if self.pos_max == 'full':
#             return x
#         # print("self.Q_thr in Quan",self.Q_thr,"self.T:",self.T)
#         if str(self.neg_min.device) == 'cpu':
#             self.neg_min = self.neg_min.to(x.device)
#         if str(self.pos_max.device) == 'cpu':
#             self.pos_max = self.pos_max.to(x.device)
#         min_val = self.neg_min
#         max_val = self.pos_max
#         # x = F.hardtanh(x, min_val=min_val, max_val=max_val.item())
#
#         # according to LSQ, the grad scale should be proportional to sqrt(1/(quantize_state*neuron_number))
#         # s_grad_scale = 1.0 / ((self.level * x.numel()) ** 0.5)
#         s_grad_scale = 1.0 / ((max_val.detach().abs().mean() * x.numel()) ** 0.5)
#
#         # s_grad_scale = s_grad_scale / ((self.level_init)/(self.pos_max))
#
#         # print("self.init_state",self.init_state)
#         # print("self.init_state<self.batch_init",self.init_state<self.batch_init)
#         # print("self.training",self.training)
#         if self.init_state == 0 and self.training:
#             threshold = threshold_optimization(np.array(x.detach().cpu()), quantization_level=int(self.level), n_trial=300, eps=1e-10)
#             self.s.data = torch.tensor(threshold / (self.level),dtype=torch.float32).cuda()
#             self.init_state += 1
#         # if self.init_state == 0 and self.training:
#         #     self.s.data = torch.tensor(x.detach().abs().mean() * 2 / (self.pos_max.detach().abs().mean() ** 0.5),dtype=torch.float32).cuda()
#         #     self.init_state += 1
#         # elif self.init_state<self.batch_init and self.training:
#         #     self.s.data = 0.9*self.s.data + 0.1*torch.tensor(torch.mean(torch.abs(x.detach()))*2/(math.sqrt(max_val.detach().abs().mean())),dtype=torch.float32)
#         #     self.init_state += 1
#
#         # elif self.init_state==self.batch_init and self.training:
#         #     # self.s = torch.nn.Parameter(self.s)
#         #     self.init_state += 1
#         #     print("initialize finish!!!!")
#
#         s_scale = grad_scale(self.s, s_grad_scale)
#         # s_scale = s_scale * ((self.level_init)/(self.pos_max))
#         output = torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val)*s_scale
#
#         if self.debug and self.tfwriter is not None:
#             self.tfwriter.add_histogram(tag="before_quan/".format(s_scale.item())+self.name+'_data', values=(x).detach().cpu(), global_step=self.global_step)
#             # self.tfwriter.add_histogram(tag="after_clip/".format(s_scale.item())+self.name+'_data', values=(floor_pass(x/s_scale)).detach().cpu(), global_step=self.global_step)
#             self.tfwriter.add_histogram(tag="after_quan/".format(s_scale.item())+self.name+'_data', values=((torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val))).detach().cpu(), global_step=self.global_step)
#             # print("(torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val))",(torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val)))
#             self.debug = False
#             self.tfwriter = None
#             self.name = ""
#             self.global_step = 0.0
#
#         # output = floor_pass(x/s_scale)*s_scale
#         return output

class MyQuan(nn.Module):
    def __init__(self,level,sym = False,**kwargs):
        super(MyQuan,self).__init__()
        # self.level_init = level
        self.s_init = 0.0
        self.level = level
        self.sym = sym
        if level >= 512:
            print("level",level)
            self.pos_max = 'full'
        else:
            print("level",level)
            self.pos_max = torch.tensor(level)
            if sym:
                self.pos_max = torch.tensor(float(level//2 - 1))
                self.neg_min = torch.tensor(float(-level//2))
            else:
                self.pos_max = torch.tensor(float(level - 1))
                self.neg_min = torch.tensor(float(0))

        self.s = nn.Parameter(torch.tensor(1.0))
        self.batch_init = 20
        self.init_state = 0
        self.debug = False
        self.tfwriter = None
        self.global_step = 0.0
        self.name = "myquan"

    def __repr__(self):
        return f"MyQuan(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, neg_min={self.neg_min}, s={self.s.data})"


    def reset(self):
        self.history_max = torch.tensor(0.0)
        self.init_state = 0
        self.is_init = True

    def profiling(self,name,tfwriter,global_step):
        self.debug = True
        self.name = name
        self.tfwriter = tfwriter
        self.global_step = global_step

    def forward(self, x):
        # print("self.pos_max",self.pos_max)
        if self.pos_max == 'full':
            return x
        # print("self.Q_thr in Quan",self.Q_thr,"self.T:",self.T)
        if str(self.neg_min.device) == 'cpu':
            self.neg_min = self.neg_min.to(x.device)
        if str(self.pos_max.device) == 'cpu':
            self.pos_max = self.pos_max.to(x.device)
        min_val = self.neg_min
        max_val = self.pos_max
        # x = F.hardtanh(x, min_val=min_val, max_val=max_val.item())

        # according to LSQ, the grad scale should be proportional to sqrt(1/(quantize_state*neuron_number))
        s_grad_scale = 1.0 / ((max_val.detach().abs().mean() * x.numel()) ** 0.5)
        # s_grad_scale = s_grad_scale / ((self.level_init)/(self.pos_max))

        # print("self.init_state",self.init_state)
        # print("self.init_state<self.batch_init",self.init_state<self.batch_init)
        # print("self.training",self.training)
        if self.init_state == 0 and self.training:
            self.s.data = torch.tensor(x.detach().abs().mean() * 2 / (self.pos_max.detach().abs().mean() ** 0.5),dtype=torch.float32).cuda()
            self.init_state += 1
        # elif self.init_state<self.batch_init and self.training:
        #     self.s.data = 0.9*self.s.data + 0.1*torch.tensor(torch.mean(torch.abs(x.detach()))*2/(math.sqrt(max_val.detach().abs().mean())),dtype=torch.float32)
        #     self.init_state += 1

        # elif self.init_state==self.batch_init and self.training:
        #     # self.s = torch.nn.Parameter(self.s)
        #     self.init_state += 1
        #     # print("initialize finish!!!!")

        s_scale = grad_scale(self.s, s_grad_scale)
        # s_scale = s_scale * ((self.level_init)/(self.pos_max))
        output = torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val)*s_scale

        if self.debug and self.tfwriter is not None:
            self.tfwriter.add_histogram(tag="before_quan/".format(s_scale.item())+self.name+'_data', values=(x).detach().cpu(), global_step=self.global_step)
            # self.tfwriter.add_histogram(tag="after_clip/".format(s_scale.item())+self.name+'_data', values=(floor_pass(x/s_scale)).detach().cpu(), global_step=self.global_step)
            self.tfwriter.add_histogram(tag="after_quan/".format(s_scale.item())+self.name+'_data', values=((torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val))).detach().cpu(), global_step=self.global_step)
            # print("(torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val))",(torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val)))
            self.debug = False
            self.tfwriter = None
            self.name = ""
            self.global_step = 0.0

        # output = floor_pass(x/s_scale)*s_scale
        return output

class QAttention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            level = 2,
            is_softmax = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.level = level
        self.is_softmax = is_softmax

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.quan_q = MyQuan(self.level,sym=True)
        self.quan_k = MyQuan(self.level,sym=True)
        self.quan_v = MyQuan(self.level,sym=True)
        # self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        # self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim,bias=True)
        self.quan_proj = MyQuan(self.level,sym=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_quan = MyQuan(self.level,sym=False)
        self.after_attn_quan = MyQuan(self.level,sym=True)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # q, k = self.q_norm(q), self.k_norm(k)
        q = self.quan_q(q)
        k = self.quan_k(k)
        v = self.quan_v(v)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.is_softmax:
            attn = attn.softmax(dim=-1)
            attn = self.attn_quan(attn)
        else:
            # print("no softmax!!!!")
            attn = self.attn_quan(attn)/N
        
        attn = self.attn_drop(attn)
        x = attn @ v
        x = self.after_attn_quan(x)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.quan_proj(x)

        return x

def multi(x1_t,x2_t,x1_sum_t,x2_sum_t):
    return x1_sum_t @ x2_t.transpose(-2, -1)  + x1_t @ x2_sum_t.transpose(-2, -1) - x1_t @ x2_t.transpose(-2, -1)

def multi1(x1_t,x2_t,x1_sum_t,x2_sum_t):
    return x1_sum_t @ x2_t + x1_t @ x2_sum_t - x1_t @ x2_t

class SAttention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            neuron_layer = IFNeuron,
            level = 2,
            is_softmax = True
            
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.neuron_layer = neuron_layer
        self.level = level
        self.is_softmax = is_softmax

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        self.k_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        self.v_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.attn_ReLU = nn.ReLU()
        self.attn_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=False)
        self.after_attn_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        self.proj = nn.Linear(dim, dim,bias=False)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        if self.is_softmax:
            self.Ssoftmax = spiking_softmax()
        self.T = 0

    def reset(self):
        # print("SAttention reset")
        self.q_IF.reset()
        self.k_IF.reset()
        self.v_IF.reset()
        self.attn_IF.reset()
        self.after_attn_IF.reset()
        self.proj_IF.reset()
        if self.is_softmax:
            self.Ssoftmax.reset()
        self.qkv.reset()
        self.proj.reset()
        self.T = 0

    def forward(self, x):
        B, N, C = x.shape
        # print("qkv:", self.qkv(x).shape, self.qkv.out_features)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = self.q_IF(q)
        k = self.k_IF(k)
        v = self.v_IF(v)
        
        q = q * self.scale
        q_acc = self.q_IF.acc_q * self.scale * self.q_IF.q_threshold
        attn = multi(q,k,q_acc.float(),(self.k_IF.acc_q*self.k_IF.q_threshold).float())

        if self.is_softmax:
            attn = self.Ssoftmax(attn)

        attn = self.attn_IF(attn)
        if not self.is_softmax:
            attn = attn/N
            acc_attn = self.attn_IF.acc_q*self.attn_IF.q_threshold/N

        attn = self.attn_drop(attn)

        if not self.is_softmax:
            x = multi1(attn,v,(acc_attn).float(),(self.v_IF.acc_q*self.v_IF.q_threshold).float())
        else:
            x = multi1(attn,v,(self.attn_IF.acc_q*self.attn_IF.q_threshold).float(),(self.v_IF.acc_q*self.v_IF.q_threshold).float())

        x = self.after_attn_IF(x)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        x = self.proj_IF(x)

        self.T = self.T + 1

        return x

class SpikeMaxPooling(nn.Module):
    def __init__(self,maxpool):
        super(SpikeMaxPooling,self).__init__()
        self.maxpool = maxpool
        
        self.accumulation = None
    
    def reset(self):
        self.accumulation = None

    def forward(self,x):
        old_accu = self.accumulation
        if self.accumulation is None:
            self.accumulation = x
        else:
            self.accumulation = self.accumulation + x
        
        if old_accu is None:
            output = self.maxpool(self.accumulation)
        else:
            output = self.maxpool(self.accumulation) - self.maxpool(old_accu)

        # print("output.shape",output.shape)
        # print(output[0][0][0:4][0:4])
        
        return output


class QuanConv2d(torch.nn.Conv2d):
    def __init__(self, m: torch.nn.Conv2d, quan_w_fn=None):
        assert type(m) == torch.nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.quan_w_fn = quan_w_fn

        self.weight = torch.nn.Parameter(m.weight.detach())
        # self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.detach())
        else:
            self.bias = None

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        return self._conv_forward(x, quantized_weight,self.bias)


class QuanLinear(torch.nn.Linear):
    def __init__(self, m: torch.nn.Linear, quan_w_fn=None):
        assert type(m) == torch.nn.Linear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.quan_w_fn = quan_w_fn

        self.weight = torch.nn.Parameter(m.weight.detach())
        # self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.detach())

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        return torch.nn.functional.linear(x, quantized_weight, self.bias)


class LLConv2d(nn.Module):
    def __init__(self,conv,**kwargs):
        super(LLConv2d,self).__init__()
        self.conv = conv
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.neuron_type = kwargs["neuron_type"]
        self.level = kwargs["level"]
        self.steps = 1
        self.realize_time = self.steps
        
        
    def reset(self):
        # print("LLConv2d reset")
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.realize_time = self.steps

    def forward(self,input):
        # print("LLConv2d.steps",self.steps)
        x = input
        N,C,H,W = x.shape
        F_h,F_w = self.conv.kernel_size
        S_h,S_w = self.conv.stride
        P_h,P_w = self.conv.padding
        C = self.conv.out_channels
        H = math.floor((H - F_h + 2*P_h)/S_h)+1
        W = math.floor((W - F_w + 2*P_w)/S_w)+1

        if self.zero_output is None:
            # self.zero_output = 0.0
            self.zero_output = torch.zeros(size=(N,C,H,W),device=x.device,dtype=x.dtype)

        if (not torch.is_tensor(x) and (x == 0.0)) or ((x==0.0).all()):
            self.is_work = False
            if self.realize_time > 0:
                output = self.zero_output + (self.conv.bias.data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)/self.steps if self.conv.bias is not None else 0.0)
                self.realize_time = self.realize_time - 1
                self.is_work = True
                return output
            return self.zero_output

        output = self.conv(x)

        if self.neuron_type == 'IF':
            pass
        else:
            if self.conv.bias is None:
                pass
            else:
                # if not self.first:
                #     output = output - self.conv.bias.data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                output = output - (self.conv.bias.data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) if self.conv.bias is not None else 0.0)
                if self.realize_time > 0:
                    output = output + (self.conv.bias.data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)/self.steps if self.conv.bias is not None else 0.0)
                    self.realize_time = self.realize_time - 1
                    # print("conv2d self.realize_time",self.realize_time)
                    

        self.is_work = True
        self.first = False

        return output

class LLLinear(nn.Module):
    def __init__(self,linear,**kwargs):
        super(LLLinear,self).__init__()
        self.linear = linear
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.neuron_type = kwargs["neuron_type"]
        self.level = kwargs["level"]
        self.steps = 1
        self.realize_time = self.steps
    def reset(self):
        # print("LLLinear reset")
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.realize_time = self.steps

    def forward(self,input):
        # print("LLLinear.steps",self.steps)
        x = input
        # if x.ndim == 2:
        #     B,N = x.shape
        # elif x.ndim == 3:
        #     B,C,N = x.shape
        # N = self.linear.out_features
        if x.dim() == 3:
            B, N, _ = x.shape
            D = self.linear.out_features
            shape_new = (B, N, D)
        elif x.dim() == 2:
            B, _ = x.shape
            D = self.linear.out_features
            shape_new = (B, D)
        if self.zero_output is None:
            self.zero_output = torch.zeros(size=shape_new,device=x.device,dtype=x.dtype)

        if (not torch.is_tensor(x) and (x == 0.0)) or ((x==0.0).all()):
            self.is_work = False
            if self.realize_time > 0:
                output = self.zero_output + (self.linear.bias.data.unsqueeze(0)/self.steps if self.linear.bias is not None else 0.0)
                self.realize_time = self.realize_time - 1
                self.is_work = True
                return output
            return self.zero_output

        output = self.linear(x)

        if self.neuron_type == 'IF':
            pass
        else:
            if self.linear.bias is None:
                pass
            else:
                output = output - (self.linear.bias.data.unsqueeze(0) if self.linear.bias is not None else 0.0)
                if self.realize_time > 0:
                    output = output + (self.linear.bias.data.unsqueeze(0)/self.steps if self.linear.bias is not None else 0.0)
                    self.realize_time = self.realize_time - 1


        self.is_work = True
        self.first = False

        return output


class Attention_no_softmax(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_Relu = nn.ReLU(inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_Relu(attn)/N
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MyBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, **kwargs):
        super(MyBatchNorm1d, self).__init__(**kwargs)
    
    def forward(self,x):
        x = x.transpose(1,2)
        F.batch_norm(x,self.running_mean,self.running_var,self.weight,self.bias,self.training,self.momentum,self.eps)
        x = x.transpose(1,2)
        return x
        
    

class MyLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.zeros(self.dim))
        self.bias = nn.Parameter(torch.zeros(self.dim))
        nn.init.constant_(self.weight, 1.)
        nn.init.constant_(self.bias, 0.)      
        self.running_mean = None
        self.running_var = None
        self.momentum = 0.1
        self.eps = 1e-6
    
    def forward(self,x):        
        if self.training:
            if self.running_mean is None:
                self.running_mean = nn.Parameter((1-self.momentum) * x.mean([0,-1], keepdim=True),requires_grad=False)
                self.running_var = nn.Parameter((1-self.momentum) * x.std([0,-1], keepdim=True),requires_grad=False)
            else:
                self.running_mean.data = (1-self.momentum) * x.mean([0,-1], keepdim=True) + self.momentum * self.running_mean # mean: [bsz, max_len, 1]
                self.running_var.data = (1-self.momentum) * x.std([0,-1], keepdim=True) + self.momentum * self.running_var # std: [bsz, max_len, 1]
            return self.weight * (x - self.running_mean) / (self.running_var + self.eps) + self.bias    
        else:
            running_mean = self.running_mean
            running_var = self.running_var
            return self.weight * (x - running_mean) / (running_var + self.eps) + self.bias    
        # 注意这里也在最后一个维度发生了广播
    
      