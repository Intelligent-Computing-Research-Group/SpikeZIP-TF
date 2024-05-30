
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from spike_quan_layer import MyQuan,IFNeuron,LLConv2d,LLLinear,ORIIFNeuron,SpikeMaxPooling,QAttention,SAttention,spiking_softmax,Spiking_LayerNorm,QuanConv2d,QuanLinear,Attention_no_softmax, MyLayerNorm,MyBatchNorm1d,ORIIFNeuron
import sys
from timm.models.vision_transformer import Attention,Mlp,Block
from copy import deepcopy

def get_subtensors(tensor,mean,std,sample_grain=255,output_num=4):
    for i in range(int(sample_grain)):
        output = (tensor/sample_grain).unsqueeze(0)
        # output = (tensor).unsqueeze(0)
        if i == 0:
            accu = output
        else:
            accu = torch.cat((accu,output),dim=0)
    return accu

def reset_model(model):
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, IFNeuron) or isinstance(child, LLConv2d) or isinstance(child, LLLinear) or isinstance(child, SAttention) or isinstance(child, Spiking_LayerNorm) or isinstance(child, ORIIFNeuron):
            model._modules[name].reset()
            is_need = True
        if not is_need:
            reset_model(child)



class Judger():
	def __init__(self):
		self.network_finish=True

	def judge_finish(self,model):
		children = list(model.named_children())
		for name, child in children:
			is_need = False
			if isinstance(child, IFNeuron) or isinstance(child, LLLinear) or isinstance(child, LLConv2d):
				self.network_finish = self.network_finish and (not model._modules[name].is_work)
				# print("child",child,"network_finish",self.network_finish,"model._modules[name].is_work",(model._modules[name].is_work))
				is_need = True
			if not is_need:
				self.judge_finish(child)

	def reset_network_finish_flag(self):
		self.network_finish = True

def attn_convert(QAttn:QAttention,SAttn:SAttention,level,neuron_type):
    SAttn.qkv = LLLinear(linear = QAttn.qkv,neuron_type = "ST-BIF",level = level)
    SAttn.proj = LLLinear(linear = QAttn.proj,neuron_type = "ST-BIF",level = level)

    SAttn.q_IF.neuron_type= neuron_type
    SAttn.q_IF.level = level
    SAttn.q_IF.q_threshold.data = QAttn.quan_q.s.data
    SAttn.q_IF.pos_max = QAttn.quan_q.pos_max
    SAttn.q_IF.neg_min = QAttn.quan_q.neg_min
    SAttn.q_IF.is_init = False

    SAttn.k_IF.neuron_type= neuron_type
    SAttn.k_IF.level = level
    SAttn.k_IF.q_threshold.data = QAttn.quan_k.s.data
    SAttn.k_IF.pos_max = QAttn.quan_k.pos_max
    SAttn.k_IF.neg_min = QAttn.quan_k.neg_min
    SAttn.k_IF.is_init = False

    SAttn.v_IF.neuron_type= neuron_type
    SAttn.v_IF.level = level
    SAttn.v_IF.q_threshold.data = QAttn.quan_v.s.data
    SAttn.v_IF.pos_max = QAttn.quan_v.pos_max
    SAttn.v_IF.neg_min = QAttn.quan_v.neg_min
    SAttn.v_IF.is_init = False

    SAttn.attn_IF.neuron_type= neuron_type
    SAttn.attn_IF.level = level
    SAttn.attn_IF.q_threshold.data = QAttn.attn_quan.s.data
    SAttn.attn_IF.pos_max = QAttn.attn_quan.pos_max
    SAttn.attn_IF.neg_min = QAttn.attn_quan.neg_min
    SAttn.attn_IF.is_init = False

    SAttn.after_attn_IF.neuron_type= neuron_type
    SAttn.after_attn_IF.level = level
    SAttn.after_attn_IF.q_threshold.data = QAttn.after_attn_quan.s.data
    SAttn.after_attn_IF.pos_max = QAttn.after_attn_quan.pos_max
    SAttn.after_attn_IF.neg_min = QAttn.after_attn_quan.neg_min
    SAttn.after_attn_IF.is_init = False

    SAttn.proj_IF.neuron_type= neuron_type
    SAttn.proj_IF.level = level
    SAttn.proj_IF.q_threshold.data = QAttn.quan_proj.s.data
    SAttn.proj_IF.pos_max = QAttn.quan_proj.pos_max
    SAttn.proj_IF.neg_min = QAttn.quan_proj.neg_min
    SAttn.proj_IF.is_init = False

    SAttn.attn_drop = QAttn.attn_drop
    SAttn.proj_drop = QAttn.proj_drop

def open_dropout(model):
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, nn.Dropout):
            child.train()
            print(child)
            is_need = True
        if not is_need:
            open_dropout(child)



def cal_l1_loss(model):
    l1_loss = 0.0
    def _cal_l1_loss(model):
        nonlocal l1_loss
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, MyQuan):
                l1_loss = l1_loss + child.act_loss
                is_need = True
            if not is_need:
                _cal_l1_loss(child)
    _cal_l1_loss(model)
    return l1_loss

class SNNWrapper(nn.Module):
    
    def __init__(self, ann_model, cfg, time_step = 2000,Encoding_type="rate",**kwargs):
        super(SNNWrapper, self).__init__()
        self.T = time_step
        self.cfg = cfg
        self.finish_judger = Judger()
        self.Encoding_type = Encoding_type
        self.level = kwargs["level"]
        self.neuron_type = kwargs["neuron_type"]
        self.model = ann_model
        self.kwargs = kwargs
        self.model_name = kwargs["model_name"]
        self.is_softmax = kwargs["is_softmax"]
        self.max_T = 0
        self.visualize = False
        # self.model_reset = None
        if self.model_name.count("vit") > 0:
            self.pos_embed = deepcopy(self.model.pos_embed.data)
            self.cls_token = deepcopy(self.model.cls_token.data)

        self._replace_weight(self.model)
        # self.model_reset = deepcopy(self.model)        
    
    def hook_mid_feature(self):
        self.feature_list = []
        self.input_feature_list = []
        def _hook_mid_feature(module, input, output):
            self.feature_list.append(output)
            self.input_feature_list.append(input[0])
        self.model.blocks[11].norm2[1].register_forward_hook(_hook_mid_feature)
        # self.model.blocks[11].attn.attn_IF.register_forward_hook(_hook_mid_feature)
    
    def get_mid_feature(self):
        self.feature_list = torch.stack(self.feature_list,dim=0)
        self.input_feature_list = torch.stack(self.input_feature_list,dim=0)
        print("self.feature_list",self.feature_list.shape) 
        print("self.input_feature_list",self.input_feature_list.shape) 
            
    def reset(self):
        # self.model = deepcopy(self.model_reset).cuda()
        self.model.pos_embed.data = deepcopy(self.pos_embed).cuda()
        self.model.cls_token.data = deepcopy(self.cls_token).cuda()
        # print(self.model.pos_embed)
        # print(self.model.cls_token)
        reset_model(self)
    
    def _replace_weight(self,model):
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, QAttention):
                SAttn = SAttention(dim=child.num_heads*child.head_dim,num_heads=child.num_heads,level=self.level,is_softmax=self.is_softmax,neuron_layer=IFNeuron)
                attn_convert(QAttn=child,SAttn=SAttn,level=self.level,neuron_type = self.neuron_type)
                model._modules[name] = SAttn
                is_need = True
            elif isinstance(child, nn.Conv2d) or isinstance(child, QuanConv2d):
                model._modules[name] = LLConv2d(child,**self.kwargs)
                is_need = True
            elif isinstance(child, nn.Linear) or isinstance(child, QuanLinear):
                model._modules[name] = LLLinear(child,**self.kwargs)
                is_need = True
            elif isinstance(child, nn.LayerNorm):
                SNN_LN = Spiking_LayerNorm(child.normalized_shape[0])
                if child.elementwise_affine:
                    SNN_LN.layernorm.weight.data = child.weight.data
                    SNN_LN.layernorm.bias.data = child.bias.data                
                model._modules[name] = SNN_LN
                is_need = True
            elif isinstance(child, MyQuan):
                neurons = IFNeuron(q_threshold = torch.tensor(1.0),sym=child.sym,level = child.pos_max)
                neurons.q_threshold=child.s.data
                neurons.neuron_type=self.neuron_type
                neurons.level = self.level
                neurons.pos_max = child.pos_max
                neurons.neg_min = child.neg_min
                neurons.is_init = False
                model._modules[name] = neurons     
                is_need = True
            elif isinstance(child, nn.ReLU):
                model._modules[name] = nn.Identity()
                is_need = True
            if not is_need:            
                self._replace_weight(child)

    def forward(self,x, verbose=False):
        accu = None
        count1 = 0
        accu_per_timestep = []
        # print("self.bit",self.bit)
        # x = x*(2**self.bit-1)+0.0
        if self.visualize:
            self.hook_mid_feature()
        if self.Encoding_type == "rate":
            self.mean = 0.0
            self.std  = 0.0
            x = get_subtensors(x,self.mean,self.std,sample_grain=self.level)
            # print("x.shape",x.shape)
        while(1):
            self.finish_judger.reset_network_finish_flag()
            self.finish_judger.judge_finish(self)
            network_finish = self.finish_judger.network_finish
            # print(f"==================={count1}===================")
            if (count1 > 0 and network_finish) or count1 >= self.T:
                self.max_T = max(count1, self.max_T)
                break
            # if self.neuron_type.count("QFFS") != -1 or self.neuron_type == 'ST-BIF':
            if self.model_name.count("vit")>0 and count1 > 0:
                self.model.pos_embed = nn.Parameter(torch.zeros(1, self.model.patch_embed.num_patches + 1, self.model.
                                                                embed_dim).to(x.device))
                self.model.cls_token = nn.Parameter(torch.zeros(1, 1, self.model.embed_dim).to(x.device))
            if self.Encoding_type == "rate":
                if count1 < x.shape[0]:
                    input = x[count1]
                else:
                    input = torch.zeros(x[0].shape).to(x.device)            
            else:
                if count1 == 0:
                    input = x
                else:
                    input = torch.zeros(x.shape).to(x.device)
            # elif self.neuron_type == 'IF':
            #     input = x
            # else:
            #     print("No implementation of neuron type:",self.neuron_type)
            #     sys.exit(0)
            
            output = self.model(input)
            # print(count1,output[0,0:100])
            # print(count1,"output",torch.abs(output.sum()))
            
            if count1 == 0:
                accu = output+0.0
            else:
                accu = accu+output
            if verbose:
                accu_per_timestep.append(accu)
            # print("accu",accu.sum(),"output",output.sum())
            count1 = count1 + 1
            if count1 % 100 == 0:
                print(count1)

        # print("verbose",verbose)
        print("\nTime Step:",count1)
        if self.visualize:
            self.get_mid_feature()
            torch.save(self.feature_list,"model_blocks11_norm2.pth")
            torch.save(self.input_feature_list,"model_blocks11_norm2_input.pth")
        if verbose:
            accu_per_timestep = torch.stack(accu_per_timestep,dim=0)
            return accu,count1,accu_per_timestep
        else:
            return accu,count1

def remove_softmax(model):
    children = list(model.named_children())
    for name, child in children:
        is_need = False
        if isinstance(child, Attention):
            reluattn = Attention_no_softmax(dim=child.num_heads*child.head_dim,num_heads=child.num_heads)
            reluattn.qkv = child.qkv
            reluattn.attn_drop = child.attn_drop
            reluattn.proj = child.proj
            reluattn.proj_drop = child.proj_drop
            is_need = True
            model._modules[name] = reluattn
        # elif isinstance(child, nn.LayerNorm):
        #     LN = MyBatchNorm1d(num_features = child.normalized_shape[0])
        #     # LN.weight.data = child.weight
        #     # LN.bias.data = child.bias
        #     model._modules[name] = LN
        if not is_need:
            remove_softmax(child)



def myquan_replace(model,level,weight_bit=32, is_softmax = True):
    index = 0
    cur_index = 0
    def get_index(model):
        nonlocal index
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, QAttention):
                index = index + 1
                is_need = True
            if not is_need:
                get_index(child)

    def _myquan_replace(model,level):
        nonlocal index
        nonlocal cur_index
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, Block):
                # print(children)
                qattn = QAttention(dim=child.attn.num_heads*child.attn.head_dim,num_heads=child.attn.num_heads,level=level,is_softmax=is_softmax)
                qattn.qkv = child.attn.qkv
                # qattn.q_norm = child.q_norm
                # qattn.k_norm = child.k_norm
                qattn.attn_drop = child.attn.attn_drop
                qattn.proj = child.attn.proj
                qattn.proj_drop = child.attn.proj_drop
                model._modules[name].attn = qattn
                # model._modules[name].act1 = MyQuan(level, sym=True)
                # model._modules[name].act2 = MyQuan(level, sym=True)
                model._modules[name].norm1 = nn.Sequential(child.norm1, MyQuan(level, sym=True))
                model._modules[name].norm2 = nn.Sequential(child.norm2, MyQuan(level, sym=True))
                model._modules[name].mlp.act = nn.Sequential(MyQuan(level, sym=False), child.mlp.act)
                model._modules[name].mlp.fc2 = nn.Sequential(child.mlp.fc2,MyQuan(level, sym=True))
                print("index",cur_index,"myquan replace finish!!!!")
                cur_index = cur_index + 1
                is_need = True
            # if isinstance(child, Attention):
            #     # print(children)
            #     qattn = QAttention(dim=child.num_heads*child.head_dim,num_heads=child.num_heads,level=level)
            #     qattn.qkv = child.qkv
            #     # qattn.q_norm = child.q_norm
            #     # qattn.k_norm = child.k_norm
            #     qattn.attn_drop = child.attn_drop
            #     qattn.proj = child.proj
            #     qattn.proj_drop = child.proj_drop
            #     model._modules[name] = qattn
            #     print("index",cur_index,"myquan replace finish!!!!")
            #     cur_index = cur_index + 1
            #     is_need = True
            # elif isinstance(child,Mlp):
            #     model._modules[name].act = nn.Sequential(MyQuan(level,sym = False),child.act)
            #     model._modules[name].fc2 = nn.Sequential(child.fc2,MyQuan(level,sym = True))
            #     is_need = True
            elif isinstance(child, nn.Conv2d):
                model._modules[name] = nn.Sequential(child,MyQuan(level,sym = True))
                is_need = True
            # elif isinstance(child, Block):
            #     model._modules[name].norm1 = nn.Sequential(child.norm1,MyQuan(level,sym = True))
            #     model._modules[name].norm2 = nn.Sequential(child.norm2,MyQuan(level,sym = True))
            #     is_need = False
            elif isinstance(child, nn.LayerNorm):
                model._modules[name] = nn.Sequential(child,MyQuan(level,sym = True))
                is_need = True
            if not is_need:
                _myquan_replace(child,level)
    
    def _weight_quantization(model,weight_bit):
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, nn.Conv2d):
                model._modules[name] = QuanConv2d(m=child,quan_w_fn=MyQuan(level = 2**weight_bit,sym=True))
                is_need = True
            elif isinstance(child, nn.Linear):
                model._modules[name] = QuanLinear(m=child,quan_w_fn=MyQuan(level = 2**weight_bit,sym=True))
                is_need = True
            if not is_need:
                _weight_quantization(child,weight_bit)
                
    get_index(model)
    _myquan_replace(model,level)
    if weight_bit < 32:
        _weight_quantization(model,weight_bit)



