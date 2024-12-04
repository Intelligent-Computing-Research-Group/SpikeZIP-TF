
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from spike_quan_layer import MyQuan,IFNeuron,LLConv2d,LLLinear,SpikeMaxPooling,QRobertaSelfAttention,SRobertaSelfAttention,spiking_softmax,Spiking_LayerNorm,LLEmbedding
import sys
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention,RobertaIntermediate,RobertaOutput,RobertaClassificationHead,RobertaSelfOutput
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
        if isinstance(child, IFNeuron) or isinstance(child, LLConv2d) or isinstance(child, LLLinear) or isinstance(child, SRobertaSelfAttention) or isinstance(child, Spiking_LayerNorm) or isinstance(child, LLEmbedding):
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

def attn_convert(QAttn:QRobertaSelfAttention,SAttn:SRobertaSelfAttention,level,neuron_type):
    SAttn.query = LLLinear(linear = QAttn.query,neuron_type = "ST-BIF",level = level)
    SAttn.key = LLLinear(linear = QAttn.key,neuron_type = "ST-BIF",level = level)
    SAttn.value = LLLinear(linear = QAttn.value,neuron_type = "ST-BIF",level = level)

    SAttn.query_IF.neuron_type= neuron_type
    SAttn.query_IF.level = level
    SAttn.query_IF.q_threshold = QAttn.query_quan.s.data
    SAttn.query_IF.pos_max = QAttn.query_quan.pos_max
    SAttn.query_IF.neg_min = QAttn.query_quan.neg_min
    SAttn.query_IF.is_init = False

    SAttn.key_IF.neuron_type= neuron_type
    SAttn.key_IF.level = level
    SAttn.key_IF.q_threshold = QAttn.key_quan.s.data
    SAttn.key_IF.pos_max = QAttn.key_quan.pos_max
    SAttn.key_IF.neg_min = QAttn.key_quan.neg_min
    SAttn.key_IF.is_init = False

    SAttn.value_IF.neuron_type= neuron_type
    SAttn.value_IF.level = level
    SAttn.value_IF.q_threshold = QAttn.value_quan.s.data
    SAttn.value_IF.pos_max = QAttn.value_quan.pos_max
    SAttn.value_IF.neg_min = QAttn.value_quan.neg_min
    SAttn.value_IF.is_init = False

    SAttn.attn_IF.neuron_type= neuron_type
    SAttn.attn_IF.level = level
    SAttn.attn_IF.q_threshold = QAttn.attn_quan.s.data
    SAttn.attn_IF.pos_max = QAttn.attn_quan.pos_max
    SAttn.attn_IF.neg_min = QAttn.attn_quan.neg_min
    SAttn.attn_IF.is_init = False

    SAttn.after_attn_IF.neuron_type= neuron_type
    SAttn.after_attn_IF.level = level
    SAttn.after_attn_IF.q_threshold = QAttn.after_attn_quan.s.data
    SAttn.after_attn_IF.pos_max = QAttn.after_attn_quan.pos_max
    SAttn.after_attn_IF.neg_min = QAttn.after_attn_quan.neg_min
    SAttn.after_attn_IF.is_init = False



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
        self.max_T = 0
        self.model_reset = None
        self._replace_weight(self.model)
        # self.model_reset = deepcopy(self.model)
    
    def reset(self):
        # self.model = deepcopy(self.model_reset).cuda()
        reset_model(self)
    
    def _replace_weight(self,model):
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, QRobertaSelfAttention):
                config = deepcopy(child.config)
                config.neuron_type = self.neuron_type
                SAttn = SRobertaSelfAttention(config)
                attn_convert(QAttn=child,SAttn=SAttn,level=self.level,neuron_type = self.neuron_type)
                model._modules[name] = SAttn
                is_need = True
            elif isinstance(child, nn.Embedding):
                model._modules[name] = LLEmbedding(child)				
                is_need = True
            elif isinstance(child, nn.Conv2d):
                model._modules[name] = LLConv2d(child,**self.kwargs)
                is_need = True
            elif isinstance(child, nn.Linear):
                model._modules[name] = LLLinear(child,**self.kwargs)
                is_need = True
            elif isinstance(child, nn.LayerNorm):
                SNN_LN = Spiking_LayerNorm(child.weight.shape[0])
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

    def forward(self,seqs, masks, segments, labels, verbose=False):
        accu = None
        count1 = 0
        accu_per_timestep = []
        # print("self.bit",self.bit)
        # x = x*(2**self.bit-1)+0.0

        if self.Encoding_type == "rate":
            self.mean = 0.0
            self.std  = 0.0
            seqs = get_subtensors(seqs,self.mean,self.std,sample_grain=self.level)
            # print("x.shape",x.shape)
        while(1):
            self.finish_judger.reset_network_finish_flag()
            self.finish_judger.judge_finish(self)
            network_finish = self.finish_judger.network_finish
            # print(f"==================={count1}===================")
            if (count1 > 0 and network_finish) or count1 >= self.T:
                self.max_T = max(count1, self.max_T)
                break    

            if self.Encoding_type == "rate":
                if count1 < seqs.shape[0]:
                    seqs = seqs[count1]
                else:
                    seqs = torch.zeros(seqs[0].shape).to(seqs.device)
            else:
                if count1 == 0:
                    seqs = seqs
                else:
                    seqs = torch.zeros(seqs.shape).to(seqs.device)
            # elif self.neuron_type == 'IF':
            #     input = x
            # else:
            #     print("No implementation of neuron type:",self.neuron_type)
            #     sys.exit(0)
            
            output = self.model(seqs, masks, segments, labels)[1]
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
        if verbose:
            accu_per_timestep = torch.stack(accu_per_timestep,dim=0)
            return accu,count1,accu_per_timestep
        else:
            return accu,count1

def myquan_replace(model,level):
    index = 0
    cur_index = 0
    def get_index(model):
        nonlocal index
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, QRobertaSelfAttention):
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
            if isinstance(child, RobertaSelfAttention):
                # print(children)
                config = deepcopy(child.config)
                config.level = level
                qattn = QRobertaSelfAttention(config)
                qattn.query = child.query
                qattn.key = child.key
                qattn.value = child.value
                model._modules[name] = qattn
                print("index",cur_index,"myquan replace finish!!!!")
                cur_index = cur_index + 1
                is_need = True
            elif isinstance(child,RobertaIntermediate):
                model._modules[name].intermediate_act_fn = nn.Sequential(MyQuan(level,sym = False),child.intermediate_act_fn)
                is_need = True
            elif isinstance(child,RobertaOutput) or isinstance(child,RobertaSelfOutput):
                model._modules[name].dense = nn.Sequential(child.dense,MyQuan(level,sym = True))   
                is_need = True
            elif isinstance(child,RobertaClassificationHead):
                model._modules[name].act = nn.Sequential(MyQuan(level,sym = False),child.act)
                is_need = True
            if not is_need:
                _myquan_replace(child,level)
    get_index(model)
    _myquan_replace(model,level)


# ================================================================ TEST PARTS ==============================================================================

class Arguments():
    def __init__(self):
        pass

def test_myquan_replace():
    from model.models import RobertModel
    args = Arguments()
    args.pretrained = "/home/kang_you/bert_snn/model_pool/Robert_base_SST-2/roberta_sst2"
    args.num_labels = 2
    bertmodel = RobertModel(requires_grad = True,args=args)
    f = open("ann_model.txt","w")
    f.write(str(bertmodel))
    f.close()
    myquan_replace(model=bertmodel,level=32)
    f = open("qann_model.txt","w")
    f.write(str(bertmodel))
    f.close()
    



