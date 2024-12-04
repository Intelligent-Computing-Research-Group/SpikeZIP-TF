import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
import math
from copy import deepcopy
from typing import List, Optional, Tuple, Union
import math
# torch.set_default_dtype(torch.double)
# torch.set_default_tensor_type(torch.DoubleTensor)

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

    def __repr__(self):
            return f"IFNeuron(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, neg_min={self.neg_min}, q_threshold={self.q_threshold})"
    
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
            return x
        
        if not torch.is_tensor(self.cur_output):
            self.cur_output = torch.zeros(x.shape,dtype=x.dtype).to(x.device)
            self.acc_q = torch.zeros(x.shape).to(x.device)
            self.q = torch.zeros(x.shape).to(x.device) + 0.5

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


class MyQuan(nn.Module):
    def __init__(self,level,sym = False,**kwargs):
        super(MyQuan,self).__init__()
        # self.level_init = level
        self.s_init = 0.0
        self.level = level
        self.sym = sym
        if level >= 256:
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
        elif self.init_state<self.batch_init and self.training:
            self.s.data = 0.9*self.s.data + 0.1*torch.tensor(torch.mean(torch.abs(x.detach()))*2/(math.sqrt(max_val.detach().abs().mean())),dtype=torch.float32)
            self.init_state += 1
            
        elif self.init_state==self.batch_init and self.training:
            # self.s = torch.nn.Parameter(self.s)
            self.init_state += 1
            print("initialize finish!!!!")

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

class QRobertaSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.query_quan = MyQuan(level=config.level,sym=True)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.key_quan = MyQuan(level=config.level,sym=True)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.value_quan = MyQuan(level=config.level,sym=True)
        self.attn_quan = MyQuan(level=config.level,sym=False)
        self.after_attn_quan = MyQuan(level=config.level,sym=False)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query_quan(self.query(hidden_states))

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key_quan(self.key(encoder_hidden_states)))
            value_layer = self.transpose_for_scores(self.value_quan(self.value(encoder_hidden_states)))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key_quan(self.key(hidden_states)))
            value_layer = self.transpose_for_scores(self.value_quan(self.value(hidden_states)))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key_quan(self.key(hidden_states)))
            value_layer = self.transpose_for_scores(self.value_quan(self.value(hidden_states)))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        attention_probs = self.attn_quan(attention_probs)
        
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = self.after_attn_quan(context_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


def multi(x1_t,x2_t,x1_sum_t,x2_sum_t):
    return x1_sum_t @ x2_t.transpose(-2, -1)  + x1_t @ x2_sum_t.transpose(-2, -1) - x1_t @ x2_t.transpose(-2, -1)

def multi1(x1_t,x2_t,x1_sum_t,x2_sum_t):
    return x1_sum_t @ x2_t + x1_t @ x2_sum_t - x1_t @ x2_t

class SRobertaSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = LLLinear(linear = nn.Linear(config.hidden_size, self.all_head_size),level=config.level,neuron_type=config.neuron_type)
        self.query_IF = IFNeuron(q_threshold=1.0,level=config.level,sym=True)
        self.key = LLLinear(linear = nn.Linear(config.hidden_size, self.all_head_size),level=config.level,neuron_type=config.neuron_type)
        self.key_IF = IFNeuron(q_threshold=1.0,level=config.level,sym=True)
        self.value = LLLinear(linear = nn.Linear(config.hidden_size, self.all_head_size),level=config.level,neuron_type=config.neuron_type)
        self.value_IF = IFNeuron(q_threshold=1.0,level=config.level,sym=True)
        self.attn_IF = IFNeuron(q_threshold=1.0,level=config.level,sym=False)
        self.after_attn_IF = IFNeuron(q_threshold=1.0,level=config.level,sym=False)
        self.Ssoftmax = spiking_softmax()

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def reset(self):
        # print("SAttention reset")
        self.query_IF.reset()
        self.key_IF.reset()
        self.value_IF.reset()
        self.attn_IF.reset()
        self.after_attn_IF.reset()
        self.Ssoftmax.reset()
        self.query.reset()
        self.key.reset()
        self.value.reset()
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query_IF(self.query(hidden_states))

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key_IF(self.key(encoder_hidden_states)))
            value_layer = self.transpose_for_scores(self.value_IF(self.value(encoder_hidden_states)))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key_IF(self.key(hidden_states)))
            value_layer = self.transpose_for_scores(self.value_IF(self.value(hidden_states)))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key_IF(self.key(hidden_states)))
            value_layer = self.transpose_for_scores(self.value_IF(self.value(hidden_states)))

        
        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.

        attention_scores = multi(query_layer,key_layer,self.transpose_for_scores(self.query_IF.acc_q*self.query_IF.q_threshold),self.transpose_for_scores(self.key_IF.acc_q*self.key_IF.q_threshold))

        # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.Ssoftmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        attention_probs = self.attn_IF(attention_probs)
        
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = multi1(attention_probs,value_layer,(self.attn_IF.acc_q*self.attn_IF.q_threshold),self.transpose_for_scores(self.value_IF.acc_q*self.value_IF.q_threshold))

        context_layer = self.after_attn_IF(context_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

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
    
class LLConv2d(nn.Module):
    def __init__(self,conv,**kwargs):
        super(LLConv2d,self).__init__()
        self.conv = conv
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.neuron_type = kwargs["neuron_type"]
        self.level = kwargs["level"]
        self.steps = self.level
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
        self.steps = self.level
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


class LLEmbedding(nn.Module):
    def __init__(self,embedding):
        super(LLEmbedding,self).__init__()
        self.embedding = embedding
        self.T = 0
        self.shape = None
    
    def reset(self):
        self.T = 0
    
    def forward(self, x):    
        if self.T == 0:
            output = self.embedding(x)
            self.shape = output.shape
            self.T = self.T + 1
            return output
        else:
            return torch.zeros(self.shape,device=x.device)


#==================================================== TEST Parts=======================================================================
class Arguments():
    def __init__(self):
        pass

def test_sattn_qattn_convert():
    from model.models import RobertModel
    from spike_quan_wrapper import attn_convert
    
    args = Arguments()
    args.pretrained = "/home/kang_you/bert_snn/model_pool/Robert_base_SST-2/roberta_sst2"
    args.num_labels = 2
    bertmodel = RobertModel(requires_grad = True,args=args)
    qconfig = deepcopy(bertmodel.bert.config)
    qconfig.level = 32
    qattn = QRobertaSelfAttention(qconfig).eval()

    sconfig = deepcopy(bertmodel.bert.config)
    sconfig.level = qconfig.level
    sconfig.neuron_type = "ST-BIF"
    sattn = SRobertaSelfAttention(sconfig).eval()

    # qattn.query_quan.s.data = torch.tensor(0.1)
    # qattn.key_quan.s.data = torch.tensor(0.1)
    # qattn.value_quan.s.data = torch.tensor(0.1)
    # qattn.attn_quan.s.data = torch.tensor(0.1)
    # qattn.after_attn_quan.s.data = torch.tensor(0.1)

    attn_convert(qattn,sattn,level=qconfig.level,neuron_type="ST-BIF")
    print(qattn)
    print(sattn)

    sattn.reset()
    x = torch.randn(1,129,768)*10

    out1 = qattn(x)[0]

    T = 32

    x_div = x/T
    accu = None
    t = 0
    x_zero = torch.zeros(x_div.shape)

    while(1):
        if t < T:
            out = sattn(x_div)[0]
        else:
            out = sattn(x_zero)[0]
        if t == 0:
            accu = out + 0.0
        else:
            accu = accu + out    
        t = t + 1
        # print("================================")
        if t > 4*T:
            break

    error = torch.sum(~(torch.abs(out1 - accu) < 1e-5))/accu.numel()
    max_error = torch.max(torch.abs(out1 - accu))
    print(error)
    print(out1.shape)
    print(max_error)
    # print(out1)
    # print(accu)


    sattn.reset()
    x = torch.randn(1,129,768)*10

    out1 = qattn(x)[0]

    T = 32

    x_div = x/T
    accu = None
    t = 0
    x_zero = torch.zeros(x_div.shape)

    while(1):
        if t < T:
            out = sattn(x_div)[0]
        else:
            out = sattn(x_zero)[0]
        if t == 0:
            accu = out + 0.0
        else:
            accu = accu + out    
        t = t + 1
        # print("================================")
        if t > 4*T:
            break

    error = torch.sum(~(torch.abs(out1 - accu) < 1e-5))/accu.numel()
    max_error = torch.max(torch.abs(out1 - accu))
    print(error)
    print(out1.shape)
    print(max_error)
    print(out1)
    print(accu)
    
    assert error == 0, "The conversion from quantized attention to spiking attention is wrong!!!"
    
def test_LLEmbedding():
    embedding = nn.Embedding(10,10)
    llembedding = LLEmbedding(embedding)
    
    x = torch.randint(0,10,(2,3,2))
    
    output1 = embedding(x)
    
    output2_1 = llembedding(x)
    output2_2 = llembedding(x)
    
    assert (output1 == output2_1).all(),"The first output of Embedding and LLEmbedding is not equivalence"
    assert (output2_2 == 0).all(),"The second output of LLEmbedding is not zero tensor"
    assert output1.shape == output2_2.shape,"The shape should be same"
    
    llembedding.reset()
    x = torch.randint(0,10,(2,3,2))
    
    output1 = embedding(x)
    
    output2_1 = llembedding(x)
    output2_2 = llembedding(x)
    
    assert (output1 == output2_1).all(),"Reset: The first output of Embedding and LLEmbedding is not equivalence"
    assert (output2_2 == 0).all(),"Reset: The second output of LLEmbedding is not zero tensor"
    assert output1.shape == output2_2.shape,"Reset: The shape should be same"
    print("Test LLEmbedding Finish!!!")
