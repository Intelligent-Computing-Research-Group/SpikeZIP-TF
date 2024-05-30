'''
Copyright (C) 2022 Guangyao Chen - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import sys

import torch.nn as nn

from .engine import get_syops_pytorch
from .utils import syops_to_string, params_to_string
import re
from .ops import IFNeuron
from .engine import SNNWrapper,MyQuan, IFNeuron

# ssa_info = {'depth': 8, 'Nheads': 8, 'embSize': 384, 'patchSize': 14, 'Tsteps': 4}  # lifconvbn-8-384
# ssa_info = {'depth': 8, 'Nheads': 8, 'embSize': 512, 'patchSize': 14, 'Tsteps': 4}  # lifconvbn-8-512
# ssa_info = {'depth': 12, 'Nheads': 12, 'embSize': 768, 'patchSize': 16, 'Tsteps': 64}  # base
ssa_info = {'depth': 12, 'Nheads': 6, 'embSize': 384, 'patchSize': 16, 'Tsteps': 15}  # small
# ssa_info = {'depth': 24, 'Nheads': 16, 'embSize': 1024, 'patchSize': 16, 'Tsteps': 32}  # large

def replace_decimal_strings(input_string):
    pattern = r'\.(\d+)'
    
    replaced_string = re.sub(pattern, r'[\1]', input_string)

    return replaced_string

def get_energy_cost(model, ssa_info):
    # calculate energy consumption according to E_mac = 4.6 pJ and E_ac = 0.9 pJ
    print('Calculating energy consumption ...')
    conv_linear_layers_info = []
    Nac = 0
    Nmac = 0
    for name, module in model.named_modules():
        # print(name, module)
        # isinstance(model.head, nn.Linear) -> True
        # isinstance(model.head, nn.Conv1d) -> False
        # isinstance(model.block[7].mlp.fc1_conv, nn.Conv2d) -> True
        # isinstance(model.block.7.mlp.fc1_conv, nn.Conv1d) -> error (invalid syntax)
        # model.patch_embed.proj_conv2 -> Conv2d(96, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        
        # if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):  # obtain same results
        if "conv" in name or "linear" in name or isinstance(module,nn.LayerNorm) or isinstance(module,IFNeuron): # SpikeZIP: linear in name
            # if 'block' in name:
            #     name_split = name.split('.', 2)
            #     name = f'block[{name_split[1]}].{name_split[2]}'
                # name = f'{name_split[0]}[{int(name_split[1])}].{name_split[2]}'
            # print(name)
            if isinstance(module,MyQuan):
                continue
            # print(replace_decimal_strings(f'model.{name}.accumulated_syops_cost'))
            accumulated_syops_cost = eval(replace_decimal_strings(f'model.{name}.accumulated_syops_cost'))
            if "conv" in name:
                accumulated_syops_cost[3] = accumulated_syops_cost[3]*ssa_info['Tsteps']
            tinfo = (name, module, accumulated_syops_cost)
            conv_linear_layers_info.append(tinfo)
            # print(accumulated_syops_cost[3])
            if abs(accumulated_syops_cost[3] - 100) < 1e-4:  # fr = 100%
                Nmac += accumulated_syops_cost[2]
                # Nmac += accumulated_syops_cost[0] * accumulated_syops_cost[3] / 100  # obtain same results
            else:
                Nac += accumulated_syops_cost[1]
                # Nac += accumulated_syops_cost[0] * accumulated_syops_cost[3] / 100  # obtain same results
    print('Info of Conv/Linear layers: ')
    for tinfo in conv_linear_layers_info:
        print(tinfo)
                
    # calculate ops for SSA
    print(isinstance(model.module,SNNWrapper))
    if isinstance(model.module,SNNWrapper):
        print('SSA info: \n', ssa_info)
        depth = ssa_info['depth']
        Nheads = ssa_info['Nheads']
        embSize = ssa_info['embSize']
        Tsteps = ssa_info['Tsteps']
        patchSize = ssa_info['patchSize']
        embSize_per_head = int(embSize/Nheads)
        SSA_Nac_base = Tsteps * Nheads * pow(patchSize, 2) * embSize_per_head * embSize_per_head
        qkv_fr = []
        for d in range(depth):    
            q_lif_r = eval(f'model.module.model.blocks[{d}].attn.q_IF.__syops__[3]') / (100)
            k_lif_r = eval(f'model.module.model.blocks[{d}].attn.k_IF.__syops__[3]') / (100)
            v_lif_r = eval(f'model.module.model.blocks[{d}].attn.v_IF.__syops__[3]') / (100)
            qkv_fr.append([q_lif_r, k_lif_r, v_lif_r])
            # calculate the number of ACs for Q*K*V matrix computation
            tNac = SSA_Nac_base*(q_lif_r + k_lif_r + min(q_lif_r,k_lif_r))
            tNac = tNac + SSA_Nac_base*(v_lif_r + min(q_lif_r,k_lif_r,v_lif_r) + min(q_lif_r,k_lif_r))
            # tNac = 0
            Nac += tNac
        print('Firing rate of Q/K/V inputs in each block: ')
        print(qkv_fr)
    
    # calculate energy consumption according to E_mac = 4.6 pJ (1e-12 J) and E_ac = 0.9 pJ
    Nmac = Nmac / 1e9 # G
    Nac = Nac / 1e9 # G
    E_mac = Nmac * 4.6 # mJ
    E_ac = Nac * 0.9 # mJ
    E_all = E_mac + E_ac
    print(f"Number of operations: {Nmac} G MACs, {Nac} G ACs")
    print(f"Energy consumption: {E_all} mJ")
    return


def get_model_complexity_info(model, input_res, dataloader=None,
                              print_per_layer_stat=True,
                              as_strings=True,
                              input_constructor=None, ost=sys.stdout,
                              verbose=False, ignore_modules=[],
                              custom_modules_hooks={}, backend='pytorch',
                              syops_units=None, param_units=None,
                              output_precision=2):
    assert type(input_res) is tuple
    assert len(input_res) >= 1
    assert isinstance(model, nn.Module)


    if backend == 'pytorch':
        syops_count, params_count, syops_model = get_syops_pytorch(model, input_res, dataloader,
                                                      print_per_layer_stat,
                                                      input_constructor, ost,
                                                      verbose, ignore_modules,
                                                      custom_modules_hooks,
                                                      output_precision=output_precision,
                                                      syops_units=syops_units,
                                                      param_units=param_units)
        # calculate energy consumption according to E_mac = 4.6 pJ and E_ac = 0.9 pJ
        get_energy_cost(syops_model, ssa_info)
    else:
        raise ValueError('Wrong backend name')

    if as_strings:
        syops_string = syops_to_string(
            syops_count[0],
            units=syops_units,
            precision=output_precision
        )
        ac_syops_string = syops_to_string(
            syops_count[1],
            units=syops_units,
            precision=output_precision
        )
        mac_syops_string = syops_to_string(
            syops_count[2],
            units=syops_units,
            precision=output_precision
        )
        params_string = params_to_string(
            params_count,
            units=param_units,
            precision=output_precision
        )
        return [syops_string, ac_syops_string, mac_syops_string], params_string

    return syops_count, params_count
