# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch
import wandb
from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
from copy import deepcopy
import torch.nn.functional as F
from spike_quan_wrapper import open_dropout
from torch.nn.utils import prune
import re


def get_logits_loss(fc_t, fc_s, one_hot_label, temp, num_classes=1000):
    s_input_for_softmax = fc_s / temp
    t_input_for_softmax = fc_t / temp

    softmax = torch.nn.Softmax(dim=1)
    logsoftmax = torch.nn.LogSoftmax()

    t_soft_label = softmax(t_input_for_softmax)

    softmax_loss = - torch.sum(t_soft_label * logsoftmax(s_input_for_softmax), 1, keepdim=True)

    fc_s_auto = fc_s.detach()
    fc_t_auto = fc_t.detach()
    log_softmax_s = logsoftmax(fc_s_auto)
    log_softmax_t = logsoftmax(fc_t_auto)
    # one_hot_label = F.one_hot(label, num_classes=num_classes).float()
    softmax_loss_s = - torch.sum(one_hot_label * log_softmax_s, 1, keepdim=True)
    softmax_loss_t = - torch.sum(one_hot_label * log_softmax_t, 1, keepdim=True)

    focal_weight = softmax_loss_s / (softmax_loss_t + 1e-7)
    ratio_lower = torch.zeros(1).cuda()
    focal_weight = torch.max(focal_weight, ratio_lower)
    focal_weight = 1 - torch.exp(- focal_weight)
    softmax_loss = focal_weight * softmax_loss

    soft_loss = (temp ** 2) * torch.mean(softmax_loss)

    return soft_loss

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            if args.mode != "SNN":
                outputs = model(samples)
            else:
                outputs, counts = model(samples, verbose=False)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        if args.mode == "SNN":
            model.module.reset()

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
            if args.mode == "SNN":
                log_writer.add_scalar('counts', counts, epoch_1000x)
            if args.wandb:
                wandb.log({'loss_curve': loss_value_reduce}, step=epoch_1000x)
                wandb.log({'lr_curve': max_lr}, step=epoch_1000x)
                if args.mode == "SNN":
                    wandb.log({'counts': counts}, step=epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def replace_decimal_strings(input_string):
    pattern = r'\.(\d+)'
    
    replaced_string = re.sub(pattern, r'[\1]', input_string)

    return replaced_string

def unstruct_prune(model,ratio):
    
    # reset weight_mask
    for name, m in model.named_modules():
        if isinstance(m,torch.nn.Linear) or isinstance(m,torch.nn.Conv2d):
            if hasattr(m,"weight_mask"):
                print(m)
                m.weight.data = m.weight_orig
                m.weight_mask[m.weight_mask==0] = 1

    parameters_to_prune = []
    for name, m in model.named_modules():
        # if isinstance(m,torch.nn.Linear) or isinstance(m,torch.nn.Conv2d):
        #     parameters_to_prune.append((m ,'weight'))
        if name.count("proj")>0 or name.count("fc2")>0:
            if isinstance(m,torch.nn.Sequential) and isinstance(m[0],torch.nn.Linear):
                # print(name,m)
                parameters_to_prune.append((m[0],'weight'))
            elif isinstance(m,torch.nn.Linear):
                # print(name,m)
                parameters_to_prune.append((m ,'weight'))
    # print(tuple(parameters_to_prune),ratio)

    # global_unstructured
    prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
        amount=ratio,
    )
    zero_number = 0
    total_bumber = 0
    for name, m in model.named_modules():
        if name.count("proj")>0 or name.count("fc2")>0:
            if isinstance(m,torch.nn.Sequential) and isinstance(m[0],torch.nn.Linear):
                zero_number = zero_number + torch.sum(m[0].weight==0)
                total_bumber = total_bumber + m[0].weight.numel()
            elif isinstance(m,torch.nn.Linear):
                zero_number = zero_number + torch.sum(m.weight==0)
                total_bumber = total_bumber + m.weight.numel()

    print("prune finish!!!!! global sparsity:",(zero_number/total_bumber)*100)
    
def train_one_epoch_distill_prune(model: torch.nn.Module, model_teacher: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    model_teacher.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    # first prune for a certain ratio
    unstruct_prune(model,args.ratio[epoch])
    
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            outputs_teacher = model_teacher(samples)
            loss = criterion(outputs, targets)
            loss_distill = get_logits_loss(outputs_teacher, outputs, targets, args.temp)
            loss_all = loss + loss_distill

        loss_value = loss.item()
        loss_distill_value = loss_distill.item()
        loss_all_value = loss_all.item()

        if not math.isfinite(loss_all_value):
            print("Loss is {}, stopping training".format(loss_all_value))
            sys.exit(1)

        loss_all /= accum_iter
        loss_scaler(loss_all, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss_all=loss_all_value, loss=loss_value, loss_distill=loss_distill_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_all_value_reduce = misc.all_reduce_mean(loss_all_value)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_distill_value_reduce = misc.all_reduce_mean(loss_distill_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss_all', loss_all_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss_distill', loss_distill_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
            if args.wandb:
                wandb.log({'loss_all_curve': loss_all_value_reduce}, step=epoch_1000x)
                wandb.log({'loss_curve': loss_value_reduce}, step=epoch_1000x)
                wandb.log({'loss_distill_curve': loss_distill_value_reduce}, step=epoch_1000x)
                wandb.log({'lr_curve': max_lr}, step=epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_distill(model: torch.nn.Module, model_teacher: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    model_teacher.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        # print(model.module.blocks[0].norm1[0].weight)
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            outputs_teacher = model_teacher(samples)
            loss = criterion(outputs, targets)
            loss_distill = get_logits_loss(outputs_teacher, outputs, targets, args.temp)
            loss_all = loss + loss_distill
        loss_value = loss.item()
        loss_distill_value = loss_distill.item()
        loss_all_value = loss_all.item()

        if not math.isfinite(loss_all_value):
            print("Loss is {}, stopping training".format(loss_all_value))
            sys.exit(1)

        loss_all /= accum_iter
        loss_scaler(loss_all, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss_all=loss_all_value, loss=loss_value, loss_distill=loss_distill_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_all_value_reduce = misc.all_reduce_mean(loss_all_value)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_distill_value_reduce = misc.all_reduce_mean(loss_distill_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss_all', loss_all_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss_distill', loss_distill_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
            if args.wandb:
                wandb.log({'loss_all_curve': loss_all_value_reduce}, step=epoch_1000x)
                wandb.log({'loss_curve': loss_value_reduce}, step=epoch_1000x)
                wandb.log({'loss_distill_curve': loss_distill_value_reduce}, step=epoch_1000x)
                wandb.log({'lr_curve': max_lr}, step=epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# @torch.no_grad()
# def evaluate(data_loader, model, device):
#     criterion = torch.nn.CrossEntropyLoss()
#
#     metric_logger = misc.MetricLogger(delimiter="  ")
#     header = 'Test:'
#
#     # switch to evaluation mode
#     model.eval()
#
#     for batch in metric_logger.log_every(data_loader, 10, header):
#         images = batch[0]
#         target = batch[-1]
#         images = images.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)
#
#         # compute output
#         with torch.cuda.amp.autocast():
#             output = model(images)
#             loss = criterion(output, target)
#
#         acc1, acc5 = accuracy(output, target, topk=(1, 5))
#
#         batch_size = images.shape[0]
#         metric_logger.update(loss=loss.item())
#         metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
#         metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
#           .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
#
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    # open_dropout(model)
    # if args.mode == "SNN":
    #     model.max_T = 0
    total_num = 0
    correct_per_timestep = None
    
    max_T = 0
    # count1 = 0

    for batch in metric_logger.log_every(data_loader, 1, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            if args.mode != "SNN":
                output = model(images)
            else:
                # accu_per_timestep: cur_T * B * n_classes
                output, count, accu_per_timestep = model.module(images, verbose=True)
                # print(accu_per_timestep.shape, count)
                max_T = max(max_T, count)
                # print(max_T)
                if accu_per_timestep.shape[0] < max_T:
                    padding_per_timestep = accu_per_timestep[-1].unsqueeze(0)
                    padding_length = max_T - accu_per_timestep.shape[0]
                    accu_per_timestep = torch.cat(
                        [accu_per_timestep, padding_per_timestep.repeat(padding_length, 1, 1)], dim=0)

                if correct_per_timestep is not None and correct_per_timestep.shape[0] < max_T:
                    for t in range(correct_per_timestep.shape[0], max_T):
                        metric_logger.meters['acc@{}'.format(t + 1)] = deepcopy(metric_logger.meters['acc@{}'.format(correct_per_timestep.shape[0])])

                _, predicted_per_time_step = torch.max(accu_per_timestep.data, 2)
                correct_per_timestep = torch.sum((predicted_per_time_step == target.unsqueeze(0)), dim=1)

                # if correct_per_timestep is None:
                #     _, predicted_per_time_step = torch.max(accu_per_timestep.data, 2)
                #     correct_per_timestep = torch.sum((predicted_per_time_step == target.unsqueeze(0)), dim=1)
                # else:
                #     _, predicted_per_time_step = torch.max(accu_per_timestep.data, 2)
                #     # print(correct_per_timestep.shape, predicted_per_time_step.shape, target.unsqueeze(0).shape)
                #     correct_per_timestep = torch.sum((predicted_per_time_step == target.unsqueeze(0)), dim=1)
            # output = model(images)
            loss = criterion(output, target)

        total_num += images.shape[0]

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # print(output.argmax(-1).reshape(-1))
        # print(target)
        # print("acc1, acc5",acc1, acc5)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        if args.mode == "SNN":
            for t in range(max_T):
                metric_logger.meters['acc@{}'.format(t + 1)].update(
                    correct_per_timestep[t].cpu().item() * 100. / batch_size, n=batch_size)
            model.module.reset()

        # count1 += 1
        # if count1 >= 10:
        #     break
    # gather the stats from all processes
    # accuracy_per_timestep = correct_per_timestep.float().cpu().data / float(total_num)
    print("Evaluation End")
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}