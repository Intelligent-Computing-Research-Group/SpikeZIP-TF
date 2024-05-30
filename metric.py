import torch

def validate_per_time_step(args, model, testloader, criterion, test_batch=10000000):
    if args.rank == 0:
        print("Waiting Test!")
    with torch.no_grad():
        correct = 0
        correct_per_time_step = None
        total = 0
        sum_loss = 0
        model.eval()
        # model.double()
        all_times = []
        last_avg = 0
        cur_avg = 0
        global linear_activation, linear_activation_per_batch, last_name
        linear_activation = []
        linear_activation_per_batch = []
        all_labels = []
        last_name = ""
        if args.rank == 0:
            testloader = tqdm(testloader)
        for i, data in enumerate(testloader):
            if i > test_batch:
                continue
            images, labels = data
            all_labels.append(labels)
            # print(images.shape,labels.shape)
            images, labels = images.cuda(), labels.cuda()
            # images = images.double()
            # h1 = model.module.fc.register_forward_hook(save_linear_activation(i))
            outputs, times, min_T, accu_per_timestep = model(images, verbose=True)

            # h1.remove()

            # make
            if min_T > 0:
                T1, B, N = accu_per_timestep.shape
                zero_preffix = torch.zeros(min_T, B, N, dtype=accu_per_timestep.dtype).to(accu_per_timestep.device)
                accu_per_timestep = torch.cat([zero_preffix, accu_per_timestep], dim=0)

            if outputs is None:
                continue
            all_times.append(times)
            loss = criterion(outputs, labels)
            sum_loss += reduce_value(args, loss, True)
            _, predicted = torch.max(outputs.data, 1)
            _, predicted_per_time_step = torch.max(accu_per_timestep.data, 2)
            total += reduce_value(args, torch.tensor(labels.size(0)).cuda(), False)
            correct += reduce_value(args, (predicted == labels).sum(), False)

            if correct_per_time_step is None:
                correct_per_time_step = torch.sum((predicted_per_time_step == labels.unsqueeze(0)), dim=1)
            else:
                cur_value = torch.sum((predicted_per_time_step == labels.unsqueeze(0)), dim=1)
                max_time = max(correct_per_time_step.shape[0], cur_value.shape[0])
                pad_time1 = max_time - correct_per_time_step.shape[0]
                pad_time2 = max_time - cur_value.shape[0]
                pad_value1 = correct_per_time_step[-1].unsqueeze(0)
                pad_value2 = cur_value[-1].unsqueeze(0)
                for i in range(pad_time1):
                    correct_per_time_step = torch.cat((correct_per_time_step, pad_value1), 0)
                for i in range(pad_time2):
                    cur_value = torch.cat((cur_value, pad_value2), 0)
                correct_per_time_step = correct_per_time_step + cur_value

            reset_model(model)

        length = len(correct_per_time_step)
        for i in range(length):
            correct_per_time_step[i] = reduce_value(args, correct_per_time_step[i], False)

        acc = 100. * correct / total
        acc_per_timestep = 100. * correct_per_time_step / total

    return acc, max(all_times), outputs, acc_per_timestep


@torch.no_grad()
def evaluate(data_loader, model, device, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    if args.mode == "SNN":
        model.max_T = 0
    total_num = 0
    correct_per_timestep = None

    for batch in metric_logger.log_every(data_loader, 10, header):
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
                output, count, accu_per_timestep = model(images, verbose=True)
            # output = model(images)
            loss = criterion(output, target)

        if accu_per_timestep.shape[0] < model.max_T:
            padding_per_timestep = accu_per_timestep[-1].unsqueeze(0)
            padding_length = model.max_T - accu_per_timestep.shape[0]
            accu_per_timestep = torch.cat([accu_per_timestep, padding_per_timestep.repeat(padding_length, 1, 1)], dim=0)

        if correct_per_timestep is not None and correct_per_timestep.shape[0] < model.max_T:
            padding_per_timestep = correct_per_timestep[-1].unsqueeze(0)
            padding_length = model.max_T - correct_per_timestep.shape[0]
            correct_per_timestep = torch.cat([correct_per_timestep, padding_per_timestep.repeat(padding_length, 1, 1)], dim=0)

        if correct_per_timestep is None:
            _, predicted_per_time_step = torch.max(accu_per_timestep.data, 2)
            correct_per_timestep = torch.sum((predicted_per_time_step == target.unsqueeze(0)), dim=1)
        else:
            _, predicted_per_time_step = torch.max(accu_per_timestep.data, 2)
            correct_per_timestep += torch.sum((predicted_per_time_step == target.unsqueeze(0)), dim=1)

        total_num += images.shape[0]

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        for t in model.max_T:
            metric_logger.meters['acc@{}'.format(t+1)].update(correct_per_timestep[t].cpu().item()*100./batch_size, n=batch_size)
        if args.mode == "SNN":
            reset_model(model)
    # gather the stats from all processes
    accuracy_per_timestep = correct_per_timestep.float().cpu().data / float(total_num)
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}