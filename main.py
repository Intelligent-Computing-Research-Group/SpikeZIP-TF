# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 00:19:30 2020

@author: Jiang Yuxin
"""
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from sys import platform

from data.load_data import DataPrecessForSentence
from utils.utils import train, validate, test
from utils.MR_trainer import train_transformer
from model.models import RobertModel
import argparse
from spike_quan_wrapper import myquan_replace,SNNWrapper
from collections import OrderedDict

datapath_dict = {"sst2": "/home/kang_you/SpikeZIP_bert/NLU/SST-2",
                 "sst5": "/home/kang_you/SpikeZIP_bert/NLU/SST-5",
                 "subj": "/home/kang_you/SpikeZIP_bert/NLU/Subj",
                 "mr": "/home/kang_you/SpikeZIP_bert/NLU/MR",}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dataset", default="sst2", type=str)
    parser.add_argument("--k1", default="label", type=str)
    parser.add_argument("--k2", default="text", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument('--if_save_model', action='store_false', default=True,
                        help='if save the trained model to the target dir.')
    parser.add_argument("--num_labels", default=2, type=int)
    parser.add_argument("--depths", default=6, type=int)
    parser.add_argument("--max_seq_len", default=50, type=int)
    parser.add_argument("--dim", default=768, type=int)
    parser.add_argument("--tau", default=10.0, type=float)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--max_grad_norm", default=10, type=int)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--qann_pretrained", default=None, type=str)
    parser.add_argument("--pretrained", default="/home/kang_you/SpikeZIP_bert/pretrained/roberta_sst2", type=str)
    # training mode
    parser.add_argument('--mode', default="ANN", type=str,
                        help='the running mode of the script["ANN", "QANN_PTQ", "QANN_QAT", "SNN"]')

    # LSQ quantization
    parser.add_argument('--level', default=32, type=int,
                        help='the quantization levels')
    parser.add_argument('--neuron_type', default="ST-BIF", type=str,
                        help='neuron type["ST-BIF", "IF"]')
    args = parser.parse_args()
    return args

def prepare_dataset(args):
    data_path = datapath_dict[args.dataset]
    if args.dataset == "sst2":
        train_df = pd.read_csv(os.path.join(data_path, "train.tsv"), sep='\t', header=None, names=[args.k1, args.k2])
        dev_df = pd.read_csv(os.path.join(data_path, "dev.tsv"), sep='\t', header=None, names=[args.k1, args.k2])
        test_df = pd.read_csv(os.path.join(data_path, "test.tsv"), sep='\t', header=None, names=[args.k1, args.k2])
    elif args.dataset == "mr":
        train_df = pd.read_csv(os.path.join(data_path, "train.csv"), names=[args.k2, args.k1])
        dev_df = pd.read_csv(os.path.join(data_path, "test.csv"), names=[args.k2, args.k1])
        test_df = pd.read_csv(os.path.join(data_path, "test.csv"), names=[args.k2, args.k1])
    elif args.dataset == "subj":
        train_df = pd.read_json(os.path.join(data_path, "train.jsonl"), lines=True)
        dev_df = pd.read_json(os.path.join(data_path, "test.jsonl"), lines=True)
        test_df = pd.read_json(os.path.join(data_path, "test.jsonl"), lines=True)
    else:
        train_df = pd.read_json(os.path.join(data_path, "train.jsonl"), lines=True)
        dev_df = pd.read_json(os.path.join(data_path, "dev.jsonl"), lines=True)
        test_df = pd.read_json(os.path.join(data_path, "test.jsonl"), lines=True)
    target_dir = "output/Roberta-{}-{}".format(args.dataset, args.mode)
    if args.mode == "QANN-QAT":
        target_dir += "-{}".format(args.level)
    elif args.mode == "SNN":
        target_dir += "-{}-SNN".format(args.level)
    args.target_dir = target_dir
    os.makedirs(target_dir, exist_ok=True)

    return train_df, dev_df, test_df, target_dir

def model_train_validate_test(args):
    """
    Parameters
    ----------
    train_df : pandas dataframe of train set.
    dev_df : pandas dataframe of dev set.
    test_df : pandas dataframe of test set.
    target_dir : the path where you want to save model.
    max_seq_len: the max truncated length.
    epochs : the default is 3.
    batch_size : the default is 32.
    lr : learning rate, the default is 2e-05.
    patience : the default is 1.
    max_grad_norm : the default is 10.0.
    if_save_model: if save the trained model to the target dir.
    checkpoint : the default is None.

    """
    max_seq_len = args.max_seq_len
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    patience = args.patience
    max_grad_norm = args.max_grad_norm
    if_save_model = args.if_save_model
    checkpoint = args.checkpoint

    train_df, dev_df, test_df, target_dir = prepare_dataset(args)

    bertmodel = RobertModel(args=args, requires_grad=True)
    tokenizer = bertmodel.tokenizer
    

    print(20 * "=", " Preparing for training ", 20 * "=")
    # Path to save the model, create a folder if not exist.
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    f = open(f"{target_dir}/ann_model_arch.txt","w")
    f.write(str(bertmodel))
    f.close()
    
    if args.mode == "QANN-QAT" or args.mode == "SNN":
        myquan_replace(bertmodel,args.level)
        if args.qann_pretrained is not None:
            qann_pre_model = torch.load(args.qann_pretrained)
            new_state_dict = OrderedDict()
            for k, v in qann_pre_model.items():
                name = "bert." + k
                new_state_dict[name] = v            
            msg = bertmodel.load_state_dict(new_state_dict)
            print(msg)
        f = open(f"{target_dir}/qann_model_arch.txt","w")
        f.write(str(bertmodel))
        f.close()
        
    
    if args.mode == "SNN":
        bertmodel = SNNWrapper(ann_model=bertmodel,cfg=None,time_step=200,Encoding_type="analog",level=args.level,model_name="roberta",neuron_type="ST=BIF")
        f = open(f"{target_dir}/snn_model_arch.txt","w")
        f.write(str(bertmodel))
        f.close()
        
    
    # -------------------- Data loading --------------------------------------#

    print("\t* Loading training data...")
    train_data = DataPrecessForSentence(tokenizer, train_df, max_seq_len=max_seq_len)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    print("\t* Loading validation data...")
    dev_data = DataPrecessForSentence(tokenizer, dev_df, max_seq_len=max_seq_len)
    dev_loader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)

    print("\t* Loading test data...")
    test_data = DataPrecessForSentence(tokenizer, test_df, max_seq_len=max_seq_len)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    # -------------------- Model definition ------------------- --------------#

    print("\t* Building model...")
    device = torch.device("cuda")
    model = bertmodel.to(device)
    if args.mode == "SNN":
        model.device = device

    # -------------------- Preparation for training  -------------------------#

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    ## Implement of warm up
    ## total_steps = len(train_loader) * epochs
    ## scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=60, num_training_steps=total_steps)

    # When the monitored value is not improving, the network performance could be improved by reducing the learning rate.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.85, patience=0)

    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    epochs_count = []
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    valid_aucs = []

    # Continuing training from a checkpoint if one was given as argument
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        print("\t* Training will continue on existing model from epoch {}...".format(start_epoch))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        train_accuracy = checkpoint["train_accuracy"]
        valid_losses = checkpoint["valid_losses"]
        valid_accuracy = checkpoint["valid_accuracy"]
        valid_auc = checkpoint["valid_auc"]
    # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, valid_accuracy, _, = validate(model, dev_loader, args)
    print("\n* Validation loss before training: {:.4f}, accuracy: {:.4f}%".format(valid_loss,
                                                                                               (valid_accuracy * 100)))
    _, test_loss, test_accuracy, _, = validate(model, test_loader, args)
    print("\n* Test loss before training: {:.4f}, accuracy: {:.4f}%".format(test_loss,
                                                                                  (test_accuracy * 100)))

    # -------------------- Training epochs -----------------------------------#

    print("\n", 20 * "=", "Training bert model on device: {}".format(device), 20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)

        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer, epoch, max_grad_norm)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%".format(epoch_time, epoch_loss,
                                                                                   (epoch_accuracy * 100)))

        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy, _, = validate(model, dev_loader, args)
        valid_losses.append(epoch_loss)
        valid_accuracies.append(epoch_accuracy)
        # valid_aucs.append(epoch_auc)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))

        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)
        ## scheduler.step()

        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            if (if_save_model):
                torch.save({"epoch": epoch,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "best_score": best_score,
                            "epochs_count": epochs_count,
                            "train_losses": train_losses,
                            "train_accuracy": train_accuracies,
                            "valid_losses": valid_losses,
                            "valid_accuracy": valid_accuracies
                            },
                           os.path.join(target_dir, "best.pth.tar"))
                new_dict = {}
                for k, v in model.state_dict().items():
                    if "bert." in k:
                        new_dict[k.replace("bert.", "")] = v
                torch.save(new_dict, os.path.join(target_dir, "pytorch_model.bin"))
                print("save model succesfully!\n")

            # run model on test set and save the prediction result to csv
            print("* Test for epoch {}:".format(epoch))
            _, _, test_accuracy, all_prob = validate(model, test_loader, args)
            print("Test accuracy: {:.4f}%\n".format(test_accuracy * 100.))
            test_prediction = pd.DataFrame({'prob_1': all_prob})
            test_prediction['prob_0'] = 1 - test_prediction['prob_1']
            test_prediction['prediction'] = test_prediction.apply(lambda x: 0 if (x['prob_0'] > x['prob_1']) else 1,
                                                                  axis=1)
            test_prediction = test_prediction[['prob_0', 'prob_1', 'prediction']]
            test_prediction.to_csv(os.path.join(target_dir, "test_prediction.csv"), index=False)

        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break


if __name__ == "__main__":
    args = parse_args()
    if args.dataset != "mr":
        model_train_validate_test(args)
    else:
        _, _, _, target_dir = prepare_dataset(args)
        train_transformer(args.pretrained, "MR", target_dir, num_train_epochs=args.epochs)