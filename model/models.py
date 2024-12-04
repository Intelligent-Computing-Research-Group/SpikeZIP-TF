# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 00:19:30 2020

@author: Jiang Yuxin
"""

import torch
from torch import nn
from transformers import (
    BertForSequenceClassification,
    AlbertForSequenceClassification,
    XLNetForSequenceClassification,
    RobertaForSequenceClassification,
    AutoTokenizer
)


class AlbertModel(nn.Module):
    def __init__(self, args, requires_grad=True):
        super(AlbertModel, self).__init__()
        self.albert = AlbertForSequenceClassification.from_pretrained('albert-xxlarge-v2', num_labels=2)
        self.tokenizer = AutoTokenizer.from_pretrained('albert-xxlarge-v2', do_lower_case=True)
        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        for param in self.albert.parameters():
            param.requires_grad = True  # Each parameter requires gradient

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.albert(input_ids=batch_seqs, attention_mask=batch_seq_masks,
                                   token_type_ids=batch_seq_segments, labels=labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities


class BertModel(nn.Module):
    def __init__(self, args, requires_grad=True):
        super(BertModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('textattack/bert-base-uncased-SST-2', num_labels=2)
        self.tokenizer = AutoTokenizer.from_pretrained('textattack/bert-base-uncased-SST-2', do_lower_case=True)
        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        for param in self.bert.parameters():
            param.requires_grad = requires_grad  # Each parameter requires gradient

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.bert(input_ids=batch_seqs, attention_mask=batch_seq_masks,
                                 token_type_ids=batch_seq_segments, labels=labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities


class RobertModel(nn.Module):
    def __init__(self, args, requires_grad=True):
        super(RobertModel, self).__init__()
        self.bert = RobertaForSequenceClassification.from_pretrained(args.pretrained, num_labels=args.num_labels)
        # if args.num_labels != self.bert.classifier.out_proj.out_features:
        #     dense_in_feature = self.bert.classifier.dense.in_feature
        #     dense_out_feature = self.bert.classifier.dense.out_feature
        #     # classifier_dropout = self.bert.classifier.dropout
        #     out_proj_in_feature = self.bert.classifier.out_proj.in_feature
        #     # out_proj_out_feature = self.bert.classifier.out_proj.out_feature
        #     self.bert.classifier.dense = nn.Linear(dense_in_feature, dense_out_feature)
        #     self.bert.classifier.out_proj = nn.Linear(out_proj_in_feature, args.num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained, do_lower_case=True)
        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        for param in self.bert.parameters():
            param.requires_grad = requires_grad  # Each parameter requires gradient

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.bert(input_ids=batch_seqs, attention_mask=batch_seq_masks,
                                 token_type_ids=batch_seq_segments, labels=labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities

class XlnetModel(nn.Module):
    def __init__(self, args, requires_grad=True):
        super(XlnetModel, self).__init__()
        self.xlnet = XLNetForSequenceClassification.from_pretrained('xlnet-large-cased', num_labels=2)
        self.tokenizer = AutoTokenizer.from_pretrained('xlnet-large-cased', do_lower_case=True)
        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        for param in self.xlnet.parameters():
            param.requires_grad = requires_grad  # Each parameter requires gradient

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.xlnet(input_ids=batch_seqs, attention_mask=batch_seq_masks,
                                  token_type_ids=batch_seq_segments, labels=labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities
