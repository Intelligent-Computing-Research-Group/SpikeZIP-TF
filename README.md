<p align="center">
<img src="spikezip_logo.png" alt="spikezip_logo" width="220" align="center">
</p>

<div align="center"><h1>&nbsp;NLU code for SpikeZIP-TF: Conversion is All You Need for Transformer-based SNN</h1></div>


<p align="center">
| <a href="http://arxiv.org/"><b>Paper</b></a> | <a href="http://arxiv.org/"><b>Blog</b></a> |
</p>


<p align="center">
  <a href="https://opensource.org/license/mulanpsl-2-0">
    <img src="https://img.shields.io/badge/License-MuLan_PSL_2.0-blue.svg" alt="License">
  </a>
  <a href="https://github.com/">
    <img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" alt="Maintenance">
  </a>
  <a href="https://github.com/">
    <img src="https://img.shields.io/badge/Contributions-welcome-brightgreen.svg?style=flat" alt="Contributions welcome">
  </a>
</p>


## Contents
- [News](#news)
- [Introduction](#introduction)
- [Usage](#Usage)
  - [Train](#Train)
  - [Conversion](#Conversion)
  - [Evaluation](#Evaluation) 

## News

- [2024/6] Code of SpikeZip-TF is released!
- [2024/12] Code of SpikeZip-TF in NLU tasks is released! You can view the code by switching to the NLU_tasks branch !! 

## Introduction

![image](https://github.com/Intelligent-Computing-Research-Group/SpikeZIP_transformer/assets/74498528/91609adb-56f2-49e1-92fe-9596b38cb9f4)


This is the official project repository for the following paper. If you find this repository helpful, Please kindly cite:
```
@inproceedings{
spikeziptf2024,
title={SpikeZIP-TF: Conversion is All You Need for Transformer-based SNN},
author={You, Kang= and Xu, Zekai= and Nie, Chen and Deng, Zhijie and Wang, Xiang and Guo, Qinghai and He, Zhezhi},
booktitle={Forty-first International Conference on Machine Learning (ICML)},
year={2024}
}
```
:fire: =: indicates equal contribution.

## Usage

### Train the ReLU-based ANN in a single GPU
1. Prepare the NLU datasets
2. Download the GeLU-based ANN from following table
3. Modify the **datapath_dict** variable to your dataset path.
4. Run the following Commands
```
python main.py --mode ANN --pretrained <GeLU-based ANN Path> --dataset <Target dataset>
```
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">roberta-gelu-pretrained-mr</th>
<th valign="bottom">roberta-gelu-pretrained-sst2</th>
<th valign="bottom">roberta-gelu-pretrained-sst5</th>
<th valign="bottom">roberta-gelu-pretrained-subj</th>
<!-- TABLE BODY -->
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://huggingface.co/XianYiyk/roberta-gelu-pretrained-mr">download</a></td>
<td align="center"><a href="https://huggingface.co/XianYiyk/roberta-gelu-pretrained-sst2">download</a></td>
<td align="center"><a href="https://huggingface.co/XianYiyk/roberta-gelu-pretrained-sst5">download</a></td>
<td align="center"><a href="https://huggingface.co/XianYiyk/roberta-gelu-pretrained-subj">download</a></td>
</tr>
<tr><td align="left">md5</td>
<td align="center"><tt>0057f30</tt></td>
<td align="center"><tt>c64123b</tt></td>
<td align="center"><tt>dd80da8</tt></td>
<td align="center"><tt>2e29274</tt></td>
</tr>
</tbody></table>



### Train the Quantized-ANN with ReLU-based ANN.
1. Prepare the NLU datasets
2. Download the ReLU-based ANN from following table
3. Modify the **datapath_dict** variable to your dataset path.
4. Run the following Commands
```
python main.py --mode QANN-QAT --level 8 --pretrained <ReLU-based ANN Path> --dataset <Target dataset>
```

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">roberta-relu-pretrained-mr</th>
<th valign="bottom">roberta-relu-pretrained-sst2</th>
<th valign="bottom">roberta-relu-pretrained-sst5</th>
<th valign="bottom">roberta-relu-pretrained-subj</th>
<!-- TABLE BODY -->
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://huggingface.co/XianYiyk/roberta-relu-pretrained-mr">download</a></td>
<td align="center"><a href="https://huggingface.co/XianYiyk/roberta-relu-pretrained-sst2">download</a></td>
<td align="center"><a href="https://huggingface.co/XianYiyk/roberta-relu-pretrained-sst5">download</a></td>
<td align="center"><a href="https://huggingface.co/XianYiyk/roberta-relu-pretrained-subj">download</a></td>
</tr>
<tr><td align="left">md5</td>
<td align="center"><tt>50eb781</tt></td>
<td align="center"><tt>1b4c67c</tt></td>
<td align="center"><tt>288663d</tt></td>
<td align="center"><tt>c1859e7</tt></td>
</tr>
</tbody></table>

### Conversion
1. Prepare the NLU datasets
2. Modify the **datapath_dict** variable to your dataset path.
3. Run the following Commands
```
python main.py --mode SNN --level 8 --qann_pretrained <QANN Model Path> --epochs -1
```

