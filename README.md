# SpikeZIP-TF: Conversion is All You Need for Transformer-based SNN
![image](https://github.com/Intelligent-Computing-Research-Group/SpikeZIP_transformer/assets/74498528/91609adb-56f2-49e1-92fe-9596b38cb9f4)
This is a PyTorch/GPU implementation of the paper [SpikeZIP-TF: Conversion is All You Need for Transformer-based SNN](https://openreview.net/forum?id=NeotatlYOL)

```
@inproceedings{
anonymous2024spikeziptf,
title={Spike{ZIP}-{TF}: Conversion is All You Need for Transformer-based {SNN}},
author={Anonymous},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=NeotatlYOL}
}
```

## Train the Quantized-ANN with pretrain model:
The following table provides the pre-trained checkpoints used in the paper:

Prepare the ImageNet dataset and run the scripts below:
```
NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port='29500' main_finetune_distill.py \
    --accum_iter 4 \
    --batch_size 64 \
    --model vit_small_patch16 \
    --model_teacher vit_small_patch16 \
    --finetune checkpoint-path \
    --pretrain_teacher checkpoint-path \
    --epochs 100 \
    --blr 1.5e-4 --layer_decay 0.65 --warmup_epochs 0 \
    --weight_decay 0.05 --drop_path 0.1 --drop_rate 0.0 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path dataset-path --output_dir output_path --log_dir log_path \
    --mode "QANN_QAT" --level 16 --act_layer relu --act_layer_teacher relu --temp 2.0 --wandb --print_freq 200 --define_params --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5
```

## Convert your QANN model to SNN
The following table provides the pre-trained QANN used in the paper:

Prepare the ImageNet dataset and run the scripts below:
```

```

