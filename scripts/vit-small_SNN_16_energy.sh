CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port='29501' main_finetune.py \
    --accum_iter 4 \
    --batch_size 4 \
    --model vit_small_patch16 \
    --finetune /data1/vit-small-imagenet-relu-q16-80.45.pth \
    --resume /data1/vit-small-imagenet-relu-q16-80.45.pth \
    --epochs 100 \
    --blr 3.536e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --drop_rate 0.0 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path /data1/ImageNet/ --output_dir /home/kang_you/SpikeZIP_transformer/output/ --log_dir /home/kang_you/SpikeZIP_transformer/output \
    --mode "SNN" --act_layer relu --eval --energy_eval --time_step 32 --encoding_type rate --level 16 --weight_quantization_bit 32 --define_params --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5