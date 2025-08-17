#!/bin/bash
# LoRA微调
# --exp_name: 实验名称
# --prompt: 文本提示
# --sh_degree: 球谐函数阶数
# --resolution: 分辨率
# --sparse_num: 稀疏视角数量
# --data_dir: 数据路径
# --gs_dir: Gaussian Splatting模型路径
# --loo_dir: Leave-One-Out路径
# --bg_white: 白色背景
# --sd_locked: 锁定Stable Diffusion
# --train_lora: 训练LoRA
# --use_prompt_list: 使用提示词列表
# --add_diffusion_lora: 添加diffusion LoRA
# --add_control_lora: 添加control LoRA
# --add_clip_lora: 添加clip LoRA
# --use_dust3r: 使用dust3r相机参数

python train_lora.py --exp_name controlnet_finetune/cuc \
    --prompt xxy5syt00 --sh_degree 2 --resolution 8 --sparse_num 4 \
    --data_dir data/realcap/cuc \
    --gs_dir output/gs_init/cuc \
    --loo_dir output/gs_init/cuc_loo \
    --bg_white --sd_locked --train_lora --use_prompt_list \
    --add_diffusion_lora --add_control_lora --add_clip_lora --use_dust3r