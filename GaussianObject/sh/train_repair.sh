#!/bin/bash
# 修复训练
# --config: 配置文件
# --train: 训练模式
# --gpu: GPU编号
# tag: 标签
# system.init_dreamer: dreamer初始化路径
# system.exp_name: 实验输出路径
# system.refresh_size: 刷新大小
# data.data_dir: 数据路径
# data.resolution: 分辨率
# data.sparse_num: 稀疏视角数量
# data.prompt: 文本提示
# data.json_path: refined_cams.json 路径
# system.sh_degree: 球谐函数阶数

python train_repair.py \
    --config configs/gaussian-object-colmap-free.yaml \
    --train --gpu 0 \
    tag="cuc" \
    system.init_dreamer="output/gs_init/cuc" \
    system.exp_name="output/controlnet_finetune/cuc" \
    system.refresh_size=8 \
    data.data_dir="data/realcap/cuc" \
    data.resolution=8 \
    data.sparse_num=4 \
    data.prompt="a photo of a xxy5syt00" \
    data.json_path="output/gs_init/cuc/refined_cams.json" \
    data.refresh_size=8 \
    system.sh_degree=2