#!/bin/bash
# 训练 Gaussian Splatting模型
# -s: 数据路径
# -m: 输出模型路径
# -r: 分辨率
# --sparse_view_num: 稀疏视角数量
# --sh_degree: 球谐函数阶数
# --init_pcd_name: 初始点云文件名
# --white_background: 白色背景
# --random_background: 随机背景
# --use_dust3r: 使用dust3r相机参数

python train_gs.py -s data/realcap/cuc \
    -m output/gs_init/cuc \
    -r 8 --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name dust3r_4 \
    --white_background --random_background --use_dust3r --iterations 20000
    