#!/bin/bash
# Leave-One-Out 第一阶段
# -s: 数据路径
# -m: 输出路径
# -r: 分辨率
# --sparse_view_num: 稀疏视角数量
# --sh_degree: 球谐函数阶数
# --init_pcd_name: 初始点云文件名
# --dust3r_json: refined_cams.json 路径
# --white_background: 白色背景
# --random_background: 随机背景
# --use_dust3r: 使用dust3r相机参数

python leave_one_out_stage1.py -s data/realcap/cuc \
    -m output/gs_init/cuc_loo \
    -r 8 --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name dust3r_4 \
    --dust3r_json output/gs_init/cuc/refined_cams.json \
    --white_background --random_background --use_dust3r
    