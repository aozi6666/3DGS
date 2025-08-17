#!/bin/bash
# 渲染结果
# -m: 模型路径
# --sparse_view_num: 稀疏视角数量
# --sh_degree: 球谐函数阶数
# --init_pcd_name: 初始点云文件名
# --dust3r_json: refined_cams.json 路径
# --white_background: 白色背景
# --render_path: 渲染路径
# --use_dust3r: 使用dust3r相机参数

python render.py \
    -m output/gs_init/cuc \
    --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name dust3r_4 \
    --dust3r_json output/gs_init/cuc/refined_cams.json \
    --white_background --render_path --use_dust3r