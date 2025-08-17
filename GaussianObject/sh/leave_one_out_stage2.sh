#!/bin/bash
# Leave-One-Out 第二阶段
# 参数同 stage1

python leave_one_out_stage2.py -s data/realcap/cuc \
    -m output/gs_init/cuc_loo \
    -r 8 --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name dust3r_4 \
    --dust3r_json output/gs_init/cuc/refined_cams.json \
    --white_background --random_background --use_dust3r