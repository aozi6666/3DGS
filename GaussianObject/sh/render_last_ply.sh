#!/bin/bash
# --load_ply: 指定要加载的ply文件

python render.py -m output/gs_init/cuc \
    --sparse_view_num 4 \
    --sh_degree 2 \
    --init_pcd_name dust3r_4 \
    --white_background --render_path \
    --use_dust3r \
    --load_ply output/gaussian_object/cuc/save/last.ply