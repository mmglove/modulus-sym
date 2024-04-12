model_item=waveguide-slab_3D-slab_3D_bs1_fp32_DP_dy2st_pir_prim
bs_item=1
fp_item=fp32
run_mode=DP
device_num=N1C1
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
# prepare
bash prepare.sh
# run
\cp test_tipc/waveguide-slab_3D-slab_3D/benchmark_common/prepare.sh ./
\cp test_tipc/waveguide-slab_3D-slab_3D/benchmark_common/run_benchmark.sh ./
\cp test_tipc/waveguide-slab_3D-slab_3D/N1C1/waveguide-slab_3D-slab_3D_bs1_fp32_DP_dy2st_pir_prim.sh ./
\cp test_tipc/waveguide-slab_3D-slab_3D/benchmark_common/analysis_log.py ./
jit=1 prim=1 cinn=0 FLAGS_enable_pir_in_executor=true FLAGS_enable_pir_api=True FLAGS_cinn_bucket_compile=True FLAGS_group_schedule_tiling_first=1 FLAGS_cinn_new_group_scheduler=1 FLAGS_nvrtc_compile_to_cubin=True bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} 2>&1;
sleep 10;
