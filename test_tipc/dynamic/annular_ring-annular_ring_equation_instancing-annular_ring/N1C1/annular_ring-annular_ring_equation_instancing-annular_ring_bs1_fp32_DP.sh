model_item=annular_ring-annular_ring_equation_instancing-annular_ring
bs_item=1
fp_item=fp32
run_mode=DP
device_num=N1C1
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
# copy files
\cp test_tipc/dynamic/annular_ring-annular_ring_equation_instancing-annular_ring/benchmark_common/prepare.sh ./
\cp test_tipc/dynamic/annular_ring-annular_ring_equation_instancing-annular_ring/benchmark_common/run_benchmark.sh ./
\cp test_tipc/dynamic/annular_ring-annular_ring_equation_instancing-annular_ring/N1C1/annular_ring-annular_ring_equation_instancing-annular_ring_bs1_fp32_DP.sh ./
\cp test_tipc/dynamic/annular_ring-annular_ring_equation_instancing-annular_ring/benchmark_common/analysis_log.py ./
# prepare
bash prepare.sh
# run
to_static=False bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} 2>&1;
sleep 10;