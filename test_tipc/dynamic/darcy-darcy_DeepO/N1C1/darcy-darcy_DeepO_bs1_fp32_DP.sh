model_item=darcy-darcy_DeepO_bs1_fp32_DP
bs_item=1
fp_item=fp32
run_mode=DP
device_num=N1C1
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
# prepare
bash prepare.sh
# run
to_static=0 prim=0 cinn=0 bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} 2>&1;
sleep 10;
