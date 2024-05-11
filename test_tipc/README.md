# Modulus(paddle后端) 动态图 benchmark 模型执行说明

动态图benchmark测试脚本说明

## 1. 目录说明

> [!NOTE]
> 与 pytorch 需要安装 **modulus-sym** 和 **modulus** 两个库不同，paddle 后端的非符号化相关代码暂时未从 models/modulus-sym 分离到 models/modulus，因此只需要克隆 models/modulus-sym 即可。

## 2. Docker 运行环境

``` sh
docker image: registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.8-cudnn8.6-trt8.5-gcc82
paddle = 0.0.0
python >= 3.9
```

## 3. 运行 benchmark 测试步骤

以 `bracket` 模型为例，几种配置的运行命令如下。

``` sh
git clone https://github.com/PaddleBenchmark/modulus-sym.git -b modified_paddle_dy2st
cd modulus-sym

# 设置卡号
export CUDA_VISIBLE_DEVICES=0

# 动态图
bash test_tipc/dynamic/bracket-bracket/N1C1/bracket-bracket_bs1_fp32_DP.sh

# 动转静+PIR
bash test_tipc/dynamicTostatic/bracket-bracket/N1C1/bracket-bracket_bs1_fp32_DP.sh
```

输出如下所示

``` sh
# 动态图
{"model_branch": "modified_paddle_dy2st", "model_commit": "48fd034546a556b067b242a1beae42bad42e3f33", "model_name": "bracket-bracket-bracket_bs1_fp32_DP.sh_bs1_fp32_DP", "batch_size": 1, "fp_item": "fp32", "run_mode": "DP", "convergence_value": 0, "convergence_key": "", "ips": 106.9, "device_num": "N1C1", "model_run_time": "178", "frame_commit": "", "frame_version": "0.0.0", "speed_unit": "ms/iteration"}

# 动转静+PIR
{"model_branch": "modified_paddle_dy2st", "model_commit": "48fd034546a556b067b242a1beae42bad42e3f33", "model_name": "bracket-bracket_bs1_fp32_DP_bs1_fp32_DP", "batch_size": 1, "fp_item": "fp32", "run_mode": "DP", "convergence_value": 0, "convergence_key": "", "ips": 138.47500000000002, "device_num": "N1C1", "model_run_time": "264", "frame_commit": "", "frame_version": "0.0.0", "speed_unit": "ms/iteration"}
```
