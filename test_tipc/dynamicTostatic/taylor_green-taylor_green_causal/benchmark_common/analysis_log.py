# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import json
import os
from pdb import line_prefix
import re
import traceback

from numpy import mean, var


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filename", type=str, help="The name of log which need to analysis."
    )
    parser.add_argument("--speed_log_file", type=str, help="json file")
    parser.add_argument(
        "--model_name",
        type=str,
        default="model_name",
        help="training model_name, transformer_base",
    )
    parser.add_argument("--base_batch_size", type=int, help="base_batch size on gpu")
    parser.add_argument("--run_mode", type=str, default="DP", help="DP|MP|PP")
    parser.add_argument("--fp_item", type=str, help="fp_item:fp16|fp32")
    parser.add_argument("--keyword", type=str, help="Keyword to specify analysis data")
    parser.add_argument("--loss_keyword", type=str, default="loss:", help="loss Keyword to specify analysis data")
    parser.add_argument(
        "--skip_steps", type=int, default=2, help="The number of steps to be skipped"
    )
    parser.add_argument(
        "--device_num", type=str, default="N1C1", help="device_num:N1C1|N1C8|N4C32"
    )
    args = parser.parse_args()
    return args


def _is_number(num):
    pattern = re.compile(r"^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$")
    result = pattern.match(num)
    if result:
        return True
    else:
        return False


class TimeAnalyzer(object):
    def __init__(self, filename, keyword=None, loss_keyword=None):
        if filename is None:
            raise Exception("Please specify the filename!")

        if keyword is None:
            raise Exception("Please specify the keyword!")

        self.filename = filename
        self.keyword = keyword
        self.loss_keyword = loss_keyword

    def get_iteration_cost(self):
        iteration_costs = []
        loss_value = None
        with open(self.filename, "r") as f_object:
            lines = f_object.read().splitlines()
            for line in lines:
                if self.keyword not in line:
                    continue
                try:
                    result = None

                    # Distill the string from a line.
                    line = line.strip()
                    line_words = line.split()
                    for i in range(len(line_words) - 1):
                        if line_words[i] == self.keyword:
                            result = float(line_words[i + 1])
                            iteration_costs.append(result)
                        if line_words[i] == self.loss_keyword:
                            # 剔除掉该值后面的逗号并保留5位小数点
                            loss_value = line_words[i + 1].replace(',', '')  
                            # 保留5位小数
                            # loss_value = float("{:.5f}".format(float(loss_str_without_comma)))
                            
                    # Distil the result from the picked string.

                except Exception as exc:
                    print("line is: {}; failed".format(line_prefix))
        if loss_value is None:
            loss_value = -1
        return mean(iteration_costs[2:]), loss_value



class CINNMetricsParser(object):  
    def __init__(self, filename):  
        """
        初始化一个类实例。
        
        Args:
            filename (str): 需要处理的文件名。
        
        Raises:
            ValueError: 如果filename为None，则抛出ValueError异常，提示需要指定文件名。
        
        Attributes:
            filename (str): 处理的文件名。
            compiling_program_times (list): 编译程序时间列表。
            compression_ratios (list): 压缩比率列表。
        """
        if filename is None:  
            raise ValueError("Please specify the filename!")  
        self.filename = filename  
        self.compiling_program_times = []  
        self.compression_ratios = []  
  
    def parse_metrics(self):  
        """
        解析文件中的编译时间和压缩比率。
        
        Args:
            无
        
        Returns:
            无
        
        Raises:
            无
        
        """
        # 定义正则表达式来匹配编译时间和压缩比率  
        compiling_time_pattern = r"Time of lowering and compiling program: .*?\[ (\d+) \] .*?seconds\."  
        compression_ratio_pattern = r"compression ratio: \d+/\d+ = ([0-9\.]+)"  
 
  
        with open(self.filename, "r") as f_object:  
            for line in f_object:  
                # 匹配编译时间  
                compiling_time_match = re.search(compiling_time_pattern, line)  
                if compiling_time_match:  
                    self.compiling_program_times.append(int(compiling_time_match.group(1)))  
  
                # 匹配压缩比率  
                compression_ratio_match = re.search(compression_ratio_pattern, line)  
                if compression_ratio_match:  
                    self.compression_ratios.append(float(compression_ratio_match.group(1)))  
        print("compiling_program_times:", self.compiling_program_times, 
            "compression_ratios:", self.compression_ratios)

    def get_average_compiling_program_time(self):
        """
        获取编译程序时间的平均值。
        
        Args:
            无
        
        Returns:
            float: 如果self.compiling_program_times不为空，则返回编译程序时间的平均值；
                 如果为空，则返回None（或者根据您的需求抛出异常）。
        
        """
        if not self.compiling_program_times:  
            return -1  
        return sum(self.compiling_program_times) / len(self.compiling_program_times)  
  
    def get_average_compression_ratio(self):
        """
        获取平均压缩率
        
        Args:
            无
        
        Returns:
            float: 平均压缩率，如果压缩率列表为空则返回 None
        
        """
        if not self.compression_ratios:  
            return -1
        return sum(self.compression_ratios) / len(self.compression_ratios)


if __name__ == "__main__":
    args = parse_args()
    run_info = dict()
    run_info["model_branch"] = os.getenv("model_branch")
    run_info["model_commit"] = os.getenv("model_commit")
    run_info["model_name"] = args.model_name
    run_info["batch_size"] = args.base_batch_size
    run_info["fp_item"] = args.fp_item
    if (
        re.match(r"DP.-MP.-PP.", args.run_mode)
        or "DP_MoE_C" in args.run_mode
        or "Sharding_MoE_C" in args.run_mode
        or re.match(r"DP._MP.", args.run_mode)
    ):
        run_info["run_mode"] = "Collective"
    else:
        run_info["run_mode"] = args.run_mode
    run_info["convergence_value"] = 0
    run_info["convergence_key"] = "loss:"
    run_info["ips"] = 0
    run_info["device_num"] = args.device_num
    run_info["model_run_time"] = os.getenv("model_run_time")
    run_info["frame_commit"] = ""
    run_info["frame_version"] = os.getenv("frame_version")
    device_num = args.device_num
    print("---device_num:-", device_num)
    if "C" in device_num:
        index_c = device_num.index("C")
        print("---index_c:-", index_c)
        gpu_num = int(device_num[index_c + 1 : len(device_num)])
    if "X" in device_num:
        index_c = device_num.index("X")
        print("---index_c:-", index_c)
        gpu_num = 1
    print("-----gpu_num:", gpu_num)
    if "pwgan" in args.model_name:
        print("------analysis ", args.model_name)
        args.keyword = "avg_ms:"

    try:
        analyzer = TimeAnalyzer(args.filename, args.keyword, args.loss_keyword)
        run_info["ips"], run_info["convergence_value"] = analyzer.get_iteration_cost()
        run_info["speed_unit"] = "ms/iteration"
        if 'data_attribute' in os.environ and 'cinn' in os.environ['data_attribute']:
            cinn_metrics_parser = CINNMetricsParser(args.filename)
            cinn_metrics_parser.parse_metrics()
            run_info['avg_cinn_compiling_time'] = cinn_metrics_parser.get_average_compiling_program_time()
            run_info['avg_cinn_compression_ratio'] = cinn_metrics_parser.get_average_compression_ratio()
    except Exception:
        traceback.print_exc()
    print(
        "{}".format(json.dumps(run_info))
    )  # it's required, for the log file path  insert to the database
    with open(args.speed_log_file, "w") as f:
        f.write(json.dumps(run_info))
