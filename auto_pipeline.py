#!/usr/bin/env python
import os
import subprocess
import time
import sys
from datetime import datetime

# 配置参数
datasets = [
    "pana",
    "iui3",
    "japan",
    "cor",
    "kwaj",
    "tokai2"
]

# 基础路径
BASE_DATASET_PATH = "/home/ci21041/final_project/new_dataset_v2"
LOG_FILE = "training_log.txt"
ITERATIONS = 20000

def run_command(cmd):
    """执行命令并返回输出"""
    print(f"执行命令: {cmd}")
    
    # 执行命令，仅返回输出而不实时打印
    process = subprocess.run(
        cmd, 
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False
    )
    
    # 打印错误信息以便调试
    if process.returncode != 0:
        print(f"错误输出: {process.stderr}")
    
    # 返回返回码和输出
    return process.returncode, process.stdout

def extract_output_path(output):
    """从训练输出中提取输出文件夹路径"""
    for line in output.split('\n'):
        if "Output folder:" in line:
            # 提取路径并确保不包含额外的时间戳或其他文本
            path = line.split("Output folder:")[1].strip()
            # 清理路径，删除可能的时间戳或其他非路径内容
            if '[' in path:
                path = path.split('[')[0].strip()
            
            # 从路径中移除可能的 "./" 前缀
            if path.startswith('./'):
                path = path[2:]
            
            return path
    return None

def log_result(dataset, output_path, mode="standard", alpha=None):
    """将数据集、输出路径和模式记录到日志文件"""
    with open(LOG_FILE, 'a') as f:
        if mode == "standard":
            f.write(f"{dataset} {output_path} standard\n")
        else:
            f.write(f"{dataset} {output_path} adaptive alpha={alpha}\n")

def main():
    # 如果日志文件不存在，则创建它；如果存在，则保留其内容
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w') as f:
            pass
    
    # 记录当前工作目录，确保所有命令都在同一目录下执行
    current_dir = os.getcwd()
    
    # 遍历所有数据集
    for dataset in datasets:
        dataset_path = os.path.join(BASE_DATASET_PATH, dataset)
        
        # 检查数据集路径是否存在
        if not os.path.exists(dataset_path):
            print(f"警告: 数据集路径不存在: {dataset_path}")
            continue
            
        print(f"\n正在处理数据集: {dataset}")
        
        # 步骤1: 运行标准训练命令
        train_cmd = f"python train.py -s {dataset_path} --iterations {ITERATIONS} --eval"
        return_code, output = run_command(train_cmd)
        
        if return_code != 0:
            print(f"标准训练失败，返回码: {return_code}")
            continue
        
        # 提取输出文件夹路径
        output_path = extract_output_path(output)
        if not output_path:
            print("无法找到输出文件夹路径，跳过此次训练")
            continue
        
        # 确保输出路径是绝对路径
        if not os.path.isabs(output_path):
            output_path = os.path.join(current_dir, output_path)
        
        # 记录标准模式结果到日志
        log_result(dataset, output_path, "standard")
        
        # 步骤2: 运行渲染
        render_cmd = f"python render.py -m {output_path}"
        return_code, _ = run_command(render_cmd)
        if return_code != 0:
            print(f"渲染失败，返回码: {return_code}")
        
        # 步骤3: 运行度量评估
        metrics_cmd = f"python metrics.py -m {output_path}"
        return_code, _ = run_command(metrics_cmd)
        if return_code != 0:
            print(f"度量评估失败，返回码: {return_code}")
        
        print(f"数据集 {dataset} 的标准模式处理完成")

if __name__ == "__main__":
    try:
        main()
        print("\n所有处理完成")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        sys.exit(1) 