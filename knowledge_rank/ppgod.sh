#!/bin/bash
#SBATCH -o %j.out # 标准输出重定向至test.out文件
#SBATCH -e %j.out # 标准错误重定向至test.err文件
#SBATCH -J my_model # 作业名指定为test
#SBATCH --nodes=1             # 申请一个节点
#SBATCH --gres=gpu:1		#分配的gpu数量
#SBATCH --nodelist=gpu[06]	#申请GPU的节点
#SBATCH --cpus-per-task=10 # 一个任务需要分配的CPU核心数为5

# 需要执行的指令
/data/huangyulong-slurm/python3/bin/python3  train_rank.py
