#!/bin/bash

conda create -n minimind python=3.10 
source ~/.bashrc
conda activate minimind
pip install -r requirements.txt
pip install modelscope

# 拉取模型 
# 方法1：直接拉取huggingface上的模型
# git clone https://huggingface.co/jingyaogong/MiniMind2

# 方法2：使用huggingface的镜像
# git clone https://huggingface.co/jingyaogong/MiniMind2 --mirror
# export HF_ENDPOINT=https://hf-mirror.com

# 方法3：使用git-lfs拉取huggingface上的模型
# curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
# sudo apt-get install git-lfs
# git lfs install
# git clone https://hf-mirror.com/jingyaogong/MiniMind2

# 命令行问答
# load=0: load from pytorch model, load=1: load from transformers-hf model
python eval_model.py --load 1 --model_mode 2



mkdir -p ./dataset
pip install modelscope
# modelscope download --dataset gongjy/minimind_dataset # 默认下载全部到~/.cache/modelscope/datasets/gongjy/minimind_dataset

# 下载数据集到指定目录
# 默认推荐下载pretrain_hq.jsonl + sft_mini_512.jsonl最快速度复现Zero聊天模型
modelscope download --dataset gongjy/minimind_dataset pretrain_hq.jsonl  sft_mini_512.jsonl  --local_dir ./dataset

# 预训练（学知识）
# python train_pretrain.py

# 监督微调（学对话方式）
# python train_full_sft.py

# 模型效果评估

# 下载数据集
mkdir -p ./test
# .pth文件下载
# modelscope download --model gongjy/MiniMind2-PyTorch *.pth --local_dir ./test  
# git clone https://www.modelscope.cn/gongjy/MiniMind2-PyTorch.git

python eval_model.py --model_mode 1 # 默认为0：测试pretrain模型效果，设置为1：测试full_sft模型效果

