# ADRec 复现说明

## 模型简介

ADRec 是一种基于扩散模型的序列推荐算法，发表于 KDD'2025 会议。该模型通过创新的自回归策略和 token-level 独立扩散机制显著提升了序列推荐的性能。

## 复现特点

本实现在原始 ReChorus 框架的基础上：

- 完整实现了 ADRec 的核心算法逻辑
- 保留了 ReChorus 框架的模块化特性
- 支持完整的三阶段训练策略
- 兼容 ReChorus 的各种训练和评估功能

## 文件结构

```bash
ADRec/                          # 原始论文作者源代码  
log/                            # 日志文件  
model/                          # 模型文件
Rechorus/                       # ReChorus 框架
├── logs/
│   └── models/
│       └── xxx_results.txt     # 训练日志 
├── src/
│   ├── main.py                 # 主程序入口
│   └── models/
│       └── sequential/
│           └── ADRec.py        # ADRec 模型实现（复现版本）
│           └── DiffuRec.py     # DiffuRec 模型实现（复现版本）,用于对比的扩散模型基线
│           └── DreamRec.py     # DreamRec 模型实现（复现版本）,用于对比的扩散模型基线
├── exp.sh                      # 实验运行脚本
README.md                   # 本说明文件
```

## ADRec 核心创新点

### 1. Token-Level 独立扩散

- 每个序列元素独立进行扩散过程
- 充分利用训练样本，避免样本稀疏问题

### 2. 三阶段训练策略

- Stage 1: 预训练嵌入层（Epoch 0-10）
- Stage 2: 骨干网络预热（Epoch 11-15）
- Stage 3: 联合训练（Epoch 16+）

### 3. 创新推理策略

- 仅对最后一个 token 添加噪声
- 历史 token 保持清洁，提供更好指导信息

## 使用方法

### 基本训练命令

```bash
python src/main.py --model_name ADRec [参数...]
```

### 推荐配置示例

```bash
# 完整版 ADRec（推荐配置）
python src/main.py --model_name ADRec \
  --emb_size 128 --hidden_size 128 \
  --diffusion_steps 50 --num_blocks 4 \
  --dropout 0.3 --lr 5e-4 --l2 1e-5 \
  --diffusion_loss_weight 0.7 --ce_loss_weight 0.3 \
  --cfg_scale 1.2 --epoch 100 \
  --use_partial_generation 1
```

### 消融实验配置

```bash
# 去除三阶段训练
python src/main.py --model_name ADRec --training_stage stage3

# 去除 token-level 独立扩散
python src/main.py --model_name ADRec --independent_diffusion 0
```

### 关键参数说明

| 参数                      | 默认值 | 说明                           |
| ------------------------- | ------ | ------------------------------ |
| `--emb_size`              | 64     | 嵌入向量维度                   |
| `--hidden_size`           | 64     | 隐层向量维度                   |
| `--diffusion_steps`       | 50     | 扩散步数                       |
| `--num_blocks`            | 2      | Transformer 块数量             |
| `--cfg_scale`             | 1.0    | 分类器无关指导系数             |
| `--training_stage`        | stage1 | 训练阶段                       |
| `--independent_diffusion` | True   | 是否使用 token-level 独立扩散  |

### 原始论文引用

```bash
bibtex
@inproceedings{chen2025unlocking,
  title={Unlocking the Power of Diffusion Models in Sequential Recommendation: A Simple and Effective Approach},
  author={Chen, Jialei and Xu, Yuanbo and Jiang, Yiheng},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2025}
}
```

### ReChorus 框架引用

```bash
bibtex
@inproceedings{li2024rechorus2,
  title={ReChorus2. 0: A Modular and Task-Flexible Recommendation Library},
  author={Li, Jiayu and Li, Hanyu and He, Zhiyu and Ma, Weizhi and Sun, Peijie and Zhang, Min and Ma, Shaoping},
  booktitle={Proceedings of the 18th ACM Conference on Recommender Systems},
  pages={454--464},
  year={2024}
}
```

### 参考仓库

复现工作主要参考了以下仓库：
[ADRec 参考仓库](https://github.com/Nemo-1024/ADRec);
[ReChorus 参考仓库](https://github.com/THUwangcy/Rechorus);
[DiffuRec 参考仓库](https://github.com/WHUIR/DiffuRec);
[DreamRec 参考仓库](https://github.com/YangZhengyi98/DreamRec)
