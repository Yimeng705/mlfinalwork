#!/bin/bash

# 创建输出目录
mkdir -p logs/results

# 输出文件
OUTPUT_FILE="logs/results/experiment_results.txt"

# 清空输出文件
> $OUTPUT_FILE

echo "开始运行所有实验..." | tee -a $OUTPUT_FILE
echo "========================" | tee -a $OUTPUT_FILE

# 1. 模型对比实验 - Grocery_and_Gourmet_Food 数据集
echo "1. 模型对比实验 - Grocery_and_Gourmet_Food 数据集" | tee -a $OUTPUT_FILE
echo "-------------------------------------------------" | tee -a $OUTPUT_FILE

echo "运行 SASRec..." | tee -a $OUTPUT_FILE
python src/main.py --model_name SASRec --emb_size 128 --hidden_size 128 --num_layers 2 --num_heads 4 --dropout 0.3 --lr 5e-4 --l2 1e-5 --epoch 100 --dataset Grocery_and_Gourmet_Food 2>&1 | tee -a $OUTPUT_FILE

echo "运行 GRU4Rec..." | tee -a $OUTPUT_FILE
python src/main.py --model_name GRU4Rec --emb_size 128 --hidden_size 128 --num_layers 2 --num_heads 4 --dropout 0.3 --lr 5e-4 --l2 1e-5 --epoch 100 --dataset Grocery_and_Gourmet_Food 2>&1 | tee -a $OUTPUT_FILE

echo "运行 DreamRec..." | tee -a $OUTPUT_FILE
python src/main.py --model_name DreamRec --emb_size 128 --hidden_size 128 --diffusion_steps 50 --dropout 0.3 --lr 5e-4 --l2 1e-5 --diffusion_loss_weight 0.7 --ce_loss_weight 0.3 --epoch 100 --dataset Grocery_and_Gourmet_Food 2>&1 | tee -a $OUTPUT_FILE

echo "运行 DiffuRec..." | tee -a $OUTPUT_FILE
python src/main.py --model_name DiffuRec --emb_size 128 --hidden_size 128 --diffusion_steps 50 --dropout 0.3 --lr 5e-4 --l2 1e-5 --diffusion_loss_weight 0.7 --ce_loss_weight 0.3 --epoch 100 --dataset Grocery_and_Gourmet_Food 2>&1 | tee -a $OUTPUT_FILE

echo "运行 ADRec..." | tee -a $OUTPUT_FILE
python src/main.py --model_name ADRec --early_stop 100 --emb_size 128 --hidden_size 128 --diffusion_steps 50 --num_blocks 4 --num_heads 4 --dropout 0.3 --lr 5e-4 --l2 1e-5 --diffusion_loss_weight 0.7 --ce_loss_weight 0.3 --cfg_scale 1.2 --epoch 100 --use_partial_generation 1 --dataset Grocery_and_Gourmet_Food 2>&1 | tee -a $OUTPUT_FILE

# 1. 模型对比实验 - MovieLens_1M 数据集
echo "" | tee -a $OUTPUT_FILE
echo "1. 模型对比实验 - MovieLens_1M 数据集" | tee -a $OUTPUT_FILE
echo "-------------------------------------" | tee -a $OUTPUT_FILE

echo "运行 SASRec..." | tee -a $OUTPUT_FILE
python src/main.py --model_name SASRec --emb_size 128 --hidden_size 128 --num_layers 2 --num_heads 4 --dropout 0.3 --lr 5e-4 --l2 1e-5 --epoch 100 --dataset MovieLens_1M 2>&1 | tee -a $OUTPUT_FILE

echo "运行 GRU4Rec..." | tee -a $OUTPUT_FILE
python src/main.py --model_name GRU4Rec --emb_size 128 --hidden_size 128 --num_layers 2 --num_heads 4 --dropout 0.3 --lr 5e-4 --l2 1e-5 --epoch 100 --dataset MovieLens_1M 2>&1 | tee -a $OUTPUT_FILE

echo "运行 DreamRec..." | tee -a $OUTPUT_FILE
python src/main.py --model_name DreamRec --emb_size 128 --hidden_size 128 --diffusion_steps 50 --dropout 0.3 --lr 5e-4 --l2 1e-5 --diffusion_loss_weight 0.7 --ce_loss_weight 0.3 --epoch 100 --dataset MovieLens_1M 2>&1 | tee -a $OUTPUT_FILE

echo "运行 DiffuRec..." | tee -a $OUTPUT_FILE
python src/main.py --model_name DiffuRec --emb_size 128 --hidden_size 128 --diffusion_steps 50 --dropout 0.3 --lr 5e-4 --l2 1e-5 --diffusion_loss_weight 0.7 --ce_loss_weight 0.3 --epoch 100 --dataset MovieLens_1M 2>&1 | tee -a $OUTPUT_FILE

echo "运行 ADRec..." | tee -a $OUTPUT_FILE
python src/main.py --model_name ADRec --early_stop 100 --emb_size 128 --hidden_size 128 --diffusion_steps 50 --num_blocks 4 --num_heads 4 --dropout 0.3 --lr 5e-4 --l2 1e-5 --diffusion_loss_weight 0.7 --ce_loss_weight 0.3 --cfg_scale 1.2 --epoch 100 --use_partial_generation 1 --dataset MovieLens_1M 2>&1 | tee -a $OUTPUT_FILE

# 2. 扩散模型对比实验
echo "" | tee -a $OUTPUT_FILE
echo "2. 扩散模型对比实验" | tee -a $OUTPUT_FILE
echo "-------------------" | tee -a $OUTPUT_FILE

echo "运行 DreamRec..." | tee -a $OUTPUT_FILE
python src/main.py --model_name DreamRec --emb_size 128 --hidden_size 128 --diffusion_steps 50 --dropout 0.3 --lr 5e-4 --l2 1e-5 --diffusion_loss_weight 0.7 --ce_loss_weight 0.3 --epoch 100 2>&1 | tee -a $OUTPUT_FILE

echo "运行 DiffuRec..." | tee -a $OUTPUT_FILE
python src/main.py --model_name DiffuRec --emb_size 128 --hidden_size 128 --diffusion_steps 50 --dropout 0.3 --lr 5e-4 --l2 1e-5 --diffusion_loss_weight 0.7 --ce_loss_weight 0.3 --epoch 100 2>&1 | tee -a $OUTPUT_FILE

echo "运行 ADRec..." | tee -a $OUTPUT_FILE
python src/main.py --model_name ADRec --early_stop 30 --emb_size 128 --hidden_size 128 --diffusion_steps 50 --dropout 0.3 --lr 5e-4 --l2 1e-5 --diffusion_loss_weight 0.7 --ce_loss_weight 0.3 --epoch 100 2>&1 | tee -a $OUTPUT_FILE

# 3. 消融实验
echo "" | tee -a $OUTPUT_FILE
echo "3. 消融实验" | tee -a $OUTPUT_FILE
echo "------------" | tee -a $OUTPUT_FILE

echo "运行 ADRec (完整版)..." | tee -a $OUTPUT_FILE
python src/main.py --model_name ADRec --early_stop 30 --emb_size 32 --hidden_size 32 --diffusion_steps 10 --num_blocks 4 --num_heads 4 --dropout 0.3 --lr 5e-4 --l2 1e-5 --diffusion_loss_weight 0.7 --ce_loss_weight 0.3 --cfg_scale 1.2 --epoch 100 --use_partial_generation 1 2>&1 | tee -a $OUTPUT_FILE

echo "运行 ADRec (去掉三阶段训练)..." | tee -a $OUTPUT_FILE
python src/main.py --model_name ADRec --early_stop 30 --emb_size 32 --hidden_size 32 --diffusion_steps 10 --num_blocks 4 --num_heads 4 --dropout 0.3 --lr 5e-4 --l2 1e-5 --diffusion_loss_weight 0.7 --ce_loss_weight 0.3 --cfg_scale 1.2 --epoch 100 --use_partial_generation 1 --training_stage stage3 2>&1 | tee -a $OUTPUT_FILE

echo "运行 ADRec (去掉token-level diffusion)..." | tee -a $OUTPUT_FILE
python src/main.py --model_name ADRec --early_stop 30 --emb_size 32 --hidden_size 32 --diffusion_steps 10 --num_blocks 4 --num_heads 4 --dropout 0.3 --lr 5e-4 --l2 1e-5 --diffusion_loss_weight 0.7 --ce_loss_weight 0.3 --cfg_scale 1.2 --epoch 100 --use_partial_generation 1 --independent_diffusion 0 2>&1 | tee -a $OUTPUT_FILE
# 4. 超参数实验
echo "" | tee -a $OUTPUT_FILE
echo "4. 超参数实验" | tee -a $OUTPUT_FILE
echo "--------------" | tee -a $OUTPUT_FILE

echo "运行 ADRec (T=5)..." | tee -a $OUTPUT_FILE
python src/main.py --model_name ADRec --early_stop 30 --emb_size 128 --hidden_size 128 --diffusion_steps 5 --num_blocks 4 --num_heads 4 --dropout 0.3 --lr 5e-4 --l2 1e-5 --diffusion_loss_weight 0.7 --ce_loss_weight 0.3 --cfg_scale 1.2 --epoch 100 --use_partial_generation 1 2>&1 | tee -a $OUTPUT_FILE

echo "运行 ADRec (T=10)..." | tee -a $OUTPUT_FILE
python src/main.py --model_name ADRec --early_stop 30 --emb_size 128 --hidden_size 128 --diffusion_steps 10 --num_blocks 4 --num_heads 4 --dropout 0.3 --lr 5e-4 --l2 1e-5 --diffusion_loss_weight 0.7 --ce_loss_weight 0.3 --cfg_scale 1.2 --epoch 100 --use_partial_generation 1 2>&1 | tee -a $OUTPUT_FILE

echo "运行 ADRec (T=20)..." | tee -a $OUTPUT_FILE
python src/main.py --model_name ADRec --early_stop 30 --emb_size 128 --hidden_size 128 --diffusion_steps 20 --num_blocks 4 --num_heads 4 --dropout 0.3 --lr 5e-4 --l2 1e-5 --diffusion_loss_weight 0.7 --ce_loss_weight 0.3 --cfg_scale 1.2 --epoch 100 --use_partial_generation 1 2>&1 | tee -a $OUTPUT_FILE

echo "运行 ADRec (T=50)..." | tee -a $OUTPUT_FILE
python src/main.py --model_name ADRec --early_stop 100 --emb_size 128 --hidden_size 128 --diffusion_steps 50 --num_blocks 4 --num_heads 4 --dropout 0.3 --lr 5e-4 --l2 1e-5 --diffusion_loss_weight 0.7 --ce_loss_weight 0.3 --cfg_scale 1.2 --epoch 100 --use_partial_generation 1 2>&1 | tee -a $OUTPUT_FILE

echo "" | tee -a $OUTPUT_FILE
echo "所有实验已完成！结果已保存到 $OUTPUT_FILE" | tee -a $OUTPUT_FILE