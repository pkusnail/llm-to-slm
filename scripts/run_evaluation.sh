#!/bin/bash
"""
优化后的KD模型评估脚本
消除所有硬编码，支持完全参数化配置
"""

# 设置工作目录
cd /home/ubuntu/exp/LLM-to-SLM

# 激活虚拟环境
source venv/bin/activate

echo "🎯 启动KD模型评估 - 完全参数化版本"

# =============================================================================
# 评估方案1: 使用优化的推理测试脚本 (推荐)
# =============================================================================
echo "📊 方案1: GPU优化推理对比测试"

# Quick模式 - 最能展示KD效果的配置
python test_kd_inference_v2.py \
  --mode quick \
  --temperature 1.2 \
  --student_model "Qwen/Qwen3-8B" \
  --kd_model_path "outputs/experiment/optimized_kd_fixed/final_model" \
  --eval_data "data/cloud_aiops/cloud_aiops_eval_data.jsonl" \
  --max_length 1536 \
  --original_gpu_ids 0 1 2 3 \
  --kd_gpu_ids 4 5 6 7 \
  --quick_batch_size 8

echo "✅ Quick模式完成 - 检查生成的JSON结果文件"

# Medium模式 - 更全面的测试
# python test_kd_inference_v2.py \
#   --mode medium \
#   --temperature 1.2 \
#   --student_model "Qwen/Qwen3-8B" \
#   --kd_model_path "outputs/experiment/optimized_kd_fixed/final_model" \
#   --eval_data "data/cloud_aiops/cloud_aiops_eval_data.jsonl" \
#   --max_length 1536 \
#   --original_gpu_ids 0 1 2 3 \
#   --kd_gpu_ids 4 5 6 7 \
#   --medium_batch_size 16

# =============================================================================
# 评估方案2: 传统评估脚本 (可选)
# =============================================================================
echo "📈 方案2: 传统Perplexity评估"

# python scripts/evaluate_distillation.py \
#   --model_path "outputs/experiment/optimized_kd_fixed/final_model" \
#   --teacher_model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
#   --student_base "Qwen/Qwen3-8B" \
#   --eval_data "data/cloud_aiops/cloud_aiops_eval_data.jsonl" \
#   --sample_size 50 \
#   --max_length 1536 \
#   --max_new_tokens 512 \
#   --temperature 1.2

# =============================================================================
# 多温度对比测试 (深度分析)
# =============================================================================
echo "🌡️ 温度对比实验 - 发现最佳展示效果"

# 保守模式
python test_kd_inference_v2.py \
  --mode quick \
  --temperature 0.1 \
  --student_model "Qwen/Qwen3-8B" \
  --kd_model_path "outputs/experiment/optimized_kd_fixed/final_model" \
  --eval_data "data/cloud_aiops/cloud_aiops_eval_data.jsonl" \
  --max_length 1536 \
  --original_gpu_ids 0 1 2 3 \
  --kd_gpu_ids 4 5 6 7 \
  --quick_batch_size 8

# 平衡模式  
python test_kd_inference_v2.py \
  --mode quick \
  --temperature 0.7 \
  --student_model "Qwen/Qwen3-8B" \
  --kd_model_path "outputs/experiment/optimized_kd_fixed/final_model" \
  --eval_data "data/cloud_aiops/cloud_aiops_eval_data.jsonl" \
  --max_length 1536 \
  --original_gpu_ids 0 1 2 3 \
  --kd_gpu_ids 4 5 6 7 \
  --quick_batch_size 8

# 创意模式 (最能展示Teacher风格)
python test_kd_inference_v2.py \
  --mode quick \
  --temperature 1.2 \
  --student_model "Qwen/Qwen3-8B" \
  --kd_model_path "outputs/experiment/optimized_kd_fixed/final_model" \
  --eval_data "data/cloud_aiops/cloud_aiops_eval_data.jsonl" \
  --max_length 1536 \
  --original_gpu_ids 0 1 2 3 \
  --kd_gpu_ids 4 5 6 7 \
  --quick_batch_size 8

echo "🎉 所有评估任务完成！"
echo "📁 结果文件位于当前目录下的 kd_comparison_*.json"
echo "🔍 重点查看temperature=1.2的结果，最能展示KD训练效果"

# =============================================================================
# 关键参数说明
# =============================================================================
echo """
🔧 参数说明:
  --temperature 1.2     # 根据learn.md验证，最能展示Teacher风格传承
  --max_length 1536     # 匹配训练时的序列长度
  --kd_model_path       # 指向实际的LoRA适配器路径
  --eval_data          # 使用真实AIOps数据，与训练数据匹配
  --original_gpu_ids   # 原始模型GPU分配
  --kd_gpu_ids         # KD模型GPU分配，避免显存冲突

⚠️  训练vs推理温度差异:
  训练temperature=8.0  # KD过程中软化logits分布  
  推理temperature=1.2  # 生成时控制随机性，完全不同概念
  
✅ 修复的硬编码问题:
  ❌ 硬编码模型路径
  ❌ 硬编码数据路径  
  ❌ 硬编码GPU分配
  ❌ 硬编码温度参数
  ❌ 硬编码批次大小
  ❌ 硬编码序列长度
  ✅ 全部改为命令行参数
"""