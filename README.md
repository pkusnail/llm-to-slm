# 大模型蒸馏实验 (Large-to-Small Model Distillation)

本项目实现了完整的大模型到小模型知识蒸馏实验，支持SFT、KD等多种蒸馏技术，专门针对4-7小时内完成的快速实验而优化。

## 🎯 实验目标

**模型配置**: 30B教师模型 → 8B学生模型 (Qwen3系列)  
**任务覆盖**: 数学推理(GSM8K) + 代码生成(HumanEval) + AIOps日志分析  
**硬件配置**: 2台p4d服务器，每台8×A100 40GB  
**实验时间**: 4-5小时完整流程  
**压缩效果**: 3.75倍参数压缩，预期显著性能提升  

## 🚀 快速开始

### 1. 环境验证

```bash
# 激活虚拟环境
source l2s/bin/activate

# 运行环境检查
python scripts/quick_test.py
```

**预期输出**: 5/5测试通过，环境准备就绪

### 2. 一键运行实验

```bash
# 完整Qwen3蒸馏实验 (约4小时)
python scripts/run_experiment.py \
  --teacher_model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
  --student_model "Qwen/Qwen3-8B" \
  --run_all \
  --experiment_name "qwen3_30b_to_8b_distill"

# 如果要用32B基座模型
python scripts/run_experiment.py \
  --teacher_model "Qwen/Qwen3-32B" \
  --student_model "Qwen/Qwen3-8B" \
  --run_all \
  --experiment_name "qwen3_32b_to_8b_distill"
```

**实验流程**:
1. **Teacher响应生成** (30分钟): 生成3164条高质量训练数据
2. **SFT监督微调** (45分钟): 基础蒸馏训练
3. **KD知识蒸馏** (1.5小时): 分布对齐训练  
4. **三模型评估** (1小时): 对比Teacher、SFT-Student、KD-Student性能

### 3. 查看结果

```bash
# 查看实验配置和结果
cat outputs/qwen_14b_to_7b_distill/results.json

# 查看评估对比报告
cat outputs/qwen_14b_to_7b_distill/evaluation/comparison_report.md

# 查看模型性能CSV
cat outputs/qwen_14b_to_7b_distill/evaluation/model_comparison.csv
```

## 📊 数据集详情

| 领域 | 数据源 | 训练量 | 评估量 | 主要指标 |
|------|--------|--------|--------|----------|
| **数学推理** | GSM8K | 2000条 | 200条 | Exact Match |
| **代码生成** | HumanEval | 164条 | 50条 | Pass@1 |  
| **AIOps分析** | 合成数据 | 1000条 | 100条 | 关键词匹配率 |

**总计**: 训练集3164条，评估集350条

## 🔧 核心技术实现

### 1. SFT监督微调
```python
# 关键特性
- QLoRA 4-bit量化训练
- 梯度检查点节省显存  
- 聊天模板标准化
- 只对回答部分计算损失
```

### 2. KD知识蒸馏
```python
# 损失函数
loss = α * CE_loss + (1-α) * KL_loss * T²
# 默认参数: α=0.5, T=2.0
```

### 3. 并行化策略
```bash
# Teacher推理: 4×GPU张量并行
# Student训练: 4×GPU数据并行 + FSDP
# 评估: 3模型×4GPU同时进行
```

## ⚡ 性能优化

### 时间优化
- **数据预生成**: Teacher生成与Student训练并行
- **批量推理**: 大batch size减少overhead  
- **内存复用**: 及时清理CUDA缓存
- **分布式评估**: 多模型并行评测

### 显存优化  
- **模型量化**: Student用4-bit QLoRA
- **梯度检查点**: 牺牲计算换显存
- **序列截断**: max_length=2048限制
- **批次控制**: 动态调整batch_size

## 🎲 实验参数

### 快速版本 (4小时)
```bash
--sft_epochs 1 --sft_batch_size 4 --sft_lr 2e-4
--kd_epochs 1 --kd_batch_size 2 --kd_lr 1.5e-4  
--temperature 2.0 --alpha 0.5
--max_new_tokens 512 --max_length 2048
```

### 精确版本 (8小时，更好效果)
```bash
--sft_epochs 2 --sft_batch_size 2 --sft_lr 1e-4
--kd_epochs 2 --kd_batch_size 1 --kd_lr 1e-4
--temperature 3.0 --alpha 0.3  
--max_new_tokens 1024 --max_length 4096
```

## 📈 预期实验结果

### 性能提升预期
| 任务 | Student Base | SFT Student | KD Student | 提升幅度 |
|------|--------------|-------------|------------|----------|
| GSM8K准确率 | 45% | 58% (+13%) | 63% (+18%) | **+18%** |
| HumanEval通过率 | 35% | 42% (+7%) | 48% (+13%) | **+13%** |
| AIOps匹配率 | 25% | 45% (+20%) | 52% (+27%) | **+27%** |

### 模型大小对比 (Qwen3系列)
- **Teacher (30B)**: ~31GB显存 (Int8量化)，需1-2×A100
- **Student (8B)**: ~8GB显存，单卡轻松运行  
- **KD Student**: 性能接近Teacher，效率接近Student
- **压缩比**: 3.75倍参数压缩，推理速度提升3-4倍

## 🛠️ 分阶段运行

### 只生成数据
```bash  
python scripts/run_experiment.py \
  --teacher_model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
  --student_model "Qwen/Qwen3-8B" \
  --run_sft --experiment_name "data_gen_only"
# 数据保存在: outputs/data_gen_only/sft/sft_train_data.jsonl
```

### 只运行KD (基于已有SFT)
```bash
python scripts/run_experiment.py \
  --teacher_model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
  --student_model "outputs/previous_sft/final_model" \
  --run_kd --experiment_name "kd_only"  
```

### 只运行评估
```bash
python scripts/run_experiment.py \
  --teacher_model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
  --student_model "outputs/kd_model/final_model" \
  --run_eval --experiment_name "eval_only"
```

## 🔍 结果分析

### 自动生成报告
实验完成后自动生成:
- `evaluation/comparison_report.md`: 详细对比报告
- `evaluation/model_comparison.csv`: 性能数据表  
- `results.json`: 完整实验记录

### 手动分析
```python
import json
import pandas as pd

# 加载结果
with open('outputs/experiment_name/results.json') as f:
    results = json.load(f)

# 分析各领域提升
for domain in ['math', 'code', 'aiops']:
    teacher_perf = results['evaluation']['teacher']['domain_results'][domain]
    student_perf = results['evaluation']['student_kd']['domain_results'][domain]
    print(f"{domain}: {student_perf['main_metric'] - teacher_perf['main_metric']:.1%} improvement")
```

## 🚨 常见问题

### Q: CUDA OOM错误
**A**: 减少batch_size，启用gradient_checkpointing
```bash
--sft_batch_size 2 --kd_batch_size 1 --gradient_accumulation_steps 8
```

### Q: Teacher模型加载失败
**A**: 检查HuggingFace token，使用镜像或本地路径
```bash
export HF_TOKEN="your_token"
# 或使用本地路径
--teacher_model "/path/to/local/model"
```

### Q: 训练收敛慢
**A**: 调整学习率和温度参数
```bash
--sft_lr 5e-4 --kd_lr 2e-4 --temperature 1.5 --alpha 0.7
```

### Q: 评估指标异常
**A**: 检查数据格式和chat template
```python
# 验证数据
python -c "
from src.utils.common import load_jsonl
data = load_jsonl('data/processed/eval_dataset.jsonl')
print('Fields:', data[0].keys())
print('Sample:', data[0])
"
```

## 📋 实验检查清单

**环境准备**:
- [ ] 虚拟环境激活  
- [ ] quick_test.py通过
- [ ] GPU显存充足 (>200GB总计)

**模型准备**:
- [ ] Teacher模型路径确认
- [ ] Student模型路径确认  
- [ ] HuggingFace token配置

**数据验证**:
- [ ] train_dataset.jsonl (3164条)
- [ ] eval_dataset.jsonl (350条)
- [ ] 三领域数据分布正确

**实验配置**:
- [ ] 实验名称设定
- [ ] 输出目录权限正常
- [ ] 预计运行时间确认

**结果验证**:
- [ ] 三模型评估完成
- [ ] 性能提升达到预期
- [ ] 对比报告生成成功

## 📚 技术细节参考

- **Transformer架构**: Qwen2.5系列模型
- **蒸馏理论**: Hinton et al. Knowledge Distillation  
- **训练框架**: HuggingFace Transformers + Accelerate
- **分布式**: PyTorch FSDP + DeepSpeed Zero
- **量化技术**: bitsandbytes QLoRA
- **评估框架**: 自定义多任务评估器

---

🔬 **实验记录**: 本项目专为快速验证大模型蒸馏效果而设计，适合在有限时间内完成完整的蒸馏实验并获得可信的对比结果。