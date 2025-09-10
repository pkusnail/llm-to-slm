# Qwen3-30B 到 8B 知识蒸馏示例

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

这是一个知识蒸馏的实践示例，展示了如何将 **Qwen3-30B-Instruct** 的领域专业知识转移到 **Qwen3-8B** 模型上。

## 🎯 项目用途说明

### 适用场景：领域特定知识迁移

本项目作为一个**实践例子**，演示如何为专业领域创建特化的AI模型。虽然通用LLM在常见任务上表现良好，但在专业领域场景中可能需要额外的训练。

### 📚 示例领域覆盖

本实现包含了几个专业领域的示例（仅作演示用途）：

**🔧 运维与系统操作 (AIOps)：**
- 系统故障排查（网络延迟、内存泄漏、磁盘使用）
- 日志分析和根因定位
- 配置优化建议

**🏥 医疗健康：**
- 医学术语和临床流程
- 药物相互作用分析
- 病例研究格式化

**💼 金融服务：**
- 风险评估场景
- 法规合规解释
- 金融产品分析

**⚖️ 法律合规：**
- 合同条款解释
- 法律文档总结
- 合规要求分析

**🎓 技术教育：**
- 逐步问题解释
- 代码审查和调试指导
- 概念解释与示例

**🏭 工业制造：**
- 设备维护程序
- 质量控制流程
- 安全协议说明

### 🎯 方法：教师-学生知识迁移

此示例展示了一个**三阶段流程**：

1. **领域数据收集**：收集或创建你的目标领域示例
2. **教师模型生成**：使用大模型创建高质量的训练回答
3. **知识蒸馏**：训练小模型来复制教师的专业知识

**注意**：这里展示的技术在该领域已经较为成熟，本项目仅提供一个带有完整代码和数据的工作示例。

## 🎯 项目概述

本项目演示了端到端知识蒸馏的**三阶段流水线**：

1. **教师数据生成**：使用 Qwen3-30B 生成高质量训练数据
2. **知识蒸馏**：训练 Qwen3-8B 学生模型学习教师知识
3. **评估对比**：通过全面测试验证蒸馏效果

### 主要特点

- ✅ **完整流水线**：从数据生成到模型评估
- ✅ **GPU优化**：高效的多GPU训练和推理
- ✅ **LoRA集成**：内存高效的微调（174.6M参数）
- ✅ **温度验证**：通过推理测试证明KD有效性
- ✅ **可复现**：包含所有训练数据、配置和模型

## 📊 示例结果

此实现展示了以下结果（你的结果可能有所不同）：

### 📈 数据集统计
- **训练数据**：教师模型生成的 3,164 个样本
- **领域覆盖**：AIOps、数学、编程和其他技术领域
- **模型大小**：从 30B 压缩到 8B 参数（73% 压缩率）
- **训练方法**：LoRA 微调，174.6M 可训练参数

### 🖥️ 实验环境配置
**硬件配置：**
- **GPU**：8x NVIDIA A100 40GB
- **系统内存**：128GB+ 
- **GPU 分配策略**： 
  - 教师模型 (Qwen3-30B)：GPU 0-5 (6张GPU)
  - 学生模型 (Qwen3-8B)：GPU 6-7 (2张GPU)

**性能基准测试：**
- **教师数据生成**：约2-3小时 (3,164个样本)
- **知识蒸馏训练**：约1小时 (100步)
- **快速推理测试**：约3分钟 (10个样本)
- **完整评估**：约30分钟 (350个样本)

**显存使用情况：**
- **教师模型**：约120GB VRAM (跨6个GPU)
- **学生模型**：约30GB VRAM (跨2个GPU)  
- **训练峰值**：约180GB 总VRAM使用量

### 🧪 验证方法
- **温度测试**：证明知识迁移的有效性
- **多领域评估**：测试跨不同专业领域的性能
- **对比测试**：提供工具对比原始模型与蒸馏模型

### ⚠️ 局限性和考虑因素
- 结果很大程度上取决于训练数据的质量
- 领域特定的性能需要仔细的数据集策划
- 模型能力受限于原始教师模型的知识
- 部署需要大量 GPU 资源以获得最佳性能

## 🎯 何时此示例可能有用

**✅ 可能有帮助的场景：**
- **学习目的**：理解实践中的知识蒸馏技术
- **研究工作**：探索领域特定的模型压缩方法
- **原型开发**：作为你自己的领域特定模型的起点
- **教育用途**：用真实代码教授知识蒸馏概念
- **实验探索**：测试不同的模型专业化方法

**⚠️ 如果以下情况可能不适合：**
- 你正在寻找现成的商业解决方案
- 你没有ML模型训练和部署的经验
- 你需要生产使用的性能保证
- 你更愿意使用已建立的商业API（GPT-4、Claude等）
- 你无法获得合适的GPU硬件

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA兼容的GPU（推理至少2x 24GB，训练需要8x 40GB）
- 64GB+ 系统内存

### 安装步骤

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd qwen3-knowledge-distillation
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv l2s
   source l2s/bin/activate  # Linux/Mac
   # 或者
   l2s\Scripts\activate     # Windows
   ```

3. **安装依赖**
   ```bash
   pip install torch transformers accelerate peft
   pip install datasets rich tqdm numpy pandas
   ```

4. **准备训练数据集**
   ```bash
   # 生成初始训练数据（GSM8K, HumanEval, AIOps合成数据）
   python scripts/download_datasets.py
   
   # 这将创建：
   # - data/processed/train_dataset.jsonl (~3K样本)
   # - data/processed/eval_dataset.jsonl (~350样本)
   ```

5. **验证安装**
   ```bash
   python -c "from src.distillation.kd import run_kd_pipeline; print('✅ 安装成功')"
   ```

### 数据要求

**训练数据来源：**
- **GSM8K**: 数学推理问题 (2,000个样本)
- **HumanEval**: Python编程挑战 (164个样本)
- **AIOps合成**: 系统故障排查场景 (1,000个样本)

**总数据集：** 约3,164个训练样本 + 350个评估样本 (~6MB)

**注意：** 训练数据文件由于大小限制未包含在仓库中，需要本地生成。请运行上述数据准备脚本来创建它们。

### 快速推理测试

使用我们预训练的检查点立即测试蒸馏模型：

```bash
# 快速测试（10个样本，约3分钟）
./run_kd_inference_test.sh quick

# 中等测试（50个样本，约15分钟）  
./run_kd_inference_test.sh medium

# 完整评估（350个样本，约30分钟）
./run_kd_inference_test.sh full
```

## 📁 项目结构

```
├── src/                           # 核心实现
│   ├── distillation/
│   │   ├── kd.py                 # 知识蒸馏引擎
│   │   ├── sft.py                # 教师数据生成
│   │   └── base.py               # 基础类和工具
│   ├── utils/
│   │   └── common.py             # 通用工具
│   └── evaluation/
│       └── evaluator.py          # 模型评估工具
├── scripts/                       # 训练脚本
│   ├── run_improved_kd.py        # 主要KD训练脚本
│   ├── evaluate_distillation.py  # 模型评估
│   └── fix_sft_data.py           # 数据预处理
├── outputs/experiment/            # 实验结果
│   ├── qwen3_30b_to_8b_ultrabatch_512/sft/
│   │   ├── sft_train_data_clean.jsonl    # 训练数据 (5.2MB)
│   │   └── sft_eval_data_clean.jsonl     # 评估数据 (0.6MB)
│   └── gpu_optimized_kd_20250910_170252/
│       ├── final_model/                   # 蒸馏模型检查点
│       ├── kd_config.json                # 训练配置
│       └── kd_results.json               # 训练指标
├── test_kd_inference_v2.py        # 推理对比工具
├── run_kd_inference_test.sh       # KD推理测试运行脚本
└── learn.md                       # 详细技术文档
```

## 🔬 详细使用说明

### 阶段 1：教师数据生成（已预完成）

教师模型 (Qwen3-30B-Instruct) 已经生成了高质量的训练数据：

**输出**：
- `sft_train_data_clean.jsonl` (3,164 个样本)
- `sft_eval_data_clean.jsonl` (评估集)

### 阶段 2：知识蒸馏训练

使用教师生成的数据训练学生模型：

```bash
python scripts/run_improved_kd.py \
    --learning_rate 2e-5 \
    --temperature 2.5 \
    --alpha 0.8 \
    --max_steps 100 \
    --output_dir outputs/experiment/my_kd_experiment
```

**关键参数**：
- `--learning_rate`：保守的学习率 (2e-5)
- `--temperature`：蒸馏温度 (2.5)
- `--alpha`：KL散度权重 (0.8)
- `--max_steps`：训练步数 (100)

### 阶段 3：评估和对比

比较原始模型与蒸馏模型的性能：

```bash
# 快速对比测试
python test_kd_inference_v2.py --mode quick --temperature 0.7

# 详细评估
python scripts/evaluate_distillation.py \
    --model_path outputs/experiment/gpu_optimized_kd_20250910_170252/final_model \
    --output_dir outputs/evaluation
```

## ⚙️ 配置说明

### 硬件要求

**最低配置（仅推理）**：
- 2x GPU，24GB VRAM
- 32GB 系统内存
- 预期时间：快速测试约10分钟

**推荐配置（完整训练）**：
- 8x GPU，40GB VRAM (A100)
- 128GB 系统内存
- 预期时间：完整流程约4-5小时

**替代配置方案：**
- **4x A100**：降低批次大小，总时长约6-8小时
- **2x A100**：仅限推理和评估
- **V100/RTX 4090**：可行但时间开销较大

### 训练参数

我们成功实验中使用的关键超参数：

```json
{
  "learning_rate": 2e-5,
  "temperature": 2.5,
  "alpha": 0.8,
  "batch_size": 1,
  "gradient_accumulation_steps": 32,
  "max_length": 1024,
  "epochs": 1,
  "warmup_steps": 20,
  "weight_decay": 0.01
}
```

## 🧪 验证方法

### 基于温度的测试

我们的关键验证方法使用温度变化来证明知识迁移：

```python
# 保守温度 - 应该显示最小差异
results_01 = test_model(temperature=0.1)

# 平衡温度 - 应该显示明显差异
results_07 = test_model(temperature=0.7)

# 创造性温度 - 应该显示显著差异
results_12 = test_model(temperature=1.2)
```

## 🔧 故障排除

### 常见问题

**1. CUDA 内存不足**
```bash
# 减少批次大小
python scripts/run_improved_kd.py --batch_size 1 --gradient_accumulation_steps 64
```

**2. 导入错误**
```bash
# 确保在项目根目录
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 激活虚拟环境
source l2s/bin/activate
```

**3. 数据路径问题**
```bash
# 验证数据文件存在
ls -la outputs/experiment/qwen3_30b_to_8b_ultrabatch_512/sft/
```

## 🤝 贡献

欢迎贡献！请查看我们的[贡献指南](CONTRIBUTING.md)了解详细信息。

## 📄 引用

如果你在研究中使用了这个工作，请引用：

```bibtex
@misc{qwen3-knowledge-distillation,
  title={Qwen3-30B to 8B Knowledge Distillation: A Complete Implementation},
  year={2024},
  publisher={GitHub},
  url={https://github.com/your-username/qwen3-knowledge-distillation}
}
```

## 📜 许可证

本项目采用 Apache License 2.0 许可证 - 请查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- **Qwen 团队** 提供出色的基础模型
- **Hugging Face** 提供 transformers 和 PEFT 库
- **PyTorch 团队** 提供深度学习框架
- **社区贡献者** 提供测试和反馈

---

⭐ 如果你觉得这个项目有帮助，请考虑给它一个星标！