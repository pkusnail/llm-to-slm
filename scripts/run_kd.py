#!/usr/bin/env python3
"""
KD知识蒸馏训练脚本
独立运行KD阶段，支持在线和离线蒸馏
"""
import os
import sys
import json
import argparse
from pathlib import Path
import time
from typing import Dict, Any

# 添加src到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from distillation.kd import run_kd_pipeline
from utils.common import setup_logging, set_seed, print_gpu_utilization, cleanup_cache

import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# wandb集成
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ wandb未安装，将跳过实验跟踪。安装命令: pip install wandb")

console = Console()
logger = logging.getLogger(__name__)


def run_kd_training(
    # 基础参数
    teacher_model: str,
    student_model: str,
    output_dir: str,
    train_data: str,
    eval_data: str = None,
    experiment_name: str = "kd_training",
    
    # KD训练参数
    kd_epochs: int = 1,
    kd_batch_size: int = 1,
    kd_lr: float = 1.5e-4,
    kd_grad_accum: int = 16,
    
    # KD蒸馏参数
    temperature: float = 2.0,
    alpha: float = 0.5,
    use_online_kd: bool = True,
    
    # 系统参数
    max_length: int = 2048,
    use_bf16: bool = True,
    gradient_checkpointing: bool = True,
    seed: int = 42,
    
    # 高级选项
    save_steps: int = 1000,
    eval_steps: int = 1000,
    logging_steps: int = 50,
    warmup_ratio: float = 0.1,
    generate_teacher_logits: bool = False,
    logits_batch_size: int = 2,
    
    # wandb参数
    use_wandb: bool = True,
    wandb_project: str = "qwen3-kd-experiments",
    wandb_tags: list = None
):
    """运行KD训练"""
    
    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    setup_logging(str(output_path / "kd_training.log"))
    set_seed(seed)
    
    console.print(f"🧠 开始KD蒸馏训练: {experiment_name}")
    console.print(f"📁 输出目录: {output_path}")
    console.print(f"👨‍🏫 Teacher: {teacher_model}")
    console.print(f"🎓 Student: {student_model}")
    
    # 初始化wandb
    wandb_run = None
    if use_wandb and WANDB_AVAILABLE:
        try:
            wandb_run = wandb.init(
                project=wandb_project,
                name=f"{experiment_name}_{time.strftime('%m%d_%H%M')}",
                tags=wandb_tags or ["knowledge-distillation", "qwen3", "corrected-loss"],
                config={
                    "experiment_name": experiment_name,
                    "teacher_model": teacher_model,
                    "student_model": student_model,
                    "kd_epochs": kd_epochs,
                    "kd_batch_size": kd_batch_size,
                    "kd_lr": kd_lr,
                    "kd_grad_accum": kd_grad_accum,
                    "temperature": temperature,
                    "alpha": alpha,
                    "use_online_kd": use_online_kd,
                    "max_length": max_length,
                    "use_bf16": use_bf16,
                    "gradient_checkpointing": gradient_checkpointing,
                    "loss_fix": "corrected_weights",
                    "formula": f"loss = {1-alpha:.1f}*CE + {alpha:.1f}*KL"
                }
            )
            console.print(f"📊 wandb已初始化: {wandb_run.url}")
        except Exception as e:
            console.print(f"⚠️ wandb初始化失败: {e}，继续训练...")
            wandb_run = None
    
    # 显示配置
    console.print("\n📋 训练配置:")
    console.print(f"   蒸馏模式: {'在线KD' if use_online_kd else '离线KD'}")
    console.print(f"   批次大小: {kd_batch_size}")
    console.print(f"   训练轮数: {kd_epochs}")
    console.print(f"   学习率: {kd_lr}")
    console.print(f"   梯度累积: {kd_grad_accum}")
    console.print(f"   蒸馏温度: {temperature}")
    console.print(f"   损失权重α: {alpha}")
    console.print(f"   序列长度: {max_length}")
    
    # 检查输入数据
    if not Path(train_data).exists():
        console.print(f"❌ 训练数据不存在: {train_data}")
        return False
    
    if eval_data and not Path(eval_data).exists():
        console.print(f"❌ 评估数据不存在: {eval_data}")
        eval_data = None
    
    # 检查学生模型 (允许HuggingFace模型标识符)
    def is_huggingface_model_id(model_path: str) -> bool:
        """检查是否是HuggingFace模型标识符"""
        return "/" in model_path and not Path(model_path).exists()
    
    if not Path(student_model).exists() and not is_huggingface_model_id(student_model):
        console.print(f"❌ 学生模型不存在: {student_model}")
        console.print("💡 请先运行SFT训练或确保学生模型路径正确")
        console.print("💡 或者使用HuggingFace模型标识符，如 'Qwen/Qwen3-8B'")
        return False
    
    # 检查GPU状态
    print_gpu_utilization()
    
    # 保存配置
    config = {
        "experiment_name": experiment_name,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "models": {
            "teacher": teacher_model,
            "student": student_model
        },
        "data": {
            "train": train_data,
            "eval": eval_data
        },
        "training": {
            "epochs": kd_epochs,
            "batch_size": kd_batch_size,
            "learning_rate": kd_lr,
            "gradient_accumulation_steps": kd_grad_accum,
            "max_length": max_length
        },
        "distillation": {
            "temperature": temperature,
            "alpha": alpha,
            "use_online_kd": use_online_kd,
            "generate_teacher_logits": generate_teacher_logits
        },
        "hardware": {
            "use_bf16": use_bf16,
            "gradient_checkpointing": gradient_checkpointing
        },
        "random_seed": seed
    }
    
    with open(output_path / "kd_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    start_time = time.time()
    
    try:
        console.print(f"\n🚀 启动KD蒸馏流程 ({'在线模式' if use_online_kd else '离线模式'})...")
        
        # 训练参数
        training_args = {
            "per_device_train_batch_size": kd_batch_size,
            "num_train_epochs": kd_epochs,
            "learning_rate": kd_lr,
            "gradient_accumulation_steps": kd_grad_accum,
            "save_steps": save_steps,
            "logging_steps": logging_steps,
            "eval_steps": eval_steps,
            "eval_strategy": "steps" if eval_data else "no",
            "warmup_ratio": warmup_ratio,
            "bf16": use_bf16,
            "gradient_checkpointing": gradient_checkpointing,
            "save_strategy": "steps",
            "load_best_model_at_end": True if eval_data else False,
            "metric_for_best_model": "eval_loss" if eval_data else None,
            "report_to": "wandb" if (use_wandb and WANDB_AVAILABLE and wandb_run) else "none"
        }
        
        # KD参数
        kd_args = {
            "temperature": temperature,
            "alpha": alpha,
            "use_online_kd": use_online_kd
        }
        
        # 显存使用提示
        if use_online_kd:
            console.print("⚠️ 在线KD需要同时加载Teacher和Student模型，显存需求较大")
            console.print("💡 如遇OOM，可尝试: --kd_batch_size 1 --kd_grad_accum 32")
        else:
            console.print("💾 离线KD模式，显存需求较小但需要预生成logits")
        
        # 运行KD流程
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]KD蒸馏训练中..."),
            console=console
        ) as progress:
            task = progress.add_task("KD", total=None)
            
            training_history = run_kd_pipeline(
                teacher_model_path=teacher_model,
                student_model_path=student_model,
                train_data_path=train_data,
                output_dir=str(output_path),
                eval_data_path=eval_data,
                use_online_kd=use_online_kd,
                generate_teacher_logits=generate_teacher_logits,
                training_args=training_args,
                kd_args=kd_args,
                max_length=max_length,
                seed=seed,
                logits_batch_size=logits_batch_size
            )
        
        # 保存训练结果
        result = {
            "config": config,
            "training_history": training_history[-10:] if training_history else [],
            "execution_time": time.time() - start_time,
            "output_files": {
                "model": str(output_path / "final_model"),
                "train_data": train_data,
                "eval_data": eval_data
            }
        }
        
        with open(output_path / "kd_results.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 记录最终指标到wandb (但不关闭，在finally中关闭)
        if wandb_run:
            try:
                if training_history:
                    final_logs = training_history[-1]
                    if isinstance(final_logs, dict):
                        wandb_run.log({
                            "final_total_loss": final_logs.get("total_loss", 0),
                            "final_ce_loss": final_logs.get("ce_loss", 0),
                            "final_kl_loss": final_logs.get("kl_loss", 0),
                            "execution_time": result['execution_time']
                        })
            except Exception as e:
                console.print(f"⚠️ wandb记录最终指标时出错: {e}")
        
        console.print(f"\n🎉 KD蒸馏训练完成!")
        console.print(f"⏱️ 训练用时: {result['execution_time']:.1f}秒")
        console.print(f"📊 模型保存到: {output_path / 'final_model'}")
        
        # 下一步建议
        console.print(f"\n💡 下一步建议:")
        console.print(f"   1. 运行评估: python scripts/run_eval.py --model {output_path / 'final_model'}")
        console.print(f"   2. 对比模型: 比较SFT模型和KD模型的性能差异")
        console.print(f"   3. 查看wandb: {wandb_run.url if wandb_run else '未启用wandb'}")
        
        return True
        
    except Exception as e:
        logger.error(f"KD训练失败: {e}")
        console.print(f"❌ KD训练失败: {e}")
        
        # 错误诊断提示
        console.print(f"\n🔧 可能的解决方案:")
        if "CUDA out of memory" in str(e):
            console.print("   - 减少batch_size: --kd_batch_size 1")
            console.print("   - 增加梯度累积: --kd_grad_accum 32")
            console.print("   - 减少序列长度: --max_length 1024")
            console.print("   - 考虑离线KD: --use_online_kd False")
        elif "model" in str(e).lower():
            console.print("   - 检查Teacher模型路径")
            console.print("   - 确认Student模型存在(需先运行SFT)")
            console.print("   - 验证模型格式和兼容性")
        elif "data" in str(e).lower():
            console.print("   - 检查训练数据格式")
            console.print("   - 确认数据路径正确")
            console.print("   - 验证数据包含teacher_response字段")
        
        return False
        
    finally:
        # 确保wandb正确关闭 (无论训练成功还是失败)
        if wandb_run:
            try:
                console.print("🔄 正在关闭wandb...")
                wandb_run.finish()
                console.print("📊 wandb已安全关闭")
            except Exception as e:
                console.print(f"⚠️ wandb关闭时出错: {e}")
        
        # 清理GPU缓存
        try:
            cleanup_cache()
        except Exception as e:
            console.print(f"⚠️ GPU缓存清理出错: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="KD知识蒸馏训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🧠 使用示例:

1. 基础KD训练 (使用SFT模型):
   python scripts/run_kd.py --teacher_model "Qwen/Qwen3-30B-A3B-Instruct-2507" \\
     --student_model "outputs/my_sft/final_model" \\
     --train_data "outputs/my_sft/sft_train_data_clean.jsonl" \\
     --output_dir outputs --experiment_name my_kd

2. 快速验证配置:
   python scripts/run_kd.py --teacher_model "Qwen/Qwen3-30B-A3B-Instruct-2507" \\
     --student_model "outputs/my_sft/final_model" \\
     --train_data "outputs/my_sft/sft_train_data_clean.jsonl" \\
     --preset quick --output_dir outputs --experiment_name quick_kd

3. 离线KD (节省显存):
   python scripts/run_kd.py --teacher_model "Qwen/Qwen3-30B-A3B-Instruct-2507" \\
     --student_model "outputs/my_sft/final_model" \\
     --train_data "outputs/my_sft/sft_train_data_clean.jsonl" \\
     --use_online_kd False --output_dir outputs --experiment_name offline_kd

4. 自定义蒸馏参数:
   python scripts/run_kd.py --teacher_model "Qwen/Qwen3-30B-A3B-Instruct-2507" \\
     --student_model "outputs/my_sft/final_model" \\
     --train_data "outputs/my_sft/sft_train_data_clean.jsonl" \\
     --temperature 3.0 --alpha 0.3 --kd_batch_size 1 \\
     --output_dir outputs --experiment_name custom_kd

📚 详细参数指南请查看: docs/PARAMETER_GUIDE.md

⚠️  注意事项:
   - KD需要先完成SFT训练
   - 在线KD显存需求大，建议batch_size=1
   - 离线KD需要额外时间生成logits但节省显存
        """
    )
    
    # === 基础参数 ===
    basic_group = parser.add_argument_group('基础配置')
    basic_group.add_argument("--teacher_model", type=str, required=True,
                           help="教师模型路径")
    basic_group.add_argument("--student_model", type=str, required=True,
                           help="学生模型路径 (通常是SFT训练后的模型)")
    basic_group.add_argument("--train_data", type=str, required=True,
                           help="训练数据路径 (推荐使用SFT生成的清理数据)")
    basic_group.add_argument("--output_dir", type=str, required=True,
                           help="输出根目录")
    basic_group.add_argument("--experiment_name", type=str, default="kd_training",
                           help="实验名称")
    basic_group.add_argument("--eval_data", type=str,
                           help="评估数据路径")
    
    # === 预设配置 ===
    preset_group = parser.add_argument_group('预设配置')
    preset_group.add_argument("--preset", type=str,
                            choices=['quick', 'standard', 'high_quality'],
                            help="预设配置: quick(快速), standard(标准), high_quality(高质量)")
    
    # === KD训练参数 ===
    kd_group = parser.add_argument_group('KD训练参数')
    kd_group.add_argument("--kd_epochs", type=int, default=1,
                        help="训练轮数 (推荐1-2)")
    kd_group.add_argument("--kd_batch_size", type=int, default=1,
                        help="批次大小 (在线KD推荐1-2)")
    kd_group.add_argument("--kd_lr", type=float, default=1.5e-4,
                        help="学习率 (略低于SFT)")
    kd_group.add_argument("--kd_grad_accum", type=int, default=16,
                        help="梯度累积步数 (补偿小batch_size)")
    
    # === KD蒸馏参数 ===
    distill_group = parser.add_argument_group('蒸馏参数')
    distill_group.add_argument("--temperature", type=float, default=2.0,
                             help="蒸馏温度 (1.5-4.0，越大越平滑)")
    distill_group.add_argument("--alpha", type=float, default=0.5,
                             help="损失权重 (0.3偏重蒸馏，0.7偏重任务)")
    distill_group.add_argument("--use_online_kd", type=bool, default=True,
                             help="使用在线KD (False为离线KD)")
    distill_group.add_argument("--generate_teacher_logits", type=bool, default=False,
                             help="生成Teacher logits (离线KD时使用)")
    distill_group.add_argument("--logits_batch_size", type=int, default=2,
                             help="生成logits的批次大小")
    
    # === 系统参数 ===
    sys_group = parser.add_argument_group('系统参数')
    sys_group.add_argument("--max_length", type=int, default=2048,
                         help="最大序列长度")
    sys_group.add_argument("--use_bf16", type=bool, default=True,
                         help="使用bf16混合精度")
    sys_group.add_argument("--gradient_checkpointing", type=bool, default=True,
                         help="启用梯度检查点")
    sys_group.add_argument("--seed", type=int, default=42,
                         help="随机种子")
    
    # === 高级选项 ===
    advanced_group = parser.add_argument_group('高级选项')
    advanced_group.add_argument("--save_steps", type=int, default=1000,
                              help="保存间隔")
    advanced_group.add_argument("--eval_steps", type=int, default=1000,
                              help="评估间隔")
    advanced_group.add_argument("--logging_steps", type=int, default=50,
                              help="日志间隔")
    advanced_group.add_argument("--warmup_ratio", type=float, default=0.1,
                              help="学习率预热比例")
    
    # === wandb参数 ===
    wandb_group = parser.add_argument_group('实验跟踪')
    wandb_group.add_argument("--use_wandb", type=bool, default=True,
                           help="启用wandb实验跟踪")
    wandb_group.add_argument("--wandb_project", type=str, default="qwen3-kd-experiments",
                           help="wandb项目名称")
    wandb_group.add_argument("--wandb_tags", type=str, nargs="*",
                           help="wandb标签 (多个标签用空格分隔)")
    
    args = parser.parse_args()
    
    # 应用预设配置
    if args.preset:
        if args.preset == 'quick':
            # 快速验证配置
            args.kd_batch_size = 2
            args.kd_grad_accum = 8
            args.max_length = 1024
            console.print("🚀 使用快速验证配置")
        elif args.preset == 'standard':
            # 标准配置
            args.kd_batch_size = 1
            args.kd_grad_accum = 16
            args.max_length = 2048
            console.print("⭐ 使用标准推荐配置")
        elif args.preset == 'high_quality':
            # 高质量配置
            args.kd_epochs = 2
            args.kd_batch_size = 1
            args.kd_grad_accum = 32
            args.max_length = 4096
            args.temperature = 3.0
            args.alpha = 0.3
            args.kd_lr = 1e-4
            console.print("💎 使用高质量配置")
    
    # 运行KD训练 (移除preset参数)
    args_dict = vars(args)
    args_dict.pop('preset', None)  # 移除preset参数
    success = run_kd_training(**args_dict)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())