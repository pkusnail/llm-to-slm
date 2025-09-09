#!/usr/bin/env python3
"""
SFT监督微调训练脚本
独立运行SFT阶段，支持数据生成和模型训练
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

from distillation.sft import run_sft_pipeline
from utils.common import setup_logging, set_seed, print_gpu_utilization, cleanup_cache

import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
logger = logging.getLogger(__name__)


def run_sft_training(
    # 基础参数
    teacher_model: str,
    student_model: str,
    output_dir: str,
    train_data: str = "data/processed/train_dataset.jsonl",
    eval_data: str = "data/processed/eval_dataset.jsonl",
    experiment_name: str = "sft_training",
    
    # SFT训练参数
    sft_epochs: int = 1,
    sft_batch_size: int = 2,
    sft_lr: float = 2e-4,
    sft_grad_accum: int = 8,
    
    # 数据生成参数
    generate_data: bool = True,
    max_new_tokens: int = 256,
    gen_temperature: float = 0.3,
    gen_top_p: float = 0.9,
    
    # 系统参数
    max_length: int = 2048,
    use_bf16: bool = True,
    gradient_checkpointing: bool = True,
    seed: int = 42,
    
    # 高级选项
    use_lora: bool = True,
    save_steps: int = 1000,
    eval_steps: int = 1000,
    logging_steps: int = 50,
    warmup_ratio: float = 0.1
):
    """运行SFT训练"""
    
    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    setup_logging(str(output_path / "sft_training.log"))
    set_seed(seed)
    
    console.print(f"🎯 开始SFT训练: {experiment_name}")
    console.print(f"📁 输出目录: {output_path}")
    console.print(f"🎓 Teacher: {teacher_model}")
    console.print(f"🎓 Student: {student_model}")
    
    # 显示配置
    console.print("\n📋 训练配置:")
    console.print(f"   批次大小: {sft_batch_size}")
    console.print(f"   训练轮数: {sft_epochs}")
    console.print(f"   学习率: {sft_lr}")
    console.print(f"   梯度累积: {sft_grad_accum}")
    console.print(f"   序列长度: {max_length}")
    console.print(f"   生成长度: {max_new_tokens}")
    
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
            "epochs": sft_epochs,
            "batch_size": sft_batch_size,
            "learning_rate": sft_lr,
            "gradient_accumulation_steps": sft_grad_accum,
            "max_length": max_length,
            "use_lora": use_lora
        },
        "generation": {
            "max_new_tokens": max_new_tokens,
            "temperature": gen_temperature,
            "top_p": gen_top_p
        },
        "hardware": {
            "use_bf16": use_bf16,
            "gradient_checkpointing": gradient_checkpointing
        },
        "random_seed": seed
    }
    
    with open(output_path / "sft_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    start_time = time.time()
    
    try:
        console.print("\n🚀 启动SFT训练流程...")
        
        # 训练参数
        training_args = {
            "per_device_train_batch_size": sft_batch_size,
            "num_train_epochs": sft_epochs,
            "learning_rate": sft_lr,
            "gradient_accumulation_steps": sft_grad_accum,
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
            "report_to": "none"
        }
        
        # 生成参数
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": gen_temperature,
            "top_p": gen_top_p,
            "do_sample": True
        }
        
        # 运行SFT流程
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]SFT训练中..."),
            console=console
        ) as progress:
            task = progress.add_task("SFT", total=None)
            
            training_history = run_sft_pipeline(
                teacher_model_path=teacher_model,
                student_model_path=student_model,
                train_data_path=train_data,
                output_dir=str(output_path),
                generate_teacher_data=generate_data,
                eval_data_path=eval_data,
                training_args=training_args,
                generation_kwargs=generation_kwargs,
                max_length=max_length,
                seed=seed
            )
        
        # 如果生成了新数据，清理数据格式
        if generate_data:
            console.print("🔧 清理数据格式...")
            try:
                from scripts.optimize_sft_data import optimize_sft_data_structure
                
                # 清理训练数据
                if (output_path / "sft_train_data.jsonl").exists():
                    optimize_sft_data_structure(
                        str(output_path / "sft_train_data.jsonl"),
                        str(output_path / "sft_train_data_clean.jsonl")
                    )
                
                # 清理评估数据
                if (output_path / "sft_eval_data.jsonl").exists():
                    optimize_sft_data_structure(
                        str(output_path / "sft_eval_data.jsonl"),
                        str(output_path / "sft_eval_data_clean.jsonl")
                    )
                console.print("✅ 数据格式清理完成")
            except Exception as e:
                console.print(f"⚠️ 数据清理失败: {e}")
        
        # 保存训练结果
        result = {
            "config": config,
            "training_history": training_history[-10:] if training_history else [],
            "execution_time": time.time() - start_time,
            "output_files": {
                "model": str(output_path / "final_model"),
                "train_data": str(output_path / "sft_train_data_clean.jsonl") if generate_data else train_data,
                "eval_data": str(output_path / "sft_eval_data_clean.jsonl") if generate_data and eval_data else eval_data
            }
        }
        
        with open(output_path / "sft_results.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        cleanup_cache()
        
        console.print(f"\n🎉 SFT训练完成!")
        console.print(f"⏱️ 训练用时: {result['execution_time']:.1f}秒")
        console.print(f"📊 模型保存到: {output_path / 'final_model'}")
        
        if generate_data:
            console.print(f"📋 清理后的训练数据: {output_path / 'sft_train_data_clean.jsonl'}")
            if eval_data:
                console.print(f"📋 清理后的评估数据: {output_path / 'sft_eval_data_clean.jsonl'}")
        
        return True
        
    except Exception as e:
        logger.error(f"SFT训练失败: {e}")
        console.print(f"❌ SFT训练失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="SFT监督微调训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 使用示例:

1. 基础SFT训练:
   python scripts/run_sft.py --teacher_model "Qwen/Qwen3-30B-A3B-Instruct-2507" \\
     --student_model "Qwen/Qwen3-8B" --output_dir outputs --experiment_name my_sft

2. 快速验证配置:
   python scripts/run_sft.py --teacher_model "Qwen/Qwen3-30B-A3B-Instruct-2507" \\
     --student_model "Qwen/Qwen3-8B" --output_dir outputs \\
     --preset quick --experiment_name quick_sft

3. 高质量训练配置:
   python scripts/run_sft.py --teacher_model "Qwen/Qwen3-30B-A3B-Instruct-2507" \\
     --student_model "Qwen/Qwen3-8B" --output_dir outputs \\
     --preset high_quality --experiment_name high_quality_sft

4. 使用现有数据直接SFT (跳过Teacher生成):
   python scripts/run_sft.py --teacher_model "Qwen/Qwen3-30B-A3B-Instruct-2507" \\
     --student_model "Qwen/Qwen3-8B" --output_dir outputs \\
     --no_generate_data --experiment_name direct_sft

5. 自定义参数:
   python scripts/run_sft.py --teacher_model "Qwen/Qwen3-30B-A3B-Instruct-2507" \\
     --student_model "Qwen/Qwen3-8B" --output_dir outputs \\
     --sft_batch_size 1 --max_length 1024 --experiment_name custom_sft

📚 详细参数指南请查看: docs/PARAMETER_GUIDE.md
        """
    )
    
    # === 基础参数 ===
    basic_group = parser.add_argument_group('基础配置')
    basic_group.add_argument("--teacher_model", type=str, required=True,
                           help="教师模型路径")
    basic_group.add_argument("--student_model", type=str, required=True,
                           help="学生模型路径")
    basic_group.add_argument("--output_dir", type=str, required=True,
                           help="输出根目录")
    basic_group.add_argument("--experiment_name", type=str, default="sft_training",
                           help="实验名称")
    basic_group.add_argument("--train_data", type=str,
                           default="data/processed/train_dataset.jsonl",
                           help="训练数据路径")
    basic_group.add_argument("--eval_data", type=str,
                           default="data/processed/eval_dataset.jsonl",
                           help="评估数据路径")
    
    # === 预设配置 ===
    preset_group = parser.add_argument_group('预设配置')
    preset_group.add_argument("--preset", type=str, 
                            choices=['quick', 'standard', 'high_quality'],
                            help="预设配置: quick(快速), standard(标准), high_quality(高质量)")
    
    # === SFT参数 ===
    sft_group = parser.add_argument_group('SFT训练参数')
    sft_group.add_argument("--sft_epochs", type=int, default=1,
                         help="训练轮数 (推荐1-2)")
    sft_group.add_argument("--sft_batch_size", type=int, default=2,
                         help="批次大小 (推荐1-4，显存不足时用1)")
    sft_group.add_argument("--sft_lr", type=float, default=2e-4,
                         help="学习率 (推荐1e-4到3e-4)")
    sft_group.add_argument("--sft_grad_accum", type=int, default=8,
                         help="梯度累积步数")
    
    # === 数据生成参数 ===
    data_group = parser.add_argument_group('数据生成参数')
    data_group.add_argument("--generate_data", action='store_true', default=True,
                          help="生成Teacher数据 (默认开启)")
    data_group.add_argument("--no_generate_data", action='store_true', 
                          help="跳过Teacher数据生成，直接使用现有数据训练")
    data_group.add_argument("--max_new_tokens", type=int, default=256,
                          help="Teacher生成的最大长度")
    data_group.add_argument("--gen_temperature", type=float, default=0.3,
                          help="生成温度 (0.1确定，0.7多样)")
    data_group.add_argument("--gen_top_p", type=float, default=0.9,
                          help="nucleus采样参数")
    
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
    advanced_group.add_argument("--use_lora", type=bool, default=True,
                              help="使用LoRA微调")
    advanced_group.add_argument("--save_steps", type=int, default=1000,
                              help="保存间隔")
    advanced_group.add_argument("--eval_steps", type=int, default=1000,
                              help="评估间隔")
    advanced_group.add_argument("--logging_steps", type=int, default=50,
                              help="日志间隔")
    advanced_group.add_argument("--warmup_ratio", type=float, default=0.1,
                              help="学习率预热比例")
    
    args = parser.parse_args()
    
    # 处理互斥的generate_data选项
    if args.no_generate_data:
        args.generate_data = False
    
    # 应用预设配置
    if args.preset:
        if args.preset == 'quick':
            # 快速验证配置
            args.sft_batch_size = 4
            args.max_length = 1024
            args.max_new_tokens = 256
            args.sft_grad_accum = 4
            console.print("🚀 使用快速验证配置")
        elif args.preset == 'standard':
            # 标准配置
            args.sft_batch_size = 2
            args.max_length = 2048
            args.max_new_tokens = 512
            console.print("⭐ 使用标准推荐配置")
        elif args.preset == 'high_quality':
            # 高质量配置
            args.sft_epochs = 2
            args.sft_batch_size = 1
            args.max_length = 4096
            args.max_new_tokens = 1024
            args.sft_lr = 1e-4
            console.print("💎 使用高质量配置")
    
    # 检查必要文件
    if not Path(args.train_data).exists():
        console.print(f"❌ 训练数据不存在: {args.train_data}")
        return 1
    
    if args.eval_data and not Path(args.eval_data).exists():
        console.print(f"❌ 评估数据不存在: {args.eval_data}")
        return 1
    
    # 运行SFT训练 (移除preset参数，因为已经应用到具体参数中)
    success = run_sft_training(
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        output_dir=args.output_dir,
        train_data=args.train_data,
        eval_data=args.eval_data,
        experiment_name=args.experiment_name,
        sft_epochs=args.sft_epochs,
        sft_batch_size=args.sft_batch_size,
        sft_lr=args.sft_lr,
        sft_grad_accum=args.sft_grad_accum,
        generate_data=args.generate_data,
        max_new_tokens=args.max_new_tokens,
        gen_temperature=args.gen_temperature,
        gen_top_p=args.gen_top_p,
        max_length=args.max_length,
        use_bf16=args.use_bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        use_lora=args.use_lora,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())