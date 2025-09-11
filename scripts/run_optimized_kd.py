#!/usr/bin/env python3
"""
优化的KD知识蒸馏训练脚本
充分利用8个A100 GPU，实施所有优化建议
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


def run_optimized_kd_training(
    # 基础参数
    teacher_model: str,
    student_model: str,
    output_dir: str,
    train_data: str,
    eval_data: str = None,
    experiment_name: str = "optimized_kd_training",
    
    # 🚀 优化的KD训练参数
    kd_epochs: int = 4,  # 增加到4个epoch
    kd_batch_size: int = 4,  # 增大batch size充分利用GPU
    kd_lr: float = 1.5e-4,  # 基础学习率
    kd_grad_accum: int = 32,  # 保持gradient accumulation
    
    # 🌡️ 优化的蒸馏参数
    temperature: float = 8.0,  # 提高温度到8.0
    alpha: float = 0.8,  # KL损失权重
    
    # 📏 扩展序列长度
    max_length: int = 1536,  # 从1024增加到1536
    
    # 📊 评估和监控参数
    eval_steps: int = 250,  # 每250步评估一次
    logging_steps: int = 50,  # 更频繁的日志
    save_steps: int = 500,  # 定期保存checkpoint
    
    # 🔧 学习率调度参数
    warmup_ratio: float = 0.1,  # 10% warmup
    lr_scheduler_type: str = "cosine",  # cosine调度
    
    # 🎯 早停参数
    early_stopping_patience: int = 3,
    
    # wandb参数
    wandb_project: str = "Optimized_KD",
    wandb_tags: list = None
) -> Dict[str, Any]:
    """运行优化的KD训练"""
    
    if wandb_tags is None:
        wandb_tags = ["optimized-hyperparams", "8gpu-utilization", "aiops-data", "temperature-8.0"]
    
    console.print(f"🚀 [bold green]启动优化KD训练[/bold green]: {experiment_name}")
    console.print(f"📁 输出目录: {output_dir}")
    console.print(f"👨‍🏫 Teacher: {teacher_model}")
    console.print(f"🎓 Student: {student_model}")
    
    if WANDB_AVAILABLE and wandb_project:
        wandb.init(
            project=wandb_project,
            name=f"{experiment_name}_{time.strftime('%m%d_%H%M')}",
            tags=wandb_tags,
            config={
                "teacher_model": teacher_model,
                "student_model": student_model,
                "kd_epochs": kd_epochs,
                "kd_batch_size": kd_batch_size,
                "kd_lr": kd_lr,
                "kd_grad_accum": kd_grad_accum,
                "effective_batch_size": kd_batch_size * kd_grad_accum,
                "temperature": temperature,
                "alpha": alpha,
                "max_length": max_length,
                "warmup_ratio": warmup_ratio,
                "lr_scheduler_type": lr_scheduler_type,
                "dataset": "real_aiops_cloud_native",
                "gpu_optimization": "8xA100_full_utilization"
            }
        )
        console.print(f"📊 wandb已初始化: https://wandb.ai/{wandb.run.entity}/{wandb_project}/runs/{wandb.run.id}")
    
    console.print("\n📋 [bold yellow]优化训练配置[/bold yellow]:")
    console.print(f"   🌡️  蒸馏温度: [red]{temperature}[/red] (从2.5优化到8.0)")
    console.print(f"   🔄 训练轮数: [red]{kd_epochs}[/red] epochs (从1增加到4)")
    console.print(f"   📦 批次大小: [red]{kd_batch_size}[/red] (从2增加到4)")
    console.print(f"   📈 有效批次: [red]{kd_batch_size * kd_grad_accum}[/red] (从64增加到128)")
    console.print(f"   🎯 学习率: [red]{kd_lr}[/red] + {lr_scheduler_type}调度")
    console.print(f"   📏 序列长度: [red]{max_length}[/red] (从1024增加到1536)")
    console.print(f"   ⏱️  评估频率: 每[red]{eval_steps}[/red]步")
    console.print(f"   🛡️  早停机制: [red]{early_stopping_patience}[/red]次无改善则停止")
    
    # 打印当前GPU状态
    print_gpu_utilization()
    
    console.print("\n🚀 [bold green]启动优化KD蒸馏流程[/bold green] (充分利用8个A100)...")
    console.print("💡 [yellow]优化重点[/yellow]:")
    console.print("   • 8个GPU全部利用：Teacher分布更均匀")
    console.print("   • 更大批次：提高GPU吞吐量")
    console.print("   • 更高温度：更好的知识传递")
    console.print("   • 多epoch：充分学习")
    console.print("   • 定期评估：防止过拟合")
    
    try:
        result = run_kd_pipeline(
            teacher_model_path=teacher_model,
            student_model_path=student_model,
            train_data_path=train_data,
            eval_data_path=eval_data,
            output_dir=output_dir,
            experiment_name=experiment_name,
            
            # KD训练参数
            kd_epochs=kd_epochs,
            kd_batch_size=kd_batch_size,
            kd_lr=kd_lr,
            kd_grad_accum=kd_grad_accum,
            
            # 蒸馏参数
            temperature=temperature,
            alpha=alpha,
            
            # 序列长度
            max_length=max_length,
            
            # 评估参数
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            
            # 学习率调度
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            
            # 早停
            early_stopping_patience=early_stopping_patience,
            
            # 硬件优化
            use_bf16=True,
            gradient_checkpointing=True,
            dataloader_num_workers=8,  # 更多数据加载线程
            
            # GPU分配优化：更均匀分布Teacher模型
            use_manual_device_map=True,
            teacher_device_map="auto",  # 让系统自动优化Teacher分布
            student_device_map="auto",  # Student也自动分布
            
            # wandb
            use_wandb=WANDB_AVAILABLE and wandb_project is not None,
            wandb_project=wandb_project,
            wandb_tags=wandb_tags
        )
        
        console.print(f"\n✅ [bold green]优化KD训练完成！[/bold green]")
        
        # 显示优化效果总结
        if "execution_time" in result:
            total_time = result["execution_time"]
            console.print(f"⏱️  总训练时间: [green]{total_time:.1f}秒[/green] ({total_time/3600:.1f}小时)")
            
        if "final_loss" in result:
            final_loss = result["final_loss"]
            console.print(f"📉 最终Loss: [green]{final_loss:.4f}[/green]")
            
        if "output_files" in result and "model" in result["output_files"]:
            model_path = result["output_files"]["model"]
            console.print(f"💾 模型保存: [blue]{model_path}[/blue]")
        
        # GPU利用率最终统计
        print_gpu_utilization()
        
        return result
        
    except KeyboardInterrupt:
        console.print("\n⚠️ [yellow]训练被用户中断[/yellow]")
        cleanup_cache()
        return {"status": "interrupted"}
    except Exception as e:
        console.print(f"\n❌ [red]训练失败[/red]: {str(e)}")
        cleanup_cache()
        raise e
    finally:
        if WANDB_AVAILABLE:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(
        description="优化的KD知识蒸馏训练 - 充分利用8个A100 GPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
    parser.add_argument("--teacher_model", required=True, help="教师模型路径或名称")
    parser.add_argument("--student_model", required=True, help="学生模型路径或名称")  
    parser.add_argument("--train_data", required=True, help="训练数据文件路径")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    
    # 可选参数
    parser.add_argument("--eval_data", help="评估数据文件路径")
    parser.add_argument("--experiment_name", default="optimized_kd", help="实验名称")
    
    # 🚀 优化的训练参数
    parser.add_argument("--kd_epochs", type=int, default=4, help="KD训练轮数")
    parser.add_argument("--kd_batch_size", type=int, default=4, help="KD批次大小")
    parser.add_argument("--kd_lr", type=float, default=1.5e-4, help="KD学习率")
    parser.add_argument("--kd_grad_accum", type=int, default=32, help="梯度累积步数")
    
    # 🌡️ 蒸馏参数
    parser.add_argument("--temperature", type=float, default=8.0, help="蒸馏温度")
    parser.add_argument("--alpha", type=float, default=0.8, help="KL损失权重")
    
    # 📏 序列参数
    parser.add_argument("--max_length", type=int, default=1536, help="最大序列长度")
    
    # 📊 评估参数
    parser.add_argument("--eval_steps", type=int, default=250, help="评估步数间隔")
    parser.add_argument("--logging_steps", type=int, default=50, help="日志步数间隔")
    parser.add_argument("--save_steps", type=int, default=500, help="保存步数间隔")
    
    # 🔧 学习率调度
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="预热比例")
    parser.add_argument("--lr_scheduler_type", default="cosine", help="学习率调度器类型")
    
    # 🎯 早停
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="早停耐心值")
    
    # wandb参数
    parser.add_argument("--wandb_project", default="Optimized_KD", help="wandb项目名")
    parser.add_argument("--wandb_tags", nargs="+", 
                       default=["optimized-hyperparams", "8gpu-utilization", "aiops-data"],
                       help="wandb标签")
    
    # 系统参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--log_level", default="INFO", help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志和随机种子
    setup_logging(level=getattr(logging, args.log_level.upper()))
    set_seed(args.seed)
    
    console.print(f"🎯 [bold blue]优化KD训练启动[/bold blue]")
    console.print(f"🔧 随机种子: {args.seed}")
    console.print(f"📋 日志级别: {args.log_level}")
    
    # 验证输入文件
    if not Path(args.train_data).exists():
        console.print(f"❌ [red]训练数据文件不存在[/red]: {args.train_data}")
        return 1
    
    if args.eval_data and not Path(args.eval_data).exists():
        console.print(f"❌ [red]评估数据文件不存在[/red]: {args.eval_data}")
        return 1
    
    try:
        result = run_optimized_kd_training(
            teacher_model=args.teacher_model,
            student_model=args.student_model,
            output_dir=args.output_dir,
            train_data=args.train_data,
            eval_data=args.eval_data,
            experiment_name=args.experiment_name,
            
            kd_epochs=args.kd_epochs,
            kd_batch_size=args.kd_batch_size,
            kd_lr=args.kd_lr,
            kd_grad_accum=args.kd_grad_accum,
            
            temperature=args.temperature,
            alpha=args.alpha,
            
            max_length=args.max_length,
            
            eval_steps=args.eval_steps,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type=args.lr_scheduler_type,
            
            early_stopping_patience=args.early_stopping_patience,
            
            wandb_project=args.wandb_project,
            wandb_tags=args.wandb_tags
        )
        
        console.print(f"\n🎉 [bold green]优化KD训练成功完成！[/bold green]")
        return 0
        
    except KeyboardInterrupt:
        console.print(f"\n⚠️ [yellow]训练被中断[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n❌ [red]训练失败[/red]: {str(e)}")
        logger.exception("训练异常")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)