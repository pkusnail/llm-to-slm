#!/usr/bin/env python3
"""
🚀 Ultra GPU优化的KD知识蒸馏训练脚本
充分发挥8个A100的GPU算力，实现最大吞吐量

主要优化：
1. 🔥 大幅增加批次大小（充分利用显存）
2. ⚡ 更激进的梯度累积
3. 🎯 优化GPU分配策略
4. 📈 动态学习率调度
5. 💾 显存优化技术
"""
import os
import sys
import time
from pathlib import Path

# 添加src到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from distillation.kd import run_kd_pipeline
from utils.common import setup_logging, set_seed, print_gpu_utilization, cleanup_cache
import logging
from rich.console import Console

# wandb集成
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

console = Console()
logger = logging.getLogger(__name__)

def check_eval_config(dataset_size, eval_steps, epochs, batch_size, grad_accum):
    """简单检查eval_steps是否合理，给出警告和建议"""
    effective_batch = batch_size * grad_accum
    steps_per_epoch = dataset_size // effective_batch
    total_steps = steps_per_epoch * epochs
    
    warnings = []
    
    # 简单规则检查
    if eval_steps >= total_steps:
        suggested = max(50, total_steps // 4)
        warnings.append(f"🚨 eval_steps ({eval_steps}) >= total_steps ({total_steps})! 建议: {suggested}")
    elif dataset_size < 10000 and eval_steps > 100:
        suggested = 100
        warnings.append(f"⚠️ 小数据集 ({dataset_size}) eval_steps太大 ({eval_steps})，建议: {suggested}")
    
    if warnings:
        console.print("[yellow]配置警告:[/yellow]")
        for warning in warnings:
            console.print(f"  {warning}")
        return suggested
    return eval_steps

def run_ultra_optimized_kd():
    """
    🚀 Ultra GPU优化的KD训练
    目标：充分利用8个A100 GPU，达到80%+利用率
    """
    
    # 🔥 Ultra优化参数
    config = {
        # 基础模型
        "teacher_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "student_model": "Qwen/Qwen3-8B", 
        "train_data": "data/cloud_aiops/cloud_aiops_train_data.jsonl",
        "eval_data": "data/cloud_aiops/cloud_aiops_eval_data.jsonl",
        "output_dir": "outputs/experiment/ultra_optimized_kd",
        "experiment_name": "ultra_kd_8xa100_maxutil",
        
        # 🚀 Ultra优化的批次参数 - 充分利用显存
        "kd_batch_size": 12,        # 从4增加到12 (3倍)
        "kd_grad_accum": 16,        # 从32减少到16，总批次保持192
        "effective_batch_size": 192, # 12 * 16 = 192
        
        # 🌡️ 知识蒸馏参数
        "temperature": 8.0,
        "alpha": 0.8,
        
        # 📏 序列长度优化
        "max_length": 1536,         # 保持较长序列
        
        # 🎯 训练轮次
        "kd_epochs": 3,             # 减少到3轮，但效率更高
        
        # 📊 监控频率
        "eval_steps": 500,          # 测试自动修复功能 (故意设太大)
        "logging_steps": 25,        # 更频繁日志
        "save_steps": 400,          # 适中保存频率
        
        # 🔧 学习率优化
        "kd_lr": 2e-4,              # 稍微提高学习率
        "warmup_ratio": 0.05,       # 减少warmup时间
        "lr_scheduler_type": "cosine_with_restarts",
        
        # ⚡ 硬件优化
        "use_bf16": True,
        "gradient_checkpointing": True,
        "dataloader_num_workers": 16,  # 增加数据加载线程
        
        # 💾 显存优化
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 16,
        
        # 🎯 Early stopping
        "early_stopping_patience": 2,  # 更激进的早停
        
        # 🖥️ GPU优化配置
        "use_manual_device_map": True,
        "teacher_device_map": "auto",   # 让系统智能分配
        "student_device_map": "auto",
        
        # 📊 WandB配置
        "wandb_project": "Ultra_Optimized_KD",
        "wandb_tags": [
            "ultra-optimized", 
            "8xa100-maxutil", 
            "batch-12x16", 
            "temp-8.0",
            "3epochs-efficient"
        ]
    }
    
    console.print(f"🚀 [bold green]Ultra GPU优化KD训练[/bold green]: {config['experiment_name']}")
    console.print(f"📁 输出目录: {config['output_dir']}")
    
    # 🔍 智能检查eval_steps配置
    import json
    try:
        with open(config["train_data"]) as f:
            dataset_size = sum(1 for _ in f)
    except:
        dataset_size = 5503  # fallback
    
    suggested_eval_steps = check_eval_config(
        dataset_size, 
        config["eval_steps"], 
        config["kd_epochs"],
        config["kd_batch_size"], 
        config["kd_grad_accum"]
    )
    
    # 如果建议值不同，自动更新
    if suggested_eval_steps != config["eval_steps"]:
        console.print(f"🔧 [yellow]自动修复: eval_steps {config['eval_steps']} → {suggested_eval_steps}[/yellow]")
        config["eval_steps"] = suggested_eval_steps
    
    console.print("\n🔥 [bold red]Ultra优化配置[/bold red]:")
    console.print(f"   📦 批次大小: [red]{config['kd_batch_size']}[/red] (从4增加到12, 3倍提升)")
    console.print(f"   🔄 梯度累积: [red]{config['kd_grad_accum']}[/red] (从32减少到16)")
    console.print(f"   🎯 有效批次: [red]{config['effective_batch_size']}[/red] (保持192)")
    console.print(f"   📈 学习率: [red]{config['kd_lr']}[/red] (提高到2e-4)")
    console.print(f"   🌡️  蒸馏温度: [red]{config['temperature']}[/red]")
    console.print(f"   🏃‍♂️ 数据线程: [red]{config['dataloader_num_workers']}[/red] (增加到16)")
    console.print(f"   ⏱️  评估频率: 每[red]{config['eval_steps']}[/red]步")
    
    # 打印GPU状态
    print_gpu_utilization()
    
    # 初始化WandB
    if WANDB_AVAILABLE:
        wandb.init(
            project=config["wandb_project"],
            name=f"{config['experiment_name']}_{time.strftime('%m%d_%H%M')}",
            tags=config["wandb_tags"],
            config=config
        )
        console.print(f"📊 wandb已初始化: https://wandb.ai/{wandb.run.entity}/{config['wandb_project']}/runs/{wandb.run.id}")
    
    console.print("\n🚀 [bold green]启动Ultra优化KD训练[/bold green]...")
    console.print("💡 [yellow]优化重点[/yellow]:")
    console.print("   • 🔥 批次大小x3：充分利用显存")
    console.print("   • ⚡ 梯度累积优化：减少同步开销")
    console.print("   • 🎯 更高学习率：加速收敛") 
    console.print("   • 💾 显存优化：bf16 + gradient checkpointing")
    console.print("   • 🏃‍♂️ 多线程数据加载：减少IO等待")
    
    try:
        result = run_kd_pipeline(
            teacher_model_path=config["teacher_model"],
            student_model_path=config["student_model"],
            train_data_path=config["train_data"],
            eval_data_path=config["eval_data"],
            output_dir=config["output_dir"],
            experiment_name=config["experiment_name"],
            
            # KD训练参数
            kd_epochs=config["kd_epochs"],
            kd_batch_size=config["kd_batch_size"],
            kd_lr=config["kd_lr"],
            kd_grad_accum=config["kd_grad_accum"],
            
            # 蒸馏参数
            temperature=config["temperature"],
            alpha=config["alpha"],
            
            # 序列长度
            max_length=config["max_length"],
            
            # 评估参数
            eval_steps=config["eval_steps"],
            logging_steps=config["logging_steps"],
            save_steps=config["save_steps"],
            
            # 学习率调度
            warmup_ratio=config["warmup_ratio"],
            lr_scheduler_type=config["lr_scheduler_type"],
            
            # 早停
            early_stopping_patience=config["early_stopping_patience"],
            
            # 硬件优化
            use_bf16=config["use_bf16"],
            gradient_checkpointing=config["gradient_checkpointing"],
            dataloader_num_workers=config["dataloader_num_workers"],
            
            # GPU分配优化
            use_manual_device_map=config["use_manual_device_map"],
            teacher_device_map=config["teacher_device_map"],
            student_device_map=config["student_device_map"],
            
            # WandB
            use_wandb=WANDB_AVAILABLE,
            wandb_project=config["wandb_project"],
            wandb_tags=config["wandb_tags"]
        )
        
        console.print(f"\n✅ [bold green]Ultra优化KD训练完成！[/bold green]")
        
        # 显示结果
        if "execution_time" in result:
            total_time = result["execution_time"]
            console.print(f"⏱️  训练时间: [green]{total_time:.1f}秒[/green] ({total_time/3600:.1f}小时)")
            
        if "final_loss" in result:
            final_loss = result["final_loss"]
            console.print(f"📉 最终Loss: [green]{final_loss:.4f}[/green]")
        
        # GPU利用率统计
        print_gpu_utilization()
        
        return result
        
    except KeyboardInterrupt:
        console.print("\n⚠️ [yellow]训练被中断[/yellow]")
        cleanup_cache()
        return {"status": "interrupted"}
    except Exception as e:
        console.print(f"\n❌ [red]训练失败[/red]: {str(e)}")
        cleanup_cache()
        raise e
    finally:
        if WANDB_AVAILABLE:
            wandb.finish()

if __name__ == "__main__":
    # 设置随机种子
    set_seed(42)
    setup_logging(level=logging.INFO)
    
    console.print("🎯 [bold blue]Ultra GPU优化KD训练启动[/bold blue]")
    
    try:
        result = run_ultra_optimized_kd()
        console.print(f"\n🎉 [bold green]Ultra优化训练成功完成！[/bold green]")
    except Exception as e:
        console.print(f"\n❌ [red]训练失败[/red]: {str(e)}")
        logger.exception("训练异常")
        exit(1)