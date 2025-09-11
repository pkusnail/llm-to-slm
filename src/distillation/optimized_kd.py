"""
优化的知识蒸馏实现 - 充分利用8个A100 GPU
改进GPU分配策略，提高吞吐量和训练效率
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from accelerate import Accelerator
import numpy as np

from distillation.base import BaseDistiller, DistillationDataset, collate_fn, KnowledgeDistillationLoss
from utils.common import load_jsonl, setup_logging, set_seed, cleanup_cache

logger = logging.getLogger(__name__)


class OptimizedKDDistiller(BaseDistiller):
    """优化的知识蒸馏器 - 充分利用8个A100"""
    
    def __init__(
        self,
        teacher_model_path: str,
        student_model_path: str,
        output_dir: str,
        temperature: float = 8.0,  # 默认高温度
        alpha: float = 0.8,
        use_online_kd: bool = True,
        
        # GPU优化参数
        enable_8gpu_optimization: bool = True,
        teacher_gpus: Optional[List[int]] = None,  # 可自定义Teacher GPU
        student_gpus: Optional[List[int]] = None,  # 可自定义Student GPU
        
        **kwargs
    ):
        super().__init__(teacher_model_path, student_model_path, output_dir, **kwargs)
        
        self.temperature = temperature
        self.alpha = alpha
        self.use_online_kd = use_online_kd
        self.enable_8gpu_optimization = enable_8gpu_optimization
        
        # GPU分配策略
        gpu_count = torch.cuda.device_count()
        logger.info(f"检测到 {gpu_count} 个GPU")
        
        if enable_8gpu_optimization and gpu_count >= 8:
            # 🚀 优化策略：充分利用8个GPU
            self.teacher_device_map, self.student_device_map = self._create_optimized_8gpu_mapping()
        elif gpu_count >= 6:
            # 备用策略：6-7个GPU
            self.teacher_device_map, self.student_device_map = self._create_balanced_mapping(gpu_count)
        else:
            # 最小配置：使用auto
            self.teacher_device_map = "auto"
            self.student_device_map = "auto"
            
        # 加载模型
        self.load_tokenizer()
        self.load_student_model(student_device_map=self.student_device_map)
        
        if use_online_kd:
            self.load_teacher_model(teacher_device_map=self.teacher_device_map)
        
        # 损失函数
        self.kd_loss_fn = KnowledgeDistillationLoss(temperature, alpha)
    
    def _create_optimized_8gpu_mapping(self) -> Tuple[Dict, Dict]:
        """
        创建优化的8GPU分配策略
        
        策略：
        - Teacher (30B, 48层): GPU 0-4 (每个GPU 9-10层，更均匀分布)
        - Student (8B, 36层): GPU 5-7 (每个GPU 12层，提高利用率)
        - GPU 7 专门处理Student的输出层，减少通信开销
        """
        
        # 🎯 Teacher分布：48层分布到5个GPU (0-4)，更均匀
        teacher_device_map = {"model.embed_tokens": 0}
        
        # 每个GPU大约9-10层，更精确分配
        teacher_layers_per_gpu = [10, 10, 10, 9, 9]  # 总计48层
        current_layer = 0
        
        for gpu_id, layer_count in enumerate(teacher_layers_per_gpu):
            for _ in range(layer_count):
                teacher_device_map[f"model.layers.{current_layer}"] = gpu_id
                current_layer += 1
        
        teacher_device_map.update({"model.norm": 4, "lm_head": 4})
        
        # 🎯 Student分布：36层分布到3个GPU (5-7)
        student_device_map = {"model.embed_tokens": 5}
        
        # 每个GPU 12层，充分利用
        student_layers_per_gpu = [12, 12, 12]  # 总计36层
        current_layer = 0
        
        for i, layer_count in enumerate(student_layers_per_gpu):
            gpu_id = 5 + i  # GPU 5, 6, 7
            for _ in range(layer_count):
                student_device_map[f"model.layers.{current_layer}"] = gpu_id
                current_layer += 1
        
        student_device_map.update({"model.norm": 7, "lm_head": 7})
        
        logger.info("🚀 启用8GPU优化分配:")
        logger.info(f"   Teacher (30B, 48层) → GPU 0-4: {teacher_layers_per_gpu}")
        logger.info(f"   Student (8B, 36层) → GPU 5-7: {student_layers_per_gpu}")
        logger.info("   预期GPU利用率显著提升!")
        
        return teacher_device_map, student_device_map
    
    def _create_balanced_mapping(self, gpu_count: int) -> Tuple[Dict, Dict]:
        """为6-7个GPU创建平衡分配"""
        
        # Teacher用前4个GPU
        teacher_device_map = {"model.embed_tokens": 0}
        layers_per_gpu = 48 // 4
        
        for i in range(48):
            gpu_id = i // layers_per_gpu
            gpu_id = min(gpu_id, 3)  # 限制在0-3
            teacher_device_map[f"model.layers.{i}"] = gpu_id
        teacher_device_map.update({"model.norm": 3, "lm_head": 3})
        
        # Student用后面的GPU
        student_device_map = {"model.embed_tokens": 4}
        remaining_gpus = gpu_count - 4
        student_layers_per_gpu = 36 // remaining_gpus
        
        for i in range(36):
            gpu_id = 4 + (i // student_layers_per_gpu)
            gpu_id = min(gpu_id, gpu_count - 1)
            student_device_map[f"model.layers.{i}"] = gpu_id
        student_device_map.update({"model.norm": gpu_count - 1, "lm_head": gpu_count - 1})
        
        logger.info(f"⚖️ 平衡分配 ({gpu_count}GPU): Teacher(0-3), Student(4-{gpu_count-1})")
        
        return teacher_device_map, student_device_map


class OptimizedKDTrainer(Trainer):
    """优化的KD训练器 - 支持动态学习率和早停"""
    
    def __init__(
        self,
        teacher_model=None,
        kd_loss_fn=None,
        temperature: float = 8.0,
        alpha: float = 0.8,
        eval_steps: int = 250,
        early_stopping_patience: int = 3,
        warmup_ratio: float = 0.1,
        lr_scheduler_type: str = "cosine",
        **kwargs
    ):
        self.teacher_model = teacher_model
        self.kd_loss_fn = kd_loss_fn
        self.temperature = temperature
        self.alpha = alpha
        self.eval_steps = eval_steps
        self.early_stopping_patience = early_stopping_patience
        self.warmup_ratio = warmup_ratio
        self.lr_scheduler_type = lr_scheduler_type
        
        # 早停相关
        self.best_eval_loss = float('inf')
        self.no_improve_count = 0
        self.should_stop = False
        
        super().__init__(**kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """计算KD损失"""
        labels = inputs.get("labels")
        
        # Student前向传播
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        if self.teacher_model is not None:
            # Teacher前向传播（无梯度）
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits
            
            # 计算KD损失
            kd_loss, loss_dict = self.kd_loss_fn(
                student_logits, teacher_logits, labels, inputs.get("attention_mask")
            )
            
            # 记录详细损失
            if self.state.global_step % self.args.logging_steps == 0:
                self.log({
                    "train/total_loss": loss_dict["total_loss"],
                    "train/ce_loss": loss_dict["ce_loss"], 
                    "train/kl_loss": loss_dict["kl_loss"],
                    "train/temperature": self.temperature,
                    "train/alpha": self.alpha
                })
            
            return (kd_loss, student_outputs) if return_outputs else kd_loss
        else:
            # 仅Student训练（离线模式）
            return student_outputs.loss if hasattr(student_outputs, 'loss') else None
    
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval"
    ):
        """增强评估 - 包含早停检查"""
        eval_result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # 早停检查
        current_eval_loss = eval_result.get(f"{metric_key_prefix}_loss", float('inf'))
        
        if current_eval_loss < self.best_eval_loss:
            self.best_eval_loss = current_eval_loss
            self.no_improve_count = 0
            logger.info(f"✅ 评估改善: {current_eval_loss:.4f} (最佳记录)")
        else:
            self.no_improve_count += 1
            logger.info(f"⚠️ 评估未改善: {current_eval_loss:.4f} (已{self.no_improve_count}次)")
            
            if self.no_improve_count >= self.early_stopping_patience:
                logger.info(f"🛑 早停触发: {self.early_stopping_patience}次未改善")
                self.should_stop = True
        
        return eval_result
    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        """重写以支持早停"""
        if self.should_stop:
            self.control.should_training_stop = True
            
        return super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)


def run_optimized_kd_pipeline(
    teacher_model_path: str,
    student_model_path: str,
    train_data: str,
    eval_data: Optional[str] = None,
    output_dir: str = "outputs/optimized_kd",
    experiment_name: str = "optimized_kd",
    
    # 训练参数
    kd_epochs: int = 4,
    kd_batch_size: int = 4,
    kd_lr: float = 1.5e-4,
    kd_grad_accum: int = 32,
    
    # 蒸馏参数
    temperature: float = 8.0,
    alpha: float = 0.8,
    
    # 序列参数
    max_length: int = 1536,
    
    # 评估参数
    eval_steps: int = 250,
    logging_steps: int = 50,
    save_steps: int = 500,
    
    # 学习率调度
    warmup_ratio: float = 0.1,
    lr_scheduler_type: str = "cosine",
    
    # 早停
    early_stopping_patience: int = 3,
    
    # 硬件优化
    use_bf16: bool = True,
    gradient_checkpointing: bool = True,
    dataloader_num_workers: int = 8,
    
    # GPU优化
    enable_8gpu_optimization: bool = True,
    
    # wandb
    use_wandb: bool = False,
    wandb_project: str = None,
    wandb_tags: List[str] = None,
    
    **kwargs
) -> Dict[str, Any]:
    """运行优化的KD管道"""
    
    logger.info("🚀 启动优化KD管道")
    logger.info(f"Teacher: {teacher_model_path}")
    logger.info(f"Student: {student_model_path}")
    logger.info(f"数据: {train_data}")
    logger.info(f"温度: {temperature} (优化)")
    logger.info(f"轮数: {kd_epochs} epochs")
    logger.info(f"批次: {kd_batch_size}x{kd_grad_accum}={kd_batch_size*kd_grad_accum}")
    
    # 创建输出目录
    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    try:
        # 🎯 步骤1: 初始化优化蒸馏器
        logger.info("📡 初始化优化KD蒸馏器...")
        distiller = OptimizedKDDistiller(
            teacher_model_path=teacher_model_path,
            student_model_path=student_model_path,
            output_dir=str(output_path),
            temperature=temperature,
            alpha=alpha,
            enable_8gpu_optimization=enable_8gpu_optimization
        )
        
        # 🎯 步骤2: 准备数据集
        logger.info("📚 加载训练数据...")
        train_dataset = DistillationDataset(
            data=load_jsonl(train_data),
            tokenizer=distiller.tokenizer,
            max_length=max_length
        )
        
        eval_dataset = None
        if eval_data:
            eval_dataset = DistillationDataset(
                data=load_jsonl(eval_data),
                tokenizer=distiller.tokenizer,
                max_length=max_length
            )
        
        logger.info(f"✅ 训练样本: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"✅ 评估样本: {len(eval_dataset)}")
        
        # 🎯 步骤3: 配置训练参数
        total_steps = (len(train_dataset) // (kd_batch_size * kd_grad_accum)) * kd_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        training_args = TrainingArguments(
            output_dir=str(output_path),
            
            # 训练轮数
            num_train_epochs=kd_epochs,
            per_device_train_batch_size=kd_batch_size,
            per_device_eval_batch_size=kd_batch_size,
            gradient_accumulation_steps=kd_grad_accum,
            
            # 学习率
            learning_rate=kd_lr,
            warmup_steps=warmup_steps,
            lr_scheduler_type=lr_scheduler_type,
            
            # 评估和保存
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=eval_steps if eval_dataset else None,
            save_steps=save_steps,
            save_total_limit=3,
            logging_steps=logging_steps,
            
            # 硬件优化
            bf16=use_bf16,
            gradient_checkpointing=gradient_checkpointing,
            dataloader_num_workers=dataloader_num_workers,
            
            # 其他
            remove_unused_columns=False,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            
            # wandb
            report_to="wandb" if use_wandb else None,
            run_name=f"{experiment_name}_{time.strftime('%m%d_%H%M')}" if use_wandb else None,
        )
        
        # 🎯 步骤4: 创建优化训练器
        logger.info("🏃 创建优化训练器...")
        trainer = OptimizedKDTrainer(
            model=distiller.student_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=distiller.tokenizer,
            data_collator=collate_fn,
            
            # KD特定参数
            teacher_model=distiller.teacher_model,
            kd_loss_fn=distiller.kd_loss_fn,
            temperature=temperature,
            alpha=alpha,
            
            # 优化参数
            eval_steps=eval_steps,
            early_stopping_patience=early_stopping_patience,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type
        )
        
        # 🎯 步骤5: 开始优化训练
        logger.info("🚀 开始优化KD训练...")
        logger.info(f"📊 总步数: {total_steps}, 预热步数: {warmup_steps}")
        
        train_result = trainer.train()
        
        # 🎯 步骤6: 保存最终模型
        logger.info("💾 保存优化后的模型...")
        final_model_path = output_path / "final_model"
        trainer.save_model(str(final_model_path))
        
        # 🎯 步骤7: 生成结果报告
        end_time = time.time()
        execution_time = end_time - start_time
        
        result = {
            "config": {
                "experiment_name": experiment_name,
                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                "models": {
                    "teacher": teacher_model_path,
                    "student": student_model_path
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
                    "effective_batch_size": kd_batch_size * kd_grad_accum,
                    "max_length": max_length,
                    "warmup_ratio": warmup_ratio,
                    "lr_scheduler_type": lr_scheduler_type
                },
                "distillation": {
                    "temperature": temperature,
                    "alpha": alpha,
                    "use_online_kd": True
                },
                "optimization": {
                    "8gpu_optimization": enable_8gpu_optimization,
                    "early_stopping_patience": early_stopping_patience,
                    "eval_steps": eval_steps
                },
                "hardware": {
                    "use_bf16": use_bf16,
                    "gradient_checkpointing": gradient_checkpointing,
                    "dataloader_num_workers": dataloader_num_workers
                }
            },
            "training_history": train_result.log_history if hasattr(train_result, 'log_history') else [],
            "execution_time": execution_time,
            "final_loss": train_result.training_loss if hasattr(train_result, 'training_loss') else None,
            "output_files": {
                "model": str(final_model_path),
                "train_data": train_data,
                "eval_data": eval_data
            }
        }
        
        # 保存结果
        result_file = output_path / "optimized_kd_results.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info("🎉 优化KD管道完成！")
        logger.info(f"⏱️  执行时间: {execution_time:.1f}秒 ({execution_time/3600:.1f}小时)")
        logger.info(f"💾 模型保存: {final_model_path}")
        logger.info(f"📊 结果保存: {result_file}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 优化KD训练失败: {e}")
        raise e
    finally:
        cleanup_cache()