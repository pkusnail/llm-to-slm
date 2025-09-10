"""
知识蒸馏 (Knowledge Distillation) 实现
在线/离线KL散度蒸馏，学习教师模型的输出分布
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer
)
from accelerate import Accelerator
import numpy as np

from .base import BaseDistiller, DistillationDataset, collate_fn, KnowledgeDistillationLoss
from ..utils.common import load_jsonl, setup_logging, set_seed, cleanup_cache

logger = logging.getLogger(__name__)


class KDDistiller(BaseDistiller):
    """知识蒸馏器"""
    
    def __init__(
        self,
        teacher_model_path: str,
        student_model_path: str,
        output_dir: str,
        temperature: float = 2.0,
        alpha: float = 0.5,
        use_online_kd: bool = True,
        **kwargs
    ):
        super().__init__(teacher_model_path, student_model_path, output_dir, **kwargs)
        
        self.temperature = temperature
        self.alpha = alpha
        self.use_online_kd = use_online_kd
        
        # 手动分配GPU：30B模型(48层)分布到6个GPU，8B模型到后2个GPU
        gpu_count = torch.cuda.device_count()
        if gpu_count >= 8:
            # Teacher模型(48层)分布到前6个GPU，每个GPU 8层
            teacher_device_map = {"model.embed_tokens": 0}
            for i in range(48):
                gpu_id = i // 8  # 每8层一个GPU：0-7→GPU0, 8-15→GPU1, ..., 40-47→GPU5
                teacher_device_map[f"model.layers.{i}"] = gpu_id
            teacher_device_map.update({"model.norm": 5, "lm_head": 5})
            
            logger.info(f"Teacher model (48层) → GPU 0-5, Student model (36层) → GPU 6-7")
            # Student模型(36层)分布到后2个GPU，每个GPU 18层
            student_device_map = {"model.embed_tokens": 6}
            for i in range(36):
                gpu_id = 6 + (i // 18)  # 0-17→GPU6, 18-35→GPU7
                student_device_map[f"model.layers.{i}"] = gpu_id
            student_device_map.update({"model.norm": 7, "lm_head": 7})
        else:
            # 备用分配：如果GPU不足8个
            teacher_device_map = {"model.embed_tokens": 0}
            layers_per_gpu = 48 // min(gpu_count-2, 6) + 1
            for i in range(48):
                gpu_id = min(i // layers_per_gpu, gpu_count-3)
                teacher_device_map[f"model.layers.{i}"] = gpu_id
            teacher_device_map.update({"model.norm": gpu_count-3, "lm_head": gpu_count-3})
            
            student_device_map = {"model.embed_tokens": gpu_count-2}
            for i in range(36):
                gpu_id = (gpu_count-2) + (i // 18)
                student_device_map[f"model.layers.{i}"] = min(gpu_id, gpu_count-1)
            student_device_map.update({"model.norm": gpu_count-1, "lm_head": gpu_count-1})
        
        logger.info("使用手动GPU分配: Teacher(GPU0-5), Student(GPU6-7)")
        
        # 加载模型
        self.load_tokenizer()
        self.load_student_model(student_device_map=student_device_map)
        
        if use_online_kd:
            self.load_teacher_model(teacher_device_map=teacher_device_map)
        
        # 损失函数
        self.kd_loss_fn = KnowledgeDistillationLoss(temperature, alpha)
    
    def distill(
        self,
        train_data: Union[List[Dict], str], 
        eval_data: Optional[Union[List[Dict], str]] = None,
        training_args: Optional[Dict] = None,
        **kwargs
    ):
        """执行KD训练"""
        
        # 加载数据
        if isinstance(train_data, str):
            train_data = load_jsonl(train_data)
        if isinstance(eval_data, str):
            eval_data = load_jsonl(eval_data)
        
        logger.info(f"KD训练开始，训练样本: {len(train_data)}, 温度: {self.temperature}, α: {self.alpha}")
        
        if self.use_online_kd:
            return self._online_distill(train_data, eval_data, training_args, **kwargs)
        else:
            return self._offline_distill(train_data, eval_data, training_args, **kwargs)
    
    def _online_distill(
        self,
        train_data: List[Dict],
        eval_data: Optional[List[Dict]],
        training_args: Optional[Dict],
        **kwargs
    ):
        """在线知识蒸馏"""
        
        # 默认训练参数
        default_args = {
            "output_dir": str(self.output_dir),
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2, 
            "gradient_accumulation_steps": 8,
            "learning_rate": 1.5e-4,
            "num_train_epochs": 2,
            "warmup_ratio": 0.1,
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 500,
            "eval_strategy": "steps" if eval_data else "no",
            "save_strategy": "steps",
            "load_best_model_at_end": True if eval_data else False,
            "bf16": True,
            "gradient_checkpointing": False,
            "dataloader_pin_memory": False,
            "remove_unused_columns": False,
            "report_to": "none"
        }
        
        if training_args:
            default_args.update(training_args)
        
        # 创建数据集
        train_dataset = DistillationDataset(
            train_data,
            self.tokenizer,
            max_length=kwargs.get('max_length', 2048)
        )
        
        eval_dataset = None
        if eval_data:
            eval_dataset = DistillationDataset(
                eval_data,
                self.tokenizer,
                max_length=kwargs.get('max_length', 2048)
            )
        
        # 创建自定义训练器
        trainer = KDTrainer(
            model=self.student_model,
            teacher_model=self.teacher_model,
            kd_loss_fn=self.kd_loss_fn,
            args=TrainingArguments(**default_args),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collate_fn,
            tokenizer=self.tokenizer
        )
        
        # 开始训练
        logger.info("开始在线KD训练...")
        trainer.train()
        
        # 保存模型
        logger.info("保存KD模型...")
        trainer.save_model(str(self.output_dir / "final_model"))
        
        return trainer.state.log_history
    
    def _offline_distill(
        self,
        train_data: List[Dict],
        eval_data: Optional[List[Dict]],
        training_args: Optional[Dict],
        **kwargs
    ):
        """离线知识蒸馏（使用预存的teacher logits）"""
        # TODO: 实现离线蒸馏
        raise NotImplementedError("Offline KD will be implemented later")


class KDTrainer(Trainer):
    """自定义KD训练器"""
    
    def __init__(self, teacher_model, kd_loss_fn, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.kd_loss_fn = kd_loss_fn
        
        # 确保教师模型在eval模式
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # 确保学生模型在训练模式，LoRA参数需要梯度
        self.model.train()
        
        # 强制启用LoRA参数的梯度
        if hasattr(self.model, 'peft_config') and self.model.peft_config:
            # 这是一个PEFT模型，需要确保LoRA参数可训练
            try:
                self.model.enable_adapters()
                print("Successfully enabled adapters")
            except Exception as e:
                print(f"Warning: Could not enable adapters: {e}")
            
            for name, param in self.model.named_parameters():
                if 'lora_' in name.lower():
                    param.requires_grad_(True)
                    print(f"Enabled gradients for LoRA parameter: {name}")
        else:
            # 如果不是PEFT模型，确保所有可训练参数需要梯度
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.requires_grad_(True)
        
        # 验证可训练参数数量
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"KDTrainer: {trainable_params:,} trainable parameters")
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """计算KD损失"""
        
        # 确保模型在训练模式且梯度启用
        model.train()
        
        # 准备输入数据并确保设备一致性
        model_inputs = {}
        student_device = next(model.parameters()).device
        
        for k, v in inputs.items():
            if k in ['input_ids', 'attention_mask', 'labels']:
                if hasattr(v, 'to'):
                    model_inputs[k] = v.to(student_device)
                else:
                    model_inputs[k] = v
        
        print(f"DEBUG: Student device={student_device}, input_ids device={model_inputs['input_ids'].device}")
        
        # 学生前向传播
        student_outputs = model(**model_inputs)
        student_logits = student_outputs.logits
        
        # 教师前向传播 - 确保输入在正确设备上
        teacher_device = next(self.teacher_model.parameters()).device
        teacher_inputs = {}
        for k, v in model_inputs.items():
            if hasattr(v, 'to'):
                teacher_inputs[k] = v.to(teacher_device)
            else:
                teacher_inputs[k] = v
        
        print(f"DEBUG: Teacher device={teacher_device}, input_ids device={teacher_inputs['input_ids'].device}")
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**teacher_inputs)
            teacher_logits = teacher_outputs.logits
            
        # 将teacher logits移动到student设备进行损失计算
        teacher_logits = teacher_logits.to(student_device)
        
        # 计算KD损失 - 确保所有输入在同一设备
        labels = model_inputs.get('labels')
        attention_mask = model_inputs.get('attention_mask')
        
        loss, loss_dict = self.kd_loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            attention_mask=attention_mask
        )
        
        # 关键修复：Trainer期望loss在cuda:0，不是student设备
        expected_device = torch.device("cuda:0")
        if loss.device != expected_device:
            loss = loss.to(expected_device)
            print(f"DEBUG: Moved loss to {expected_device} (Trainer requirement)")
        
        # 验证损失张量需要梯度
        if not loss.requires_grad:
            print("WARNING: Loss tensor does not require grad!")
            # 强制重新计算确保梯度追踪
            loss = loss.clone().requires_grad_(True)
        
        # 记录各部分损失
        self.log(loss_dict)
        
        if return_outputs:
            return loss, student_outputs
        return loss


class TeacherLogitsGenerator:
    """教师模型logits生成器（用于离线KD）"""
    
    def __init__(
        self,
        teacher_model_path: str,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        self.teacher_model_path = teacher_model_path
        logger.info(f"Loading teacher model for logits generation: {teacher_model_path}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            teacher_model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            teacher_model_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_logits_dataset(
        self,
        input_data: Union[List[Dict], str],
        output_path: str,
        top_k: int = 20,
        batch_size: int = 4,
        max_length: int = 2048
    ):
        """生成包含teacher logits的数据集"""
        
        if isinstance(input_data, str):
            input_data = load_jsonl(input_data)
        
        logger.info(f"生成teacher logits数据集，样本数: {len(input_data)}, top_k: {top_k}")
        
        # 创建数据集
        dataset = DistillationDataset(input_data, self.tokenizer, max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        
        output_data = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # 教师前向
                teacher_outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                teacher_logits = teacher_outputs.logits  # [batch_size, seq_len, vocab_size]
                
                # 获取top-k logits
                top_k_logits, top_k_indices = torch.topk(teacher_logits, k=top_k, dim=-1)
                
                # 转换为CPU并保存
                for i in range(len(batch['input_ids'])):
                    item_data = input_data[batch_idx * batch_size + i].copy()
                    
                    # 添加teacher logits信息
                    seq_len = batch['attention_mask'][i].sum().item()  # 实际序列长度
                    
                    item_data['teacher_logits'] = {
                        'top_k_values': top_k_logits[i, :seq_len].cpu().numpy().tolist(),
                        'top_k_indices': top_k_indices[i, :seq_len].cpu().numpy().tolist(),
                        'input_ids': batch['input_ids'][i, :seq_len].cpu().numpy().tolist(),
                        'labels': batch['labels'][i, :seq_len].cpu().numpy().tolist()
                    }
                    
                    output_data.append(item_data)
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
                    cleanup_cache()  # 清理显存
        
        # 保存数据
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in output_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Teacher logits数据集已保存到: {output_path}")
        return output_data


def run_kd_pipeline(
    teacher_model_path: str,
    student_model_path: str,
    train_data_path: str,
    output_dir: str,
    eval_data_path: Optional[str] = None,
    use_online_kd: bool = True,
    generate_teacher_logits: bool = False,
    training_args: Optional[Dict] = None,
    kd_args: Optional[Dict] = None,
    **kwargs
):
    """运行完整的KD管道"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(str(output_dir / "kd.log"))
    set_seed(kwargs.get('seed', 42))
    
    logger.info(f"开始KD管道：{teacher_model_path} -> {student_model_path}")
    
    # 默认KD参数
    default_kd_args = {
        "temperature": 2.0,
        "alpha": 0.5,
        "use_online_kd": use_online_kd
    }
    if kd_args:
        default_kd_args.update(kd_args)
    
    # 1. 生成teacher logits（如果使用离线KD）
    if not use_online_kd and generate_teacher_logits:
        logger.info("步骤 1/2: 生成teacher logits")
        
        logits_generator = TeacherLogitsGenerator(teacher_model_path)
        
        # 生成训练数据logits
        kd_train_path = output_dir / "kd_train_data_with_logits.jsonl"
        logits_generator.generate_logits_dataset(
            train_data_path,
            str(kd_train_path),
            top_k=kwargs.get('top_k', 20),
            batch_size=kwargs.get('logits_batch_size', 2)
        )
        
        # 生成评估数据logits
        kd_eval_path = None
        if eval_data_path:
            kd_eval_path = output_dir / "kd_eval_data_with_logits.jsonl"
            logits_generator.generate_logits_dataset(
                eval_data_path,
                str(kd_eval_path),
                top_k=kwargs.get('top_k', 20),
                batch_size=kwargs.get('logits_batch_size', 2)
            )
        
        # 清理
        del logits_generator
        torch.cuda.empty_cache()
    else:
        kd_train_path = train_data_path
        kd_eval_path = eval_data_path
    
    # 2. KD训练
    logger.info(f"步骤 2/2: {'在线' if use_online_kd else '离线'}KD训练")
    
    distiller = KDDistiller(
        teacher_model_path=teacher_model_path,
        student_model_path=student_model_path,
        output_dir=str(output_dir),
        **default_kd_args
    )
    
    training_history = distiller.distill(
        train_data=str(kd_train_path),
        eval_data=str(kd_eval_path) if kd_eval_path else None,
        training_args=training_args,
        **kwargs
    )
    
    logger.info("KD管道完成！")
    return training_history