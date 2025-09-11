"""
蒸馏基类和通用组件
"""
import os
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    PreTrainedModel
)
from peft import LoraConfig, get_peft_model
import logging

logger = logging.getLogger(__name__)


class BaseDistiller(ABC):
    """蒸馏器基类"""
    
    def __init__(
        self,
        teacher_model_path: str,
        student_model_path: str,
        output_dir: str,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        self.teacher_model_path = teacher_model_path
        self.student_model_path = student_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        
        # 模型和分词器将在子类中初始化
        self.teacher_model = None
        self.student_model = None
        self.tokenizer = None
    
    def load_teacher_model(self, teacher_device_map=None, **kwargs):
        """加载教师模型"""
        logger.info(f"Loading teacher model: {self.teacher_model_path}")
        
        # 使用专门的teacher device_map或默认值
        device_map = teacher_device_map or self.device_map
        logger.info(f"Teacher device_map: {device_map}")
        
        try:
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                self.teacher_model_path,
                torch_dtype=self.torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                **kwargs
            )
            self.teacher_model.eval()
        except Exception as e:
            logger.error(f"Failed to load teacher model: {e}")
            raise
        
        # 冻结教师模型
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        logger.info(f"Teacher model loaded with {sum(p.numel() for p in self.teacher_model.parameters())/1e6:.1f}M parameters")
    
    def load_student_model(self, use_lora: bool = True, lora_config: Optional[Dict] = None, student_device_map=None, **kwargs):
        """加载学生模型"""
        logger.info(f"Loading student model: {self.student_model_path}")
        
        # 使用专门的student device_map或默认值
        device_map = student_device_map or self.device_map
        logger.info(f"Student device_map: {device_map}")
        
        # 默认LoRA配置
        default_lora_config = {
            "r": 64,
            "lora_alpha": 16, 
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }
        
        if lora_config:
            default_lora_config.update(lora_config)
        
        try:
            self.student_model = AutoModelForCausalLM.from_pretrained(
                self.student_model_path,
                torch_dtype=self.torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to load student model: {e}")
            raise
        
        if use_lora:
            lora_config_obj = LoraConfig(**default_lora_config)
            self.student_model = get_peft_model(self.student_model, lora_config_obj)
            # 确保LoRA参数需要梯度
            self.student_model.train()
            logger.info(f"Applied LoRA to student model")
            
        # 统计可训练参数
        total_params = sum(p.numel() for p in self.student_model.parameters())
        trainable_params = sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)
        
        logger.info(f"Student model loaded: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")
    
    def load_tokenizer(self, model_path: Optional[str] = None):
        """加载分词器"""
        if model_path is None:
            model_path = self.student_model_path
            
        logger.info(f"Loading tokenizer: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    @abstractmethod
    def distill(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None, **kwargs):
        """执行蒸馏训练"""
        pass
    
    def save_model(self, save_path: Optional[str] = None):
        """保存学生模型"""
        if save_path is None:
            save_path = self.output_dir / "final_model"
        
        logger.info(f"Saving model to {save_path}")
        
        if hasattr(self.student_model, 'save_pretrained'):
            self.student_model.save_pretrained(save_path)
        else:
            torch.save(self.student_model.state_dict(), save_path / "pytorch_model.bin")
        
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_path)


class KnowledgeDistillationLoss(nn.Module):
    """知识蒸馏损失函数
    
    Args:
        temperature: 软化概率分布的温度参数，建议2.0-5.0
        alpha: KL损失的权重，建议0.7-0.8
               当alpha=0.8时: loss = 0.2*CE_loss + 0.8*KL_loss
               即80%学习teacher分布，20%保持accuracy
    """
    
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # KL损失权重，建议设置为0.7-0.8
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self, 
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor, 
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算知识蒸馏损失
        
        Args:
            student_logits: 学生模型输出 [batch_size, seq_len, vocab_size]
            teacher_logits: 教师模型输出 [batch_size, seq_len, vocab_size]  
            labels: 标签 [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
        
        Returns:
            总损失和各部分损失的字典
        """
        # 交叉熵损失 (对真实标签)
        ce_loss = self.ce_loss(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
        
        # KL散度损失 (对教师知识)
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        if attention_mask is not None:
            # 只在有效位置计算KL
            mask = (labels != -100) & (attention_mask.bool())
            mask = mask.unsqueeze(-1).expand_as(student_probs)
            
            masked_student_probs = student_probs[mask].view(-1, student_probs.size(-1))
            masked_teacher_probs = teacher_probs[mask].view(-1, teacher_probs.size(-1))
            
            if masked_student_probs.size(0) > 0:
                kl_loss = self.kl_div(masked_student_probs, masked_teacher_probs)
            else:
                kl_loss = torch.tensor(0.0, device=student_logits.device)
        else:
            kl_loss = self.kl_div(student_probs.view(-1, student_probs.size(-1)), 
                                teacher_probs.view(-1, teacher_probs.size(-1)))
        
        # 温度平方补偿
        kl_loss = kl_loss * (self.temperature ** 2)
        
        # 总损失 - 修复权重：让KL Loss占主导地位
        # alpha=0.8时: total_loss = 0.2 * ce_loss + 0.8 * kl_loss
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * kl_loss
        
        loss_dict = {
            "total_loss": total_loss.item(),
            "ce_loss": ce_loss.item(), 
            "kl_loss": kl_loss.item()
        }
        
        return total_loss, loss_dict


class DistillationDataset(torch.utils.data.Dataset):
    """蒸馏数据集"""
    
    def __init__(
        self, 
        data: List[Dict[str, Any]], 
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        chat_template: bool = True
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chat_template = chat_template
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt'] 
        # 优先使用teacher_response，向后兼容expected_answer
        response = item.get('teacher_response', item.get('expected_answer', ''))
        
        if self.chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
            # 使用聊天模板
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            
            # 计算prompt长度用于标签掩码
            prompt_messages = [{"role": "user", "content": prompt}]
            prompt_text = self.tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=False)
            prompt_tokens = self.tokenizer(prompt_text, add_special_tokens=False)['input_ids']
            prompt_length = len(prompt_tokens)
        else:
            # 简单拼接
            text = f"### 问题:\n{prompt}\n\n### 回答:\n{response}"
            # 估算prompt长度
            prompt_text = f"### 问题:\n{prompt}\n\n### 回答:\n"
            prompt_tokens = self.tokenizer(prompt_text, add_special_tokens=False)['input_ids']
            prompt_length = len(prompt_tokens)
        
        # 分词
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        
        # 创建标签：只对回答部分计算损失
        labels = input_ids.copy()
        labels[:prompt_length] = [-100] * min(prompt_length, len(labels))
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long), 
            'labels': torch.tensor(labels, dtype=torch.long),
            'id': item.get('id', idx),
            'domain': item.get('domain', 'unknown')
        }


def collate_fn(batch):
    """数据整理函数"""
    # 获取批次中的最大长度
    max_length = max(len(item['input_ids']) for item in batch)
    
    input_ids = []
    attention_mask = []
    labels = []
    ids = []
    domains = []
    
    for item in batch:
        # 右填充
        pad_length = max_length - len(item['input_ids'])
        
        input_ids.append(torch.cat([
            item['input_ids'],
            torch.full((pad_length,), 0, dtype=torch.long)  # 使用pad_token_id=0
        ]))
        
        attention_mask.append(torch.cat([
            item['attention_mask'], 
            torch.zeros(pad_length, dtype=torch.long)
        ]))
        
        labels.append(torch.cat([
            item['labels'],
            torch.full((pad_length,), -100, dtype=torch.long)
        ]))
        
        ids.append(item['id'])
        domains.append(item['domain'])
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack(labels),
        'ids': ids,
        'domains': domains
    }