"""
通用工具函数
"""
import json
import os
import random
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import torch
import numpy as np
from transformers import AutoTokenizer


def set_seed(seed: int = 42):
    """设置随机种子以确保实验可重复"""
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file: Optional[str] = None, level: Union[str, int] = "INFO"):
    """设置日志系统"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    # Handle both string and int level inputs
    if isinstance(level, str):
        log_level = getattr(logging, level.upper())
    else:
        log_level = level
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )


def load_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: Union[str, Path]):
    """保存JSONL文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def get_model_size(model) -> int:
    """获取模型参数量"""
    return sum(p.numel() for p in model.parameters())


def get_model_memory_usage(model) -> float:
    """获取模型显存占用 (GB)"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024**3)


def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


class ChatTemplate:
    """聊天模板处理器"""
    
    def __init__(self, tokenizer_name_or_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def apply_chat_template(self, messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> str:
        """应用聊天模板"""
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=add_generation_prompt,
                tokenize=False
            )
        else:
            # 简单的模板回退
            formatted = ""
            for msg in messages:
                role = msg['role']
                content = msg['content']
                if role == 'user':
                    formatted += f"Human: {content}\n"
                elif role == 'assistant':
                    formatted += f"Assistant: {content}\n"
                elif role == 'system':
                    formatted += f"System: {content}\n"
            
            if add_generation_prompt:
                formatted += "Assistant:"
            
            return formatted
    
    def format_instruction_data(self, prompt: str, response: str = None) -> Dict[str, Any]:
        """格式化指令数据"""
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        if response:
            messages.append({"role": "assistant", "content": response})
        
        text = self.apply_chat_template(messages, add_generation_prompt=response is None)
        
        result = {"text": text, "messages": messages}
        
        # 如果有响应，计算labels
        if response:
            # 找到assistant回复的开始位置
            prompt_text = self.apply_chat_template(messages[:-1], add_generation_prompt=True)
            prompt_length = len(self.tokenizer.encode(prompt_text))
            
            full_ids = self.tokenizer.encode(text)
            labels = [-100] * prompt_length + full_ids[prompt_length:]
            
            result["input_ids"] = full_ids
            result["labels"] = labels
        
        return result


def calculate_metrics(predictions: List[str], references: List[str], metric_type: str = "exact_match") -> Dict[str, float]:
    """计算评测指标"""
    if metric_type == "exact_match":
        matches = sum(1 for pred, ref in zip(predictions, references) if pred.strip().lower() == ref.strip().lower())
        return {"exact_match": matches / len(predictions) if predictions else 0.0}
    
    elif metric_type == "contains":
        # 检查预测是否包含参考答案的关键部分
        matches = sum(1 for pred, ref in zip(predictions, references) if ref.strip().lower() in pred.strip().lower())
        return {"contains_match": matches / len(predictions) if predictions else 0.0}
    
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


def print_gpu_utilization():
    """打印GPU使用情况"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_used = torch.cuda.memory_allocated(i) / (1024**3)
            memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3) 
            print(f"GPU {i}: {memory_used:.1f}GB / {memory_total:.1f}GB ({memory_used/memory_total*100:.1f}%)")


def cleanup_cache():
    """清理CUDA缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()