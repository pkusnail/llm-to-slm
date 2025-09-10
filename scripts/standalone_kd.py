#!/usr/bin/env python3
"""
独立的知识蒸馏脚本
基于成功实验的配置重新实现，无外部依赖
"""

import os
import json
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import LoraConfig, get_peft_model
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class KDConfig:
    """Knowledge Distillation Configuration"""
    teacher_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    student_model: str = "Qwen/Qwen3-8B"
    train_data_path: str = "outputs/experiment/qwen3_30b_to_8b_ultrabatch_512/sft/sft_train_data_clean.jsonl"
    eval_data_path: str = "outputs/experiment/qwen3_30b_to_8b_ultrabatch_512/sft/sft_eval_data_clean.jsonl"
    output_dir: str = "outputs/experiment/standalone_kd"
    
    # Training parameters (based on successful config)
    learning_rate: float = 2e-5
    num_epochs: int = 1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    max_length: int = 1024
    
    # KD parameters
    temperature: float = 2.5
    alpha: float = 0.8  # Weight for KL loss
    
    # Hardware
    use_bf16: bool = True
    gradient_checkpointing: bool = True
    
    # GPU allocation
    teacher_gpus: List[int] = None  # Will be set based on available GPUs
    student_gpus: List[int] = None
    
    def __post_init__(self):
        if self.teacher_gpus is None:
            gpu_count = torch.cuda.device_count()
            if gpu_count >= 8:
                self.teacher_gpus = [0, 1, 2, 3, 4, 5]
                self.student_gpus = [6, 7]
            elif gpu_count >= 4:
                self.teacher_gpus = [0, 1, 2]
                self.student_gpus = [3]
            elif gpu_count >= 2:
                self.teacher_gpus = [0]
                self.student_gpus = [1]
            else:
                self.teacher_gpus = [0]
                self.student_gpus = [0]

class DistillationDataset(torch.utils.data.Dataset):
    """Dataset for Knowledge Distillation"""
    
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 1024):
        self.data = self.load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def load_data(self, data_path: str) -> List[Dict]:
        """Load JSONL data"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format prompt for Qwen models
        prompt = item['prompt']
        if not prompt.startswith('<|im_start|>'):
            prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze().clone()
        }

class KnowledgeDistillationTrainer(Trainer):
    """Custom trainer for Knowledge Distillation"""
    
    def __init__(self, teacher_model: PreTrainedModel, temperature: float = 2.0, alpha: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = CrossEntropyLoss()
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute distillation loss = α * KL_loss + (1 - α) * CE_loss
        """
        # Get student outputs
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        # Get teacher outputs (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        
        # Ensure same sequence length
        seq_len = min(student_logits.size(1), teacher_logits.size(1))
        student_logits = student_logits[:, :seq_len, :]
        teacher_logits = teacher_logits[:, :seq_len, :]
        
        # Compute KL divergence loss
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(
            student_probs, 
            teacher_probs, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Compute cross-entropy loss
        labels = inputs['labels'][:, :seq_len]
        ce_loss = self.ce_loss(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
        
        # Combined loss
        total_loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss
        
        # Log individual losses for monitoring
        if hasattr(self, '_log_counter'):
            self._log_counter += 1
        else:
            self._log_counter = 1
            
        if self._log_counter % 10 == 0:
            logger.info(f"Step {self._log_counter}: KL_loss: {kl_loss:.4f}, CE_loss: {ce_loss:.4f}, Total: {total_loss:.4f}")
        
        return (total_loss, student_outputs) if return_outputs else total_loss

def create_device_map(num_layers: int, gpu_ids: List[int]) -> Dict[str, int]:
    """Create device map for model parallelism"""
    if len(gpu_ids) == 1:
        return {"": gpu_ids[0]}
    
    device_map = {}
    layers_per_gpu = num_layers // len(gpu_ids)
    
    # Embedding layer
    device_map["model.embed_tokens"] = gpu_ids[0]
    
    # Distribute transformer layers
    for i in range(num_layers):
        gpu_idx = min(i // layers_per_gpu, len(gpu_ids) - 1)
        device_map[f"model.layers.{i}"] = gpu_ids[gpu_idx]
    
    # Output layers
    device_map["model.norm"] = gpu_ids[-1]
    device_map["lm_head"] = gpu_ids[-1]
    
    return device_map

def load_models(config: KDConfig):
    """Load teacher and student models with proper GPU allocation"""
    
    logger.info(f"Loading tokenizer: {config.student_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading student model: {config.student_model}")
    student_device_map = create_device_map(36, config.student_gpus)  # Qwen3-8B has 36 layers
    logger.info(f"Student device map: {student_device_map}")
    
    student_model = AutoModelForCausalLM.from_pretrained(
        config.student_model,
        torch_dtype=torch.bfloat16 if config.use_bf16 else torch.float16,
        device_map=student_device_map,
        trust_remote_code=True
    )
    
    # Apply LoRA to student model
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    student_model = get_peft_model(student_model, lora_config)
    logger.info(f"Applied LoRA to student model")
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in student_model.parameters())
    logger.info(f"Student model: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")
    
    logger.info(f"Loading teacher model: {config.teacher_model}")
    teacher_device_map = create_device_map(48, config.teacher_gpus)  # Qwen3-30B has 48 layers
    logger.info(f"Teacher device map: {teacher_device_map}")
    
    teacher_model = AutoModelForCausalLM.from_pretrained(
        config.teacher_model,
        torch_dtype=torch.bfloat16 if config.use_bf16 else torch.float16,
        device_map=teacher_device_map,
        trust_remote_code=True
    )
    teacher_model.eval()
    
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    logger.info(f"Teacher model: {teacher_params/1e6:.1f}M parameters")
    
    return student_model, teacher_model, tokenizer

def run_knowledge_distillation(config: KDConfig):
    """Run knowledge distillation training"""
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_dict = {
        "experiment_name": output_dir.name,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "models": {
            "teacher": config.teacher_model,
            "student": config.student_model
        },
        "data": {
            "train": config.train_data_path,
            "eval": config.eval_data_path
        },
        "training": {
            "epochs": config.num_epochs,
            "batch_size": config.per_device_batch_size,
            "learning_rate": config.learning_rate,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "max_length": config.max_length
        },
        "distillation": {
            "temperature": config.temperature,
            "alpha": config.alpha,
            "use_online_kd": True,
            "generate_teacher_logits": False
        },
        "hardware": {
            "use_bf16": config.use_bf16,
            "gradient_checkpointing": config.gradient_checkpointing,
            "teacher_gpus": config.teacher_gpus,
            "student_gpus": config.student_gpus
        },
        "random_seed": 42
    }
    
    with open(output_dir / "kd_config.json", 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Starting KD: {config.teacher_model} -> {config.student_model}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load models
    student_model, teacher_model, tokenizer = load_models(config)
    
    # Load datasets
    train_dataset = DistillationDataset(config.train_data_path, tokenizer, config.max_length)
    eval_dataset = DistillationDataset(config.eval_data_path, tokenizer, config.max_length)
    
    logger.info(f"Training samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    logger.info(f"KD parameters: temperature={config.temperature}, α={config.alpha}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        weight_decay=0.01,
        max_grad_norm=0.5,
        logging_steps=5,
        save_steps=50,
        eval_steps=50,
        evaluation_strategy="steps",
        save_strategy="steps",
        bf16=config.use_bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to=None
    )
    
    # Create custom trainer
    trainer = KnowledgeDistillationTrainer(
        teacher_model=teacher_model,
        temperature=config.temperature,
        alpha=config.alpha,
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )
    
    logger.info("Starting knowledge distillation training...")
    
    # Train
    train_result = trainer.train()
    
    # Save final model
    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    # Save training results
    results = {
        "train_results": train_result.metrics if hasattr(train_result, 'metrics') else {},
        "config": config_dict,
        "final_model_path": str(final_model_dir)
    }
    
    with open(output_dir / "kd_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Knowledge distillation completed!")
    logger.info(f"Model saved to: {final_model_dir}")
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Standalone Knowledge Distillation")
    parser.add_argument("--teacher_model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--student_model", default="Qwen/Qwen3-8B")
    parser.add_argument("--train_data", default="outputs/experiment/qwen3_30b_to_8b_ultrabatch_512/sft/sft_train_data_clean.jsonl")
    parser.add_argument("--eval_data", default="outputs/experiment/qwen3_30b_to_8b_ultrabatch_512/sft/sft_eval_data_clean.jsonl")
    parser.add_argument("--output_dir", default="outputs/experiment/standalone_kd")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=2.5)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=1024)
    
    args = parser.parse_args()
    
    # Check data files exist
    if not os.path.exists(args.train_data):
        logger.error(f"Training data not found: {args.train_data}")
        return
    
    if not os.path.exists(args.eval_data):
        logger.error(f"Evaluation data not found: {args.eval_data}")
        return
    
    # Create config
    config = KDConfig(
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        alpha=args.alpha,
        num_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # Run KD
    try:
        results = run_knowledge_distillation(config)
        logger.info("✅ Knowledge distillation completed successfully!")
    except Exception as e:
        logger.error(f"❌ Knowledge distillation failed: {e}")
        raise

if __name__ == "__main__":
    main()