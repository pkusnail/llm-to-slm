"""
监督微调 (SFT) 实现
这是最基础的蒸馏方法：用教师生成的数据训练学生
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from trl import SFTTrainer
import wandb

from distillation.base import BaseDistiller, DistillationDataset, collate_fn
from utils.common import load_jsonl, setup_logging, set_seed

logger = logging.getLogger(__name__)


class SFTDistiller(BaseDistiller):
    """监督微调蒸馏器"""
    
    def __init__(
        self,
        teacher_model_path: str,
        student_model_path: str, 
        output_dir: str,
        skip_teacher_loading: bool = True,
        **kwargs
    ):
        # 修改base class初始化以支持跳过教师模型加载
        self.teacher_model_path = teacher_model_path
        self.student_model_path = student_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 从kwargs中提取常用参数
        self.device_map = kwargs.get('device_map', 'auto')
        self.torch_dtype = kwargs.get('torch_dtype', torch.bfloat16)
        
        # SFT通常不需要加载教师模型，除非需要生成数据
        self.teacher_model = None
        if not skip_teacher_loading:
            try:
                self.load_teacher_model()
            except Exception as e:
                logger.error(f"Teacher model loading failed: {e}")
                raise
        
        try:
            self.load_student_model()
            self.load_tokenizer()
        except Exception as e:
            logger.error(f"SFT初始化失败: {e}")
            raise
    
    def load_teacher_model(self, device_map=None, **kwargs):
        """加载教师模型"""
        logger.info(f"Loading teacher model: {self.teacher_model_path}")
        
        from transformers import AutoModelForCausalLM
        
        device_map = device_map or self.device_map
        
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
        
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
        
        # 使用专门的student device_map或默认值
        device_map = student_device_map or self.device_map
        logger.info(f"Student device_map: {device_map}")
        
        # 默认LoRA配置
        default_lora_config = {
            "r": 64,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.1,
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
            
            if use_lora:
                lora_config_obj = LoraConfig(**default_lora_config)
                self.student_model = get_peft_model(self.student_model, lora_config_obj)
                # 确保LoRA参数需要梯度
                self.student_model.train()
                logger.info("Applied LoRA to student model")
            
        except Exception as e:
            logger.error(f"Failed to load student model: {e}")
            raise
            
        logger.info(f"Student model loaded with {sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)/1e6:.1f}M trainable parameters")
    
    def load_tokenizer(self):
        """加载分词器"""
        from transformers import AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.student_model_path,
            trust_remote_code=True
        )
        
        # 确保有pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info("Tokenizer loaded")
    
    def distill(
        self,
        train_data: Union[List[Dict], str],
        eval_data: Optional[Union[List[Dict], str]] = None,
        training_args: Optional[Dict] = None,
        **kwargs
    ):
        """执行SFT训练"""
        
        # 加载数据
        if isinstance(train_data, str):
            train_data = load_jsonl(train_data)
        if isinstance(eval_data, str):
            eval_data = load_jsonl(eval_data)
        
        logger.info(f"SFT训练开始，训练样本: {len(train_data)}")
        if eval_data:
            logger.info(f"评估样本: {len(eval_data)}")
        
        # 默认训练参数
        default_args = {
            "output_dir": str(self.output_dir),
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "num_train_epochs": 3,
            "warmup_ratio": 0.1,
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 500,
            "eval_strategy": "steps" if eval_data else "no",
            "save_strategy": "steps",
            "load_best_model_at_end": True if eval_data else False,
            "metric_for_best_model": "eval_loss" if eval_data else None,
            "bf16": True,
            "remove_unused_columns": False,
            "gradient_checkpointing": True,
            "dataloader_pin_memory": False,
            "report_to": "none"  # 关闭wandb
        }
        
        if training_args:
            default_args.update(training_args)
        
        training_arguments = TrainingArguments(**default_args)
        
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
        
        # 创建训练器
        trainer = Trainer(
            model=self.student_model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collate_fn,
            tokenizer=self.tokenizer
        )
        
        # 开始训练
        logger.info("开始SFT训练...")
        trainer.train()
        
        # 保存最终模型
        logger.info("保存SFT模型...")
        trainer.save_model(str(self.output_dir / "final_model"))
        
        # 返回训练历史
        return trainer.state.log_history


class TeacherResponseGenerator:
    """教师模型响应生成器"""
    
    def __init__(
        self, 
        teacher_model_path: str,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        self.teacher_model_path = teacher_model_path
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        
        logger.info(f"Loading teacher model for generation: {teacher_model_path}")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
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
    
    def generate_responses(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        batch_size: int = 512,  # 超大批次最大化GPU利用率
        **kwargs
    ) -> List[str]:
        """批量生成教师响应"""
        
        responses = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # 构建消息格式
            batch_messages = []
            for prompt in batch_prompts:
                messages = [{"role": "user", "content": prompt}]
                batch_messages.append(messages)
            
            # 应用聊天模板
            batch_texts = []
            for messages in batch_messages:
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False
                    )
                else:
                    text = f"Human: {messages[0]['content']}\n\nAssistant:"
                batch_texts.append(text)
            
            # 分词
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # 解码响应
            for j, output in enumerate(outputs):
                input_length = inputs['input_ids'][j].shape[0]
                response_ids = output[input_length:]
                response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                responses.append(response.strip())
            
            logger.info(f"Generated batch {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}")
        
        return responses
    
    def generate_sft_dataset(
        self,
        input_data: Union[List[Dict], str],
        output_path: str,
        **generation_kwargs
    ) -> List[Dict]:
        """生成SFT训练数据集"""
        
        if isinstance(input_data, str):
            input_data = load_jsonl(input_data)
        
        logger.info(f"使用教师模型生成SFT数据集，输入样本: {len(input_data)}")
        
        # 提取prompts
        prompts = [item['prompt'] for item in input_data]
        
        # 生成responses
        responses = self.generate_responses(prompts, **generation_kwargs)
        
        # 构建SFT数据
        sft_data = []
        for item, response in zip(input_data, responses):
            sft_item = item.copy()
            sft_item['teacher_response'] = response
            sft_item['expected_answer'] = response  # SFT使用教师响应作为目标
            sft_data.append(sft_item)
        
        # 保存数据
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in sft_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"SFT数据集已保存到: {output_path}")
        return sft_data


def run_sft_pipeline(
    teacher_model_path: str,
    student_model_path: str,
    train_data_path: str,
    output_dir: str,
    generate_teacher_data: bool = True,
    eval_data_path: Optional[str] = None,
    training_args: Optional[Dict] = None,
    generation_kwargs: Optional[Dict] = None,
    **kwargs
):
    """运行完整的SFT管道"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(str(output_dir / "sft.log"))
    set_seed(kwargs.get('seed', 42))
    
    logger.info(f"开始SFT管道：{teacher_model_path} -> {student_model_path}")
    
    # 1. 生成教师数据（如果需要） - 使用并行生成器
    if generate_teacher_data:
        logger.info("步骤 1/2: 生成教师响应数据 (使用多GPU并行)")
        
        # 临时使用简单单实例生成器避免多进程CUDA问题
        
        # 加载训练数据
        train_data = load_jsonl(train_data_path)
        
        # 默认生成参数
        default_gen_kwargs = {
            "max_new_tokens": 256,  # 优化：减少生成长度
            "temperature": 0.3,      # 优化：降低温度提速
            "top_p": 0.9
        }
        if generation_kwargs:
            default_gen_kwargs.update(generation_kwargs)
        
        # 创建优化配置
        config_overrides = {
            "total_gpus": torch.cuda.device_count(),
            "batch_size": 256,  # 大批次获得最佳性能
            "target_memory_usage": 0.85,
            "max_new_tokens": default_gen_kwargs["max_new_tokens"],
            "temperature": default_gen_kwargs["temperature"],
            "top_p": default_gen_kwargs["top_p"],
            "log_gpu_memory": True,
            "enable_progress_bar": True
        }
        
        # 生成训练数据 - 使用简单生成器
        sft_train_path = output_dir / "sft_train_data.jsonl"
        logger.info(f"使用单实例生成器处理 {len(train_data)} 个训练样本")
        
        generator = TeacherResponseGenerator(
            teacher_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        sft_train_data = generator.generate_sft_dataset(
            train_data,
            str(sft_train_path),
            **default_gen_kwargs
        )
        
        # 生成评估数据
        sft_eval_path = None
        if eval_data_path:
            eval_data = load_jsonl(eval_data_path)
            sft_eval_path = output_dir / "sft_eval_data.jsonl"
            logger.info(f"使用单实例生成器处理 {len(eval_data)} 个评估样本")
            
            sft_eval_data = generator.generate_sft_dataset(
                eval_data,
                str(sft_eval_path),
                **default_gen_kwargs
            )
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
    else:
        sft_train_path = train_data_path
        sft_eval_path = eval_data_path
    
    # 2. SFT训练
    logger.info("步骤 2/2: SFT训练")
    
    distiller = SFTDistiller(
        teacher_model_path=teacher_model_path,
        student_model_path=student_model_path,
        output_dir=str(output_dir),
        skip_teacher_loading=not generate_teacher_data  # 如果不生成数据，跳过教师模型加载
    )
    
    training_history = distiller.distill(
        train_data=str(sft_train_path),
        eval_data=str(sft_eval_path) if sft_eval_path else None,
        training_args=training_args,
        **kwargs
    )
    
    logger.info("SFT管道完成！")
    return training_history