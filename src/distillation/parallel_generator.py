"""
多GPU并行Teacher响应生成器
充分利用8×A100算力
"""
import os
import json
import torch
import logging
from typing import List, Dict, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
# 设置multiprocessing启动方法为spawn以支持CUDA
mp.set_start_method('spawn', force=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

class ParallelTeacherGenerator:
    """多GPU并行教师响应生成器"""
    
    def __init__(
        self,
        model_path: str,
        num_gpus: int = 8,
        max_new_tokens: int = 256,  # 减少生成长度
        temperature: float = 0.3,   # 降低温度提速
        batch_size_per_gpu: int = 8  # 每GPU批次大小
    ):
        self.model_path = model_path
        self.num_gpus = min(num_gpus, torch.cuda.device_count())
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size_per_gpu = batch_size_per_gpu
        self.total_batch_size = batch_size_per_gpu * self.num_gpus
        
        logger.info(f"初始化并行生成器: {self.num_gpus}×GPU, 总批次大小: {self.total_batch_size}")
    
    def generate_parallel(
        self,
        prompts: List[str],
        output_path: str,
        **kwargs
    ) -> List[Dict]:
        """多GPU并行生成"""
        
        # 将数据分块给每个GPU
        chunk_size = len(prompts) // self.num_gpus
        prompt_chunks = []
        
        for i in range(self.num_gpus):
            start_idx = i * chunk_size
            if i == self.num_gpus - 1:  # 最后一个GPU处理剩余
                end_idx = len(prompts)
            else:
                end_idx = (i + 1) * chunk_size
            prompt_chunks.append(prompts[start_idx:end_idx])
        
        logger.info(f"数据分块完成: {[len(chunk) for chunk in prompt_chunks]}")
        
        # 使用多进程启动每个GPU的生成任务
        with mp.Pool(processes=self.num_gpus) as pool:
            results = []
            for gpu_id, chunk in enumerate(prompt_chunks):
                if len(chunk) > 0:  # 只处理非空chunk
                    result = pool.apply_async(
                        self._generate_on_gpu,
                        args=(gpu_id, chunk, kwargs)
                    )
                    results.append(result)
            
            # 收集所有结果
            all_responses = []
            for result in results:
                gpu_responses = result.get()
                all_responses.extend(gpu_responses)
        
        logger.info(f"并行生成完成: {len(all_responses)} 条响应")
        
        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in all_responses:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        return all_responses
    
    def _generate_on_gpu(self, gpu_id: int, prompts: List[str], kwargs: Dict) -> List[Dict]:
        """在指定GPU上生成响应"""
        
        try:
            # 设置CUDA设备
            torch.cuda.set_device(gpu_id)
            device = f"cuda:{gpu_id}"
            
            logger.info(f"GPU {gpu_id}: 开始加载模型，处理 {len(prompts)} 个prompt")
            
            # 加载模型到指定GPU
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True
            )
            model.eval()
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info(f"GPU {gpu_id}: 模型加载完成，开始生成")
            
            responses = []
            
            # 批量处理
            for i in range(0, len(prompts), self.batch_size_per_gpu):
                batch_prompts = prompts[i:i+self.batch_size_per_gpu]
                
                # 构建消息格式
                batch_messages = []
                for prompt in batch_prompts:
                    messages = [{"role": "user", "content": prompt}]
                    batch_messages.append(messages)
                
                # 应用聊天模板
                batch_texts = []
                for messages in batch_messages:
                    if hasattr(tokenizer, 'apply_chat_template'):
                        text = tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=False
                        )
                    else:
                        text = f"Human: {messages[0]['content']}\\n\\nAssistant:"
                    batch_texts.append(text)
                
                # 分词
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                ).to(device)
                
                # 生成
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        **kwargs
                    )
                
                # 解码响应
                for j, output in enumerate(outputs):
                    input_length = inputs['input_ids'][j].shape[0]
                    response_ids = output[input_length:]
                    response = tokenizer.decode(response_ids, skip_special_tokens=True)
                    responses.append({
                        "prompt": batch_prompts[j],
                        "response": response.strip(),
                        "gpu_id": gpu_id
                    })
                
                # 清理GPU缓存
                torch.cuda.empty_cache()
                
                if (i // self.batch_size_per_gpu + 1) % 10 == 0:
                    logger.info(f"GPU {gpu_id}: 完成 {i + len(batch_prompts)}/{len(prompts)} 样本")
            
            logger.info(f"GPU {gpu_id}: 生成完成，共 {len(responses)} 条响应")
            return responses
            
        except Exception as e:
            logger.error(f"GPU {gpu_id} 生成失败: {e}")
            return []


class SuperFastTeacherGenerator:
    """超高速单GPU批量生成器（优化版）"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        
        # 加载模型 - 使用model parallel跨多GPU
        logger.info("加载Teacher模型到多GPU...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",  # 自动跨GPU分布
            trust_remote_code=True
        )
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_batch_optimized(
        self,
        prompts: List[str],
        batch_size: int = 32,  # 大批次
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        **kwargs
    ) -> List[str]:
        """优化的批量生成"""
        
        responses = []
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        
        logger.info(f"开始超速生成: {len(prompts)}样本, {total_batches}批次, 批次大小{batch_size}")
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # 快速模板处理
            batch_texts = []
            for prompt in batch_prompts:
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    messages = [{"role": "user", "content": prompt}]
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False
                    )
                else:
                    text = f"Human: {prompt}\\n\\nAssistant:"
                batch_texts.append(text)
            
            # 快速分词
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1536  # 稍微减少输入长度
            )
            
            # 确保输入在正确设备
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # 快速解码
            for j, output in enumerate(outputs):
                input_length = inputs['input_ids'][j].shape[0]
                response_ids = output[input_length:]
                response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                responses.append(response.strip())
            
            # 清理缓存
            torch.cuda.empty_cache()
            
            current_batch = i // batch_size + 1
            logger.info(f"完成批次 {current_batch}/{total_batches}")
        
        return responses


def run_optimized_generation(
    model_path: str,
    input_data: List[Dict],
    output_path: str,
    use_parallel: bool = False
) -> List[Dict]:
    """运行优化的生成"""
    
    if use_parallel:
        # 多GPU数据并行
        generator = ParallelTeacherGenerator(model_path, num_gpus=8)
        prompts = [item['prompt'] for item in input_data]
        results = generator.generate_parallel(prompts, output_path)
        
        # 组合结果
        sft_data = []
        for item, result in zip(input_data, results):
            sft_item = item.copy()
            sft_item['teacher_response'] = result['response']
            sft_item['expected_answer'] = result['response']
            sft_data.append(sft_item)
    
    else:
        # 优化的单模型生成
        generator = SuperFastTeacherGenerator(model_path)
        prompts = [item['prompt'] for item in input_data]
        responses = generator.generate_batch_optimized(
            prompts,
            batch_size=64,  # 进一步增加批次大小
            max_new_tokens=256,
            temperature=0.3
        )
        
        # 组合结果
        sft_data = []
        for item, response in zip(input_data, responses):
            sft_item = item.copy()
            sft_item['teacher_response'] = response
            sft_item['expected_answer'] = response
            sft_data.append(sft_item)
        
        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in sft_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\\n')
    
    return sft_data