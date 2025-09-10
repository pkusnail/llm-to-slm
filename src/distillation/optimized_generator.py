"""
高性能双实例并行生成器
专为开源项目设计，支持灵活配置和GPU资源最大化利用
"""
import os
import json
import torch
import logging
import multiprocessing as mp
from typing import List, Dict, Any, Optional
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """生成配置类 - 开源用户可以轻松调整的所有参数"""
    
    # === 基础模型配置 ===
    model_path: str = ""
    torch_dtype: str = "bfloat16"  # "bfloat16", "float16", "float32"
    trust_remote_code: bool = True
    
    # === GPU和内存配置 ===
    total_gpus: int = 8  # 总GPU数量
    gpus_per_instance: int = 4  # 每个实例使用的GPU数量
    memory_per_gpu_gb: int = 40  # 每GPU显存（GB）
    target_memory_usage: float = 0.85  # 目标显存使用率（85%）
    
    # === 并行实例配置 ===
    num_instances: int = 2  # 并行实例数量 (total_gpus / gpus_per_instance)
    
    # === 批次配置（自动计算或手动设置） ===
    batch_size: Optional[int] = None  # None表示自动计算最优批次大小
    min_batch_size: int = 32  # 最小批次大小
    max_batch_size: int = 512  # 最大批次大小
    
    # === 生成参数 ===
    max_new_tokens: int = 256
    temperature: float = 0.3
    top_p: float = 0.9
    do_sample: bool = True
    
    # === 性能优化 ===
    use_flash_attention: bool = False  # 是否启用Flash Attention
    use_torch_compile: bool = False  # 是否使用torch.compile优化
    prefetch_factor: int = 2  # 数据预加载因子
    
    # === 监控和调试 ===
    enable_progress_bar: bool = True
    log_gpu_memory: bool = True
    save_intermediate_results: bool = False
    
    def __post_init__(self):
        """配置验证和自动计算"""
        # 验证GPU配置
        available_gpus = torch.cuda.device_count()
        if self.total_gpus > available_gpus:
            logger.warning(f"请求{self.total_gpus}个GPU，但只有{available_gpus}个可用")
            self.total_gpus = available_gpus
        
        # 自动计算实例数量
        self.num_instances = self.total_gpus // self.gpus_per_instance
        if self.num_instances < 1:
            raise ValueError(f"GPU数量不足：需要至少{self.gpus_per_instance}个GPU")
        
        # 自动计算最优批次大小（如果未指定）
        if self.batch_size is None:
            self.batch_size = self._calculate_optimal_batch_size()
        
        logger.info(f"配置验证完成：{self.num_instances}个实例，每实例{self.gpus_per_instance}GPU，批次大小{self.batch_size}")
    
    def _calculate_optimal_batch_size(self) -> int:
        """
        自动计算最优批次大小
        
        基于以下因素：
        1. GPU显存容量
        2. 模型大小估算
        3. 目标显存使用率
        """
        # 简单的启发式算法
        # 对于30B模型，在A100 40GB上，安全的批次大小
        if "30B" in self.model_path or "32B" in self.model_path:
            base_batch_size = 128
        elif "13B" in self.model_path or "14B" in self.model_path:
            base_batch_size = 256
        elif "7B" in self.model_path or "8B" in self.model_path:
            base_batch_size = 512
        else:
            base_batch_size = 128  # 默认值
        
        # 根据显存调整
        memory_factor = self.memory_per_gpu_gb / 40  # A100 40GB为基准
        adjusted_batch_size = int(base_batch_size * memory_factor * self.target_memory_usage)
        
        # 限制在合理范围内
        return max(self.min_batch_size, min(adjusted_batch_size, self.max_batch_size))
    
    def get_device_map(self, instance_id: int):
        """获取指定实例的设备映射"""
        start_gpu = instance_id * self.gpus_per_instance
        end_gpu = start_gpu + self.gpus_per_instance
        gpu_list = list(range(start_gpu, min(end_gpu, self.total_gpus)))
        
        # 返回设备映射配置
        if len(gpu_list) == 1:
            return gpu_list[0]  # 单GPU返回int
        else:
            # 多GPU时返回"auto"，让transformers自动分配
            return "auto"


class OptimizedDualInstanceGenerator:
    """优化的双实例并行生成器"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.instances = []
        
        logger.info(f"初始化双实例生成器：{config.num_instances}个实例")
        logger.info(f"目标批次大小：{config.batch_size}")
        logger.info(f"预计显存使用：{config.target_memory_usage*100:.1f}%")
    
    def generate_parallel(
        self,
        input_data: List[Dict],
        output_path: str,
        **kwargs
    ) -> List[Dict]:
        """双实例并行生成 - 支持增量保存和断点续传"""
        
        total_samples = len(input_data)
        logger.info(f"开始双实例并行生成：{total_samples}个样本")
        
        # 检查是否有未完成的生成任务（断点续传）
        partial_files = []
        output_dir = Path(output_path).parent
        output_name = Path(output_path).stem
        
        for i in range(self.config.num_instances):
            partial_file = output_dir / f"{output_name}_instance_{i}_partial.jsonl"
            partial_files.append(partial_file)
        
        # 检查已完成的数据
        completed_samples = set()
        for partial_file in partial_files:
            if partial_file.exists():
                logger.info(f"发现未完成任务，加载: {partial_file}")
                try:
                    with open(partial_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                item = json.loads(line)
                                completed_samples.add(item.get('id', ''))
                except Exception as e:
                    logger.warning(f"加载部分文件失败: {e}")
        
        # 过滤已完成的样本
        if completed_samples:
            remaining_data = [item for item in input_data if item.get('id', '') not in completed_samples]
            logger.info(f"断点续传：跳过{len(completed_samples)}个已完成样本，剩余{len(remaining_data)}个")
            input_data = remaining_data
            total_samples = len(remaining_data)
        
        if total_samples == 0:
            logger.info("所有样本已完成，合并最终文件...")
            return self._merge_partial_files(partial_files, output_path)
        
        # 数据分割：每个实例处理一半数据
        chunk_size = total_samples // self.config.num_instances
        data_chunks = []
        
        for i in range(self.config.num_instances):
            start_idx = i * chunk_size
            if i == self.config.num_instances - 1:
                end_idx = total_samples  # 最后一个实例处理剩余所有数据
            else:
                end_idx = (i + 1) * chunk_size
            
            chunk = input_data[start_idx:end_idx]
            data_chunks.append(chunk)
        
        logger.info(f"数据分割完成：{[len(chunk) for chunk in data_chunks]}")
        
        # 使用进程池并行执行
        # 设置spawn方法避免CUDA问题
        ctx = mp.get_context('spawn')
        
        with ProcessPoolExecutor(
            max_workers=self.config.num_instances,
            mp_context=ctx
        ) as executor:
            # 提交任务（传递部分文件路径用于增量保存）
            futures = []
            for instance_id, chunk in enumerate(data_chunks):
                if len(chunk) > 0:
                    future = executor.submit(
                        self._generate_on_instance_incremental,
                        instance_id,
                        chunk,
                        str(partial_files[instance_id]),
                        kwargs
                    )
                    futures.append((instance_id, future))
            
            # 收集结果（但数据已经增量保存了）
            completed_instances = []
            for instance_id, future in futures:
                try:
                    success = future.result()
                    completed_instances.append(instance_id)
                    logger.info(f"实例{instance_id}完成并保存")
                except Exception as e:
                    logger.error(f"实例{instance_id}执行失败：{e}")
        
        logger.info(f"双实例并行生成完成，{len(completed_instances)}个实例成功")
        
        # 合并所有部分文件到最终输出
        return self._merge_partial_files(partial_files, output_path)
    
    def _merge_partial_files(self, partial_files: List[Path], output_path: str) -> List[Dict]:
        """合并部分文件到最终输出"""
        
        all_results = []
        
        logger.info("开始合并部分文件...")
        for i, partial_file in enumerate(partial_files):
            if partial_file.exists():
                try:
                    with open(partial_file, 'r', encoding='utf-8') as f:
                        count = 0
                        for line in f:
                            if line.strip():
                                item = json.loads(line)
                                all_results.append(item)
                                count += 1
                        logger.info(f"从实例{i}文件加载{count}个结果")
                except Exception as e:
                    logger.error(f"合并实例{i}文件失败: {e}")
        
        # 写入最终文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in all_results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"最终文件已保存: {output_path} ({len(all_results)}个样本)")
        
        # 清理部分文件
        for partial_file in partial_files:
            if partial_file.exists():
                try:
                    partial_file.unlink()
                    logger.debug(f"清理部分文件: {partial_file}")
                except Exception as e:
                    logger.warning(f"清理文件失败: {e}")
        
        return all_results
    
    def _generate_on_instance_incremental(
        self,
        instance_id: int,
        data_chunk: List[Dict],
        partial_file_path: str,
        generation_kwargs: Dict
    ) -> bool:
        """在指定实例上执行增量生成，实时保存到磁盘"""
        
        try:
            # 设置GPU设备
            device_map = self.config.get_device_map(instance_id)
            logger.info(f"实例{instance_id}：使用GPU {device_map}，处理{len(data_chunk)}个样本")
            
            # 加载模型
            torch_dtype = getattr(torch, self.config.torch_dtype)
            
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=self.config.trust_remote_code
            )
            model.eval()
            
            # 使用torch.compile优化（如果启用）
            if self.config.use_torch_compile:
                model = torch.compile(model)
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info(f"实例{instance_id}：模型加载完成")
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(partial_file_path), exist_ok=True)
            
            # 打开文件用于增量写入（追加模式）
            batch_size = self.config.batch_size
            total_batches = (len(data_chunk) + batch_size - 1) // batch_size
            
            logger.info(f"实例{instance_id}：开始增量生成，{total_batches}个批次，实时保存到 {partial_file_path}")
            
            processed_count = 0
            
            with open(partial_file_path, 'a', encoding='utf-8') as f:
                for batch_idx in range(0, len(data_chunk), batch_size):
                    batch_data = data_chunk[batch_idx:batch_idx + batch_size]
                    prompts = [item['prompt'] for item in batch_data]
                    
                    # 应用聊天模板
                    batch_texts = []
                    for prompt in prompts:
                        if hasattr(tokenizer, 'apply_chat_template'):
                            messages = [{"role": "user", "content": prompt}]
                            text = tokenizer.apply_chat_template(
                                messages,
                                add_generation_prompt=True,
                                tokenize=False
                            )
                        else:
                            text = f"Human: {prompt}\n\nAssistant:"
                        batch_texts.append(text)
                    
                    # 分词
                    inputs = tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=2048
                    )
                    
                    # 移动到正确的设备
                    if isinstance(device_map, int):
                        inputs = {k: v.to(f"cuda:{device_map}") for k, v in inputs.items()}
                    elif isinstance(device_map, list) and len(device_map) == 1:
                        inputs = {k: v.to(f"cuda:{device_map[0]}") for k, v in inputs.items()}
                    # 多GPU情况下让model自动处理设备
                    
                    # 生成
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=self.config.max_new_tokens,
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                            do_sample=self.config.do_sample,
                            pad_token_id=tokenizer.eos_token_id,
                            **generation_kwargs
                        )
                    
                    # 解码并立即写入文件
                    batch_results = []
                    for j, output in enumerate(outputs):
                        input_length = inputs['input_ids'][j].shape[0]
                        response_ids = output[input_length:]
                        response = tokenizer.decode(response_ids, skip_special_tokens=True)
                        
                        # 构建结果
                        result_item = batch_data[j].copy()
                        result_item['teacher_response'] = response.strip()
                        result_item['expected_answer'] = response.strip()
                        result_item['instance_id'] = instance_id
                        result_item['batch_id'] = batch_idx // batch_size
                        
                        batch_results.append(result_item)
                        
                        # 立即写入文件（每个样本一行）
                        f.write(json.dumps(result_item, ensure_ascii=False) + '\n')
                        f.flush()  # 强制刷新到磁盘
                    
                    processed_count += len(batch_results)
                    
                    # 清理显存
                    torch.cuda.empty_cache()
                    
                    # 进度报告
                    current_batch = batch_idx // batch_size + 1
                    if current_batch % 2 == 0 or current_batch == total_batches:
                        logger.info(f"实例{instance_id}：完成批次 {current_batch}/{total_batches} ({processed_count}/{len(data_chunk)} 样本已保存)")
                        
                        # 记录GPU内存使用（如果启用）
                        if self.config.log_gpu_memory:
                            if isinstance(device_map, int):
                                memory_used = torch.cuda.memory_allocated(device_map) / 1024**3
                                memory_total = torch.cuda.get_device_properties(device_map).total_memory / 1024**3
                                logger.info(f"实例{instance_id} GPU{device_map}：{memory_used:.1f}GB/{memory_total:.1f}GB ({memory_used/memory_total*100:.1f}%)")
            
            logger.info(f"实例{instance_id}：生成完成，共{processed_count}个结果已保存到 {partial_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"实例{instance_id}生成失败：{e}")
            return False
    
    def _generate_on_instance(
        self,
        instance_id: int,
        data_chunk: List[Dict],
        generation_kwargs: Dict
    ) -> List[Dict]:
        """在指定实例上执行生成"""
        
        try:
            # 设置GPU设备
            device_map = self.config.get_device_map(instance_id)
            logger.info(f"实例{instance_id}：使用GPU {device_map}，处理{len(data_chunk)}个样本")
            
            # 加载模型
            torch_dtype = getattr(torch, self.config.torch_dtype)
            
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=self.config.trust_remote_code
            )
            model.eval()
            
            # 使用torch.compile优化（如果启用）
            if self.config.use_torch_compile:
                model = torch.compile(model)
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info(f"实例{instance_id}：模型加载完成")
            
            # 批量生成
            results = []
            batch_size = self.config.batch_size
            total_batches = (len(data_chunk) + batch_size - 1) // batch_size
            
            logger.info(f"实例{instance_id}：开始生成，{total_batches}个批次")
            
            for batch_idx in range(0, len(data_chunk), batch_size):
                batch_data = data_chunk[batch_idx:batch_idx + batch_size]
                prompts = [item['prompt'] for item in batch_data]
                
                # 应用聊天模板
                batch_texts = []
                for prompt in prompts:
                    if hasattr(tokenizer, 'apply_chat_template'):
                        messages = [{"role": "user", "content": prompt}]
                        text = tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=False
                        )
                    else:
                        text = f"Human: {prompt}\n\nAssistant:"
                    batch_texts.append(text)
                
                # 分词
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                )
                
                # 移动到正确的设备
                if isinstance(device_map, int):
                    inputs = {k: v.to(f"cuda:{device_map}") for k, v in inputs.items()}
                elif isinstance(device_map, list) and len(device_map) == 1:
                    inputs = {k: v.to(f"cuda:{device_map[0]}") for k, v in inputs.items()}
                # 多GPU情况下让model自动处理设备
                
                # 生成
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=self.config.do_sample,
                        pad_token_id=tokenizer.eos_token_id,
                        **generation_kwargs
                    )
                
                # 解码
                for j, output in enumerate(outputs):
                    input_length = inputs['input_ids'][j].shape[0]
                    response_ids = output[input_length:]
                    response = tokenizer.decode(response_ids, skip_special_tokens=True)
                    
                    # 构建结果
                    result_item = batch_data[j].copy()
                    result_item['teacher_response'] = response.strip()
                    result_item['expected_answer'] = response.strip()
                    result_item['instance_id'] = instance_id
                    results.append(result_item)
                
                # 清理显存
                torch.cuda.empty_cache()
                
                # 进度报告
                current_batch = batch_idx // batch_size + 1
                if current_batch % 5 == 0 or current_batch == total_batches:
                    logger.info(f"实例{instance_id}：完成批次 {current_batch}/{total_batches}")
                    
                    # 记录GPU内存使用（如果启用）
                    if self.config.log_gpu_memory:
                        if isinstance(device_map, int):
                            memory_used = torch.cuda.memory_allocated(device_map) / 1024**3
                            memory_total = torch.cuda.get_device_properties(device_map).total_memory / 1024**3
                            logger.info(f"实例{instance_id} GPU{device_map}：{memory_used:.1f}GB/{memory_total:.1f}GB ({memory_used/memory_total*100:.1f}%)")
            
            logger.info(f"实例{instance_id}：生成完成，共{len(results)}个结果")
            return results
            
        except Exception as e:
            logger.error(f"实例{instance_id}生成失败：{e}")
            return []


def create_config_from_args(**kwargs) -> GenerationConfig:
    """从参数创建配置对象，便于开源用户使用"""
    return GenerationConfig(**kwargs)


def run_optimized_dual_generation(
    model_path: str,
    input_data: List[Dict],
    output_path: str,
    config_overrides: Optional[Dict] = None,
    **generation_kwargs
) -> List[Dict]:
    """
    运行优化的双实例生成
    
    Args:
        model_path: 模型路径
        input_data: 输入数据
        output_path: 输出路径
        config_overrides: 配置覆盖参数
        **generation_kwargs: 生成参数
    
    Returns:
        生成的结果列表
    """
    
    # 创建配置
    base_config = {
        "model_path": model_path,
        "total_gpus": torch.cuda.device_count(),
    }
    
    if config_overrides:
        base_config.update(config_overrides)
    
    config = GenerationConfig(**base_config)
    
    # 创建生成器并执行
    generator = OptimizedDualInstanceGenerator(config)
    return generator.generate_parallel(input_data, output_path, **generation_kwargs)


# 开源项目配置模板
EXAMPLE_CONFIGS = {
    "8x_A100_40GB_30B_model": {
        "total_gpus": 8,
        "gpus_per_instance": 4,
        "num_instances": 2,
        "batch_size": 256,  # 预计使用~32GB显存/GPU
        "target_memory_usage": 0.85,
        "use_flash_attention": False,  # 根据模型支持情况
    },
    
    "4x_A100_40GB_13B_model": {
        "total_gpus": 4,
        "gpus_per_instance": 2,
        "num_instances": 2,
        "batch_size": 512,
        "target_memory_usage": 0.8,
    },
    
    "8x_H100_80GB_70B_model": {
        "total_gpus": 8,
        "gpus_per_instance": 4,
        "num_instances": 2,
        "batch_size": 128,
        "memory_per_gpu_gb": 80,
        "target_memory_usage": 0.9,
        "use_flash_attention": True,
    }
}


if __name__ == "__main__":
    # 使用示例
    config = GenerationConfig(
        model_path="Qwen/Qwen3-30B-A3B-Instruct-2507",
        total_gpus=8,
        batch_size=256
    )
    
    generator = OptimizedDualInstanceGenerator(config)
    print("双实例生成器初始化完成")
    print(f"配置：{config.num_instances}个实例，批次大小{config.batch_size}")