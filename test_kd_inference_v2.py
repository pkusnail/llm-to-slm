#!/usr/bin/env python3
"""
知识蒸馏模型推理测试脚本 - 多方案支持版本
对比原始8B模型和KD训练后模型的推理能力

使用方法:
python test_kd_inference_v2.py --mode quick    # 快速验证 (10样本)
python test_kd_inference_v2.py --mode medium   # 中等规模 (50样本)  
python test_kd_inference_v2.py --mode full     # 完整评估 (350样本)
"""
import torch
import json
import time
import random
import argparse
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUOptimizedTester:
    def __init__(self, mode="quick", temperature=0.7):
        self.mode = mode
        self.temperature = temperature
        self.tokenizer = None
        self.test_samples = []
        
        # GPU优化配置
        self.gpu_count = torch.cuda.device_count()
        self.batch_sizes = {
            "quick": min(8, self.gpu_count * 2),     # 激进batch：每GPU 2个样本
            "medium": min(16, self.gpu_count * 2),   # 中等batch：每GPU 2个样本
            "full": min(24, self.gpu_count * 3)      # 最大batch：每GPU 3个样本
        }
        
        logger.info(f"检测到 {self.gpu_count} 个GPU")
        logger.info(f"模式: {mode}, 批次大小: {self.batch_sizes[mode]}, 温度: {temperature}")
        
    def load_eval_data(self, data_path="outputs/experiment/qwen3_30b_to_8b_ultrabatch_512/sft/sft_eval_data_clean.jsonl"):
        """加载评估数据"""
        logger.info(f"加载评估数据: {data_path}")
        
        all_samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_samples.append(json.loads(line))
        
        # 根据模式选择样本
        if self.mode == "quick":
            # 快速模式：精选10个代表性样本
            selected_samples = self._select_representative_samples(all_samples, 10)
            logger.info("快速模式：选择10个代表性样本")
        elif self.mode == "medium":
            # 中等模式：随机采样50个
            selected_samples = random.sample(all_samples, min(50, len(all_samples)))
            logger.info("中等模式：随机采样50个样本")
        else:
            # 完整模式：全部样本
            selected_samples = all_samples
            logger.info(f"完整模式：使用全部{len(all_samples)}个样本")
            
        self.test_samples = selected_samples
        return len(selected_samples)
    
    def _select_representative_samples(self, all_samples, count=10):
        """选择代表性样本"""
        # 按领域分类
        by_domain = {}
        for sample in all_samples:
            domain = sample.get('domain', 'unknown')
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(sample)
        
        # 均匀选择
        selected = []
        domains = list(by_domain.keys())
        per_domain = count // len(domains)
        remainder = count % len(domains)
        
        for i, domain in enumerate(domains):
            n = per_domain + (1 if i < remainder else 0)
            selected.extend(random.sample(by_domain[domain], min(n, len(by_domain[domain]))))
        
        return selected[:count]
    
    def load_model_optimized(self, model_path, adapter_path=None, gpu_ids=None):
        """加载模型并优化GPU分配"""
        logger.info(f"加载模型: {model_path}")
        
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 智能GPU分配 - 使用指定的GPU
        device_map = {}
        if gpu_ids:
            # 手动指定GPU分配
            layers_per_gpu = 5  # 8B模型大约36层，分配到指定GPU
            gpu_idx = 0
            device_map["model.embed_tokens"] = gpu_ids[gpu_idx]
            
            total_layers = 36  # Qwen3-8B层数
            for i in range(total_layers):
                if i > 0 and i % layers_per_gpu == 0:
                    gpu_idx = (gpu_idx + 1) % len(gpu_ids)
                device_map[f"model.layers.{i}"] = gpu_ids[gpu_idx]
                
            device_map["model.norm"] = gpu_ids[-1]
            device_map["lm_head"] = gpu_ids[-1]
        else:
            device_map = "auto"
        
        # 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True
        )
        
        # 如果有adapter，加载LoRA
        if adapter_path:
            logger.info(f"加载LoRA适配器: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
        
        return model
    
    def batch_generate(self, model, prompts_batch, max_length=512, temperature=0.7):
        """批量生成回复"""
        # 批量编码
        inputs = self.tokenizer(
            prompts_batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=1024
        ).to(next(model.parameters()).device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
                use_cache=True
            )
        
        # 解码批量结果
        responses = []
        input_length = inputs.input_ids.shape[1]
        
        for i, output in enumerate(outputs):
            generated = output[input_length:]
            response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            responses.append(response)
        
        return responses
    
    def evaluate_model_batch(self, model, model_name):
        """批量评估模型"""
        logger.info(f"开始批量评估: {model_name}")
        
        batch_size = self.batch_sizes[self.mode]
        all_results = []
        total_time = 0
        
        # 准备批次
        batches = []
        for i in range(0, len(self.test_samples), batch_size):
            batch = self.test_samples[i:i+batch_size]
            batches.append(batch)
        
        logger.info(f"总计 {len(batches)} 个批次，每批次 {batch_size} 个样本")
        
        for batch_idx, batch_samples in enumerate(batches):
            start_time = time.time()
            
            # 提取prompts
            prompts = [sample["prompt"] for sample in batch_samples]
            
            # 批量推理
            responses = self.batch_generate(model, prompts, temperature=self.temperature)
            
            batch_time = time.time() - start_time
            total_time += batch_time
            
            # 保存结果
            for sample, response in zip(batch_samples, responses):
                result = {
                    "id": sample["id"],
                    "domain": sample.get("domain", "unknown"),
                    "difficulty": sample.get("difficulty", "unknown"),
                    "prompt": sample["prompt"],
                    f"{model_name}_response": response,
                    f"{model_name}_time": batch_time / len(batch_samples),  # 平均时间
                    "expected_answer": sample.get("original_answer", ""),
                    "teacher_response": sample.get("teacher_response", "")
                }
                all_results.append(result)
            
            logger.info(f"批次 {batch_idx+1}/{len(batches)} 完成，用时 {batch_time:.2f}s")
        
        logger.info(f"{model_name} 评估完成，总用时 {total_time:.2f}s")
        return all_results, total_time
    
    def run_comparison_test(self):
        """运行对比测试"""
        logger.info(f"=== 开始 {self.mode} 模式对比测试 ===")
        
        # 加载数据
        sample_count = self.load_eval_data()
        logger.info(f"加载了 {sample_count} 个测试样本")
        
        all_results = []
        timing_info = {}
        
        try:
            # 测试原始模型
            logger.info("=== 加载原始8B模型 ===")
            original_model = self.load_model_optimized(
                "Qwen/Qwen3-8B", 
                gpu_ids=[0, 1, 2, 3]  # 使用前4个GPU
            )
            
            original_results, original_time = self.evaluate_model_batch(
                original_model, "original"
            )
            timing_info["original_time"] = original_time
            
            # 清理内存
            del original_model
            torch.cuda.empty_cache()
            
            # 测试KD模型
            logger.info("=== 加载KD训练后模型 ===")
            kd_model = self.load_model_optimized(
                "Qwen/Qwen3-8B",
                adapter_path="outputs/experiment/gpu_optimized_kd_fixed/gpu_optimized_kd_20250910_170252/final_model",
                gpu_ids=[4, 5, 6, 7]  # 使用后4个GPU
            )
            
            kd_results, kd_time = self.evaluate_model_batch(
                kd_model, "kd"
            )
            timing_info["kd_time"] = kd_time
            
            # 合并结果
            for orig, kd in zip(original_results, kd_results):
                combined = {**orig}
                combined.update({
                    k: v for k, v in kd.items() 
                    if k.startswith("kd_")
                })
                all_results.append(combined)
                
        except Exception as e:
            logger.error(f"测试过程中出现错误: {e}")
            raise
        
        # 保存结果
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"kd_comparison_{self.mode}_{timestamp}.json"
        
        final_results = {
            "config": {
                "mode": self.mode,
                "sample_count": sample_count,
                "batch_size": self.batch_sizes[self.mode],
                "gpu_count": self.gpu_count,
                "temperature": self.temperature
            },
            "timing": timing_info,
            "results": all_results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        # 打印总结
        self.print_summary(final_results, filename)
        
        return final_results
    
    def print_summary(self, results, filename):
        """打印测试总结"""
        config = results["config"]
        timing = results["timing"]
        
        print("\n" + "="*60)
        print(f"🎯 KD模型对比测试完成 - {config['mode'].upper()}模式")
        print("="*60)
        print(f"📊 测试规模: {config['sample_count']} 个样本")
        print(f"⚡ 批次大小: {config['batch_size']}")
        print(f"🖥️  使用GPU: {config['gpu_count']} 个")
        print()
        print("⏱️  性能统计:")
        print(f"   原始8B模型: {timing.get('original_time', 0):.2f}s")
        print(f"   KD训练后模型: {timing.get('kd_time', 0):.2f}s")
        print(f"   总用时: {sum(timing.values()):.2f}s")
        print()
        print(f"💾 详细结果已保存到: {filename}")
        print("="*60)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="KD模型推理对比测试")
    parser.add_argument(
        "--mode", 
        choices=["quick", "medium", "full"],
        default="quick",
        help="测试模式: quick(10样本,~3分钟), medium(50样本,~15分钟), full(350样本,~2小时)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="推理温度 (0.1=保守, 0.7=平衡, 1.2=创意). 默认0.7"
    )
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    print(f"\n🚀 启动KD模型对比测试 - {args.mode.upper()}模式")
    
    # 设置随机种子
    random.seed(42)
    torch.manual_seed(42)
    
    try:
        tester = GPUOptimizedTester(mode=args.mode, temperature=args.temperature)
        results = tester.run_comparison_test()
        
        print(f"\n✅ 测试成功完成！")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        print(f"\n❌ 测试失败: {e}")
        raise

if __name__ == "__main__":
    main()