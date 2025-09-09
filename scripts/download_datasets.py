#!/usr/bin/env python3
"""
一键下载和预处理数据集
支持GSM8K, HumanEval, 以及生成AIOps合成数据
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset
from rich.console import Console
from rich.progress import track

console = Console()

def setup_directories():
    """创建必要的目录"""
    dirs = ["data/raw", "data/processed"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    console.print("[green]✓[/green] 目录结构已创建")

def download_gsm8k(sample_size: int = 2000) -> List[Dict]:
    """下载并处理GSM8K数据集"""
    console.print(f"[blue]正在下载GSM8K数据集 (样本数: {sample_size})...[/blue]")
    
    try:
        dataset = load_dataset("gsm8k", "main", cache_dir="data/raw/.cache")
        
        # 保存原始数据
        train_data = list(dataset['train'])
        test_data = list(dataset['test'])
        
        with open("data/raw/gsm8k_train.json", "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open("data/raw/gsm8k_test.json", "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        # 采样训练数据
        if sample_size < len(train_data):
            sampled_train = random.sample(train_data, sample_size)
        else:
            sampled_train = train_data
        
        processed_data = []
        for i, item in enumerate(track(sampled_train, description="处理GSM8K")):
            processed_data.append({
                "id": f"gsm8k_{i:04d}",
                "domain": "math",
                "difficulty": "medium",
                "prompt": f"请解决这个数学问题，要求给出详细的解题步骤：\n\n{item['question']}",
                "expected_answer": item['answer'],
                "metadata": {
                    "source": "gsm8k",
                    "original_index": i
                }
            })
        
        console.print(f"[green]✓[/green] GSM8K处理完成: {len(processed_data)} 条样本")
        return processed_data
        
    except Exception as e:
        console.print(f"[red]✗[/red] GSM8K下载失败: {e}")
        return []

def download_humaneval(sample_size: int = 164) -> List[Dict]:
    """下载并处理HumanEval数据集"""
    console.print(f"[blue]正在下载HumanEval数据集 (样本数: {sample_size})...[/blue]")
    
    try:
        dataset = load_dataset("openai_humaneval", cache_dir="data/raw/.cache")
        
        # 保存原始数据
        test_data = list(dataset['test'])
        with open("data/raw/humaneval.json", "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        processed_data = []
        for i, item in enumerate(track(test_data[:sample_size], description="处理HumanEval")):
            processed_data.append({
                "id": f"humaneval_{item['task_id']}",
                "domain": "code", 
                "difficulty": "hard",
                "prompt": f"请完成以下Python函数，要求代码简洁高效：\n\n```python\n{item['prompt']}\n```\n\n请直接给出完整的函数实现。",
                "expected_answer": item['canonical_solution'],
                "metadata": {
                    "source": "humaneval",
                    "task_id": item['task_id'],
                    "test_cases": item['test'],
                    "entry_point": item['entry_point']
                }
            })
        
        console.print(f"[green]✓[/green] HumanEval处理完成: {len(processed_data)} 条样本")
        return processed_data
        
    except Exception as e:
        console.print(f"[red]✗[/red] HumanEval下载失败: {e}")
        return []

def generate_aiops_synthetic(sample_size: int = 1000) -> List[Dict]:
    """生成AIOps合成数据集"""
    console.print(f"[blue]正在生成AIOps合成数据集 (样本数: {sample_size})...[/blue]")
    
    # 日志模板和根因模板
    templates = [
        {
            "scenario": "数据库连接池耗尽",
            "logs": [
                "2024-09-09 10:23:11 ERROR [ConnectionPool-1] HikariPool timeout after 30s, total=20, active=20, idle=0",
                "2024-09-09 10:23:12 WARN [PaymentService] Database connection failed: pool exhausted",
                "2024-09-09 10:23:13 ERROR [CircuitBreaker] Circuit opened for mysql-primary:3306"
            ],
            "config": {
                "hikari_pool_size": 20,
                "connection_timeout": 30000,
                "max_lifetime": 300000
            },
            "metrics": {
                "cpu_usage": "45%",
                "memory_usage": "78%", 
                "active_connections": 20,
                "qps": 850
            },
            "root_cause": "数据库连接池配置过小，在高并发场景下无法满足需求",
            "fix_plan": [
                "增加连接池大小到50-100",
                "优化慢查询减少连接占用时间",
                "添加连接池监控告警"
            ]
        },
        {
            "scenario": "内存泄漏导致OOM",
            "logs": [
                "2024-09-09 11:15:23 WARN [GC] Full GC took 2.3s, heap: 7.8GB/8GB used", 
                "2024-09-09 11:15:45 ERROR [JVM] OutOfMemoryError: Java heap space",
                "2024-09-09 11:15:46 ERROR [K8s] Pod order-service-7d9f restarts: 3/5"
            ],
            "config": {
                "jvm_heap": "8GB",
                "gc_algorithm": "G1GC",
                "pod_memory_limit": "10GB"
            },
            "metrics": {
                "heap_usage": "95%",
                "gc_frequency": "每30s一次FullGC",
                "restart_count": 3
            },
            "root_cause": "应用存在内存泄漏，对象无法被GC回收导致堆内存耗尽",
            "fix_plan": [
                "使用Memory Profiler定位内存泄漏点",
                "临时增加堆内存到12GB",
                "优化对象生命周期管理"
            ]
        },
        {
            "scenario": "磁盘空间不足",
            "logs": [
                "2024-09-09 14:30:12 ERROR [FileSystem] Disk usage: 98% on /var/log",
                "2024-09-09 14:30:15 WARN [LogRotate] Cannot create new log file: No space left on device", 
                "2024-09-09 14:30:18 ERROR [ES] Elasticsearch cluster RED: disk threshold exceeded"
            ],
            "config": {
                "log_retention": "30天",
                "disk_size": "500GB",
                "es_shard_size": "50GB"
            },
            "metrics": {
                "disk_usage": "98%",
                "log_growth_rate": "10GB/day",
                "es_index_size": "450GB"
            },
            "root_cause": "日志文件和ES索引占用过多磁盘空间，且缺乏自动清理机制",
            "fix_plan": [
                "立即清理7天前的日志文件",
                "配置日志自动轮转和压缩",
                "设置ES索引自动删除策略"
            ]
        },
        {
            "scenario": "网络延迟异常",
            "logs": [
                "2024-09-09 16:45:30 WARN [HTTP] Upstream response time: 5.2s (normal: 200ms)",
                "2024-09-09 16:45:33 ERROR [LoadBalancer] Backend server timeout: api-server-3",
                "2024-09-09 16:45:35 INFO [K8s] Node network plugin showing errors"
            ],
            "config": {
                "upstream_timeout": "3s",
                "keepalive_requests": 100,
                "worker_connections": 1024
            },
            "metrics": {
                "avg_response_time": "5.2s", 
                "p99_latency": "8.5s",
                "network_errors": "12/min"
            },
            "root_cause": "网络插件异常导致跨节点通信延迟增加",
            "fix_plan": [
                "重启异常节点的网络插件",
                "检查网络设备配置",
                "临时将流量切换到正常节点"
            ]
        }
    ]
    
    processed_data = []
    for i in track(range(sample_size), description="生成AIOps数据"):
        template = random.choice(templates)
        
        # 构建提示词
        prompt_parts = [
            f"## 故障场景分析\n\n**系统日志:**",
        ]
        
        for log in template["logs"]:
            prompt_parts.append(f"```\n{log}\n```")
        
        prompt_parts.extend([
            f"\n**配置信息:**",
            f"```json\n{json.dumps(template['config'], ensure_ascii=False, indent=2)}\n```",
            f"\n**监控指标:**",  
            f"```json\n{json.dumps(template['metrics'], ensure_ascii=False, indent=2)}\n```",
            f"\n请分析上述日志和配置，找出最可能的根本原因，并给出具体的修复建议。"
        ])
        
        processed_data.append({
            "id": f"aiops_{i:04d}",
            "domain": "aiops",
            "difficulty": "hard", 
            "prompt": "\n".join(prompt_parts),
            "expected_answer": template["root_cause"],
            "metadata": {
                "source": "synthetic",
                "scenario": template["scenario"],
                "fix_plan": template["fix_plan"]
            }
        })
    
    console.print(f"[green]✓[/green] AIOps数据生成完成: {len(processed_data)} 条样本")
    return processed_data

def save_unified_dataset(data_list: List[List[Dict]], output_path: str):
    """保存统一格式的数据集"""
    all_data = []
    for data in data_list:
        all_data.extend(data)
    
    # 打乱数据
    random.shuffle(all_data)
    
    # 保存为JSONL格式
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    console.print(f"[green]✓[/green] 统一数据集已保存: {output_path}")
    console.print(f"[cyan]总样本数: {len(all_data)}[/cyan]")
    
    # 统计信息
    domain_counts = {}
    for item in all_data:
        domain = item['domain']
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    console.print("[cyan]领域分布:[/cyan]")
    for domain, count in domain_counts.items():
        console.print(f"  {domain}: {count} 条")

def main():
    """主函数"""
    console.print("[bold blue]🚀 开始准备大模型蒸馏数据集[/bold blue]\n")
    
    # 设置随机种子
    random.seed(42)
    
    # 创建目录
    setup_directories()
    
    # 下载和处理数据集
    gsm8k_data = download_gsm8k(sample_size=2000)
    humaneval_data = download_humaneval(sample_size=164) 
    aiops_data = generate_aiops_synthetic(sample_size=1000)
    
    # 保存统一数据集
    if gsm8k_data or humaneval_data or aiops_data:
        save_unified_dataset(
            [gsm8k_data, humaneval_data, aiops_data],
            "data/processed/train_dataset.jsonl"
        )
        
        # 创建小规模评测集 (从每个领域抽取部分作为评测)
        eval_data = []
        if gsm8k_data:
            eval_data.extend(random.sample(gsm8k_data, min(200, len(gsm8k_data))))
        if humaneval_data:
            eval_data.extend(random.sample(humaneval_data, min(50, len(humaneval_data))))
        if aiops_data:
            eval_data.extend(random.sample(aiops_data, min(100, len(aiops_data))))
        
        random.shuffle(eval_data)
        with open("data/processed/eval_dataset.jsonl", "w", encoding="utf-8") as f:
            for item in eval_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        console.print(f"[green]✓[/green] 评测数据集已保存: data/processed/eval_dataset.jsonl ({len(eval_data)} 条)")
    
    console.print("\n[bold green]🎉 数据集准备完成！[/bold green]")
    console.print("[cyan]文件位置:[/cyan]")
    console.print("  📁 训练集: data/processed/train_dataset.jsonl")
    console.print("  📁 评测集: data/processed/eval_dataset.jsonl") 
    console.print("  📁 原始数据: data/raw/")

if __name__ == "__main__":
    main()