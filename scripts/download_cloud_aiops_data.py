#!/usr/bin/env python3
"""
快速下载多个云原生和微服务AIOps数据集
专注于AWS、Kubernetes、Golang相关的真实数据

数据来源：
1. LogHub - 较小的云基础设施数据集 (OpenStack, Hadoop)
2. Kubernetes安全数据集 - 专门的K8s日志
3. 微服务依赖图数据集 - 微服务架构相关
4. 并行下载多个来源以减少时间
"""

import os
import json
import random
import requests
import zipfile
import tarfile
import asyncio
import aiohttp
import concurrent.futures
from typing import Dict, List, Any, Tuple
from pathlib import Path
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudAIOpsDatasetDownloader:
    """云原生AIOps数据集快速下载器"""
    
    def __init__(self):
        # 精选的云原生和微服务数据集 - 小尺寸，快速下载
        self.datasets = {
            # LogHub 较小数据集
            "OpenStack": {
                "description": "OpenStack云基础设施日志 - 真实云环境",
                "url": "https://zenodo.org/records/8196385/files/OpenStack.tar.gz?download=1",
                "size": "58.6MB",
                "log_lines": "207,820",
                "type": "cloud_infrastructure",
                "format": "tar.gz",
                "priority": 1
            },
            "Hadoop": {
                "description": "Hadoop大数据集群日志 - 分布式计算",
                "url": "https://zenodo.org/records/8196385/files/Hadoop.zip?download=1",
                "size": "16.3MB", 
                "log_lines": "394,308",
                "type": "big_data_cluster",
                "format": "zip",
                "priority": 1
            },
            
            # 更快的替代数据源
            "KubernetesSecurityData": {
                "description": "K8s安全检测数据集 - 网络流量和异常",
                "url": "https://github.com/yigitsever/kubernetes-dataset/archive/main.zip",
                "size": "~10MB",
                "log_lines": "~50,000",
                "type": "k8s_security",
                "format": "zip",
                "priority": 2
            },
            "MicroserviceDepGraph": {
                "description": "微服务依赖图数据集 - 20个微服务项目",
                "url": "https://github.com/clowee/MicroserviceDataset/archive/main.zip", 
                "size": "~5MB",
                "log_lines": "~10,000",
                "type": "microservices",
                "format": "zip",
                "priority": 2
            }
        }
        
        # 云原生AIOps问答模板
        self.cloud_qa_templates = [
            {
                "category": "k8s_troubleshooting",
                "template": "Kubernetes集群中Pod状态异常，日志显示：\n\n{log_content}\n\n请诊断问题并提供K8s原生的解决方案。",
                "response_template": "**K8s诊断**:\n{k8s_diagnosis}\n\n**解决方案**:\n```bash\n{kubectl_commands}\n```\n\n**预防措施**: {prevention}"
            },
            {
                "category": "cloud_performance",
                "template": "云基础设施性能监控发现异常：\n\n{performance_data}\n\n请基于云原生架构分析性能瓶颈并给出优化建议。",
                "response_template": "**性能分析**: {performance_analysis}\n\n**云原生优化**:\n{cloud_optimization}\n\n**预期效果**: {expected_results}"
            },
            {
                "category": "microservice_issues", 
                "template": "微服务架构中服务间通信异常，相关日志：\n\n{service_logs}\n\n请分析服务依赖关系并提供微服务治理建议。",
                "response_template": "**服务分析**: {service_analysis}\n\n**依赖关系问题**: {dependency_issues}\n\n**微服务治理建议**:\n{governance_recommendations}"
            }
        ]
    
    async def download_dataset_async(self, session: aiohttp.ClientSession, dataset_name: str, output_dir: str) -> bool:
        """异步下载数据集"""
        dataset_info = self.datasets[dataset_name]
        output_path = Path(output_dir) / dataset_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_format = dataset_info.get("format", "zip")
        if file_format == "tar.gz":
            archive_file = output_path / f"{dataset_name}.tar.gz"
        else:
            archive_file = output_path / f"{dataset_name}.zip"
        
        # 检查是否已存在
        if archive_file.exists():
            logger.info(f"✅ {dataset_name} 已存在，跳过下载")
            return True
            
        url = dataset_info["url"]
        logger.info(f"🔄 开始下载 {dataset_name} ({dataset_info['size']})")
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    with open(archive_file, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    
                    logger.info(f"✅ {dataset_name} 下载完成")
                    return True
                else:
                    logger.error(f"❌ {dataset_name} 下载失败: HTTP {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ {dataset_name} 下载异常: {e}")
            return False
    
    async def download_all_async(self, output_dir: str = "data/cloud_aiops") -> List[str]:
        """并行下载所有数据集"""
        connector = aiohttp.TCPConnector(limit=4)  # 限制并发连接数
        timeout = aiohttp.ClientTimeout(total=600)  # 10分钟超时
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []
            
            # 按优先级排序
            sorted_datasets = sorted(self.datasets.items(), key=lambda x: x[1].get("priority", 3))
            
            for dataset_name, _ in sorted_datasets:
                task = self.download_dataset_async(session, dataset_name, output_dir)
                tasks.append((dataset_name, task))
            
            # 并行执行下载
            successful_downloads = []
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            for (dataset_name, _), result in zip(tasks, results):
                if isinstance(result, bool) and result:
                    successful_downloads.append(dataset_name)
                elif isinstance(result, Exception):
                    logger.error(f"❌ {dataset_name} 下载异常: {result}")
            
            return successful_downloads
    
    def extract_logs_fast(self, dataset_name: str, output_dir: str = "data/cloud_aiops", sample_limit: int = 1000) -> List[Dict[str, Any]]:
        """快速解压和处理日志"""
        dataset_path = Path(output_dir) / dataset_name
        dataset_info = self.datasets[dataset_name]
        file_format = dataset_info.get("format", "zip")
        
        if file_format == "tar.gz":
            archive_file = dataset_path / f"{dataset_name}.tar.gz"
        else:
            archive_file = dataset_path / f"{dataset_name}.zip"
        
        if not archive_file.exists():
            logger.warning(f"⚠️  数据集文件不存在: {archive_file}")
            return []
        
        samples = []
        
        try:
            # 解压缩
            if file_format == "tar.gz":
                with tarfile.open(archive_file, 'r:gz') as tar_ref:
                    tar_ref.extractall(dataset_path)
            else:
                with zipfile.ZipFile(archive_file, 'r') as zip_ref:
                    zip_ref.extractall(dataset_path)
            
            # 查找文件
            log_files = list(dataset_path.rglob("*.log")) + list(dataset_path.rglob("*.txt")) + list(dataset_path.rglob("*.csv"))
            json_files = list(dataset_path.rglob("*.json")) + list(dataset_path.rglob("*.jsonl"))
            
            if not (log_files or json_files):
                logger.warning(f"⚠️  在 {dataset_path} 中未找到可处理的文件")
                return []
            
            # 处理日志文件
            processed_count = 0
            for log_file in log_files[:3]:  # 限制处理文件数量
                if processed_count >= sample_limit:
                    break
                    
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        for i, line in enumerate(f):
                            if processed_count >= sample_limit:
                                break
                                
                            line = line.strip()
                            if line:
                                qa_pair = self.generate_cloud_qa(dataset_name, line, dataset_info)
                                samples.append({
                                    "id": f"{dataset_name.lower()}_cloud_{processed_count+1}",
                                    "domain": f"cloud_aiops_{dataset_name.lower()}",
                                    "prompt": qa_pair["prompt"],
                                    "expected_answer": qa_pair["answer"],
                                    "metadata": {
                                        "dataset": dataset_name,
                                        "source_file": log_file.name,
                                        "type": dataset_info["type"],
                                        "cloud_native": True
                                    }
                                })
                                processed_count += 1
                except Exception as e:
                    logger.warning(f"⚠️  处理文件 {log_file} 失败: {e}")
            
            # 处理JSON文件
            for json_file in json_files[:2]:
                if processed_count >= sample_limit:
                    break
                    
                try:
                    with open(json_file, 'r', encoding='utf-8', errors='ignore') as f:
                        if json_file.suffix == '.jsonl':
                            for line in f:
                                if processed_count >= sample_limit:
                                    break
                                try:
                                    data = json.loads(line)
                                    text_content = str(data)[:500]  # 限制长度
                                    qa_pair = self.generate_cloud_qa(dataset_name, text_content, dataset_info)
                                    samples.append({
                                        "id": f"{dataset_name.lower()}_json_{processed_count+1}",
                                        "domain": f"cloud_aiops_{dataset_name.lower()}",
                                        "prompt": qa_pair["prompt"],
                                        "expected_answer": qa_pair["answer"],
                                        "metadata": {
                                            "dataset": dataset_name,
                                            "source_file": json_file.name,
                                            "type": dataset_info["type"],
                                            "cloud_native": True
                                        }
                                    })
                                    processed_count += 1
                                except:
                                    continue
                        else:
                            data = json.load(f)
                            if isinstance(data, list):
                                for item in data[:100]:  # 限制处理数量
                                    if processed_count >= sample_limit:
                                        break
                                    text_content = str(item)[:500]
                                    qa_pair = self.generate_cloud_qa(dataset_name, text_content, dataset_info)
                                    samples.append({
                                        "id": f"{dataset_name.lower()}_json_{processed_count+1}",
                                        "domain": f"cloud_aiops_{dataset_name.lower()}",
                                        "prompt": qa_pair["prompt"],
                                        "expected_answer": qa_pair["answer"],
                                        "metadata": {
                                            "dataset": dataset_name,
                                            "source_file": json_file.name,
                                            "type": dataset_info["type"],
                                            "cloud_native": True
                                        }
                                    })
                                    processed_count += 1
                except Exception as e:
                    logger.warning(f"⚠️  处理JSON文件 {json_file} 失败: {e}")
            
            logger.info(f"✅ {dataset_name} 处理完成: {len(samples)} 样本")
            return samples
            
        except Exception as e:
            logger.error(f"❌ 处理数据集 {dataset_name} 失败: {e}")
            return []
    
    def generate_cloud_qa(self, dataset_name: str, content: str, dataset_info: Dict) -> Dict[str, str]:
        """生成云原生场景的问答对"""
        
        dataset_type = dataset_info["type"]
        
        # 选择合适的模板
        if dataset_type == "k8s_security":
            template = random.choice([t for t in self.cloud_qa_templates if t["category"] == "k8s_troubleshooting"])
            prompt = template["template"].format(log_content=f"```\n{content[:400]}\n```")
            answer = self._generate_k8s_answer(content, dataset_name)
        elif dataset_type == "microservices":
            template = random.choice([t for t in self.cloud_qa_templates if t["category"] == "microservice_issues"])
            prompt = template["template"].format(service_logs=f"```\n{content[:400]}\n```")
            answer = self._generate_microservice_answer(content, dataset_name)
        else:  # cloud_infrastructure, big_data_cluster
            template = random.choice([t for t in self.cloud_qa_templates if t["category"] == "cloud_performance"])
            prompt = template["template"].format(performance_data=f"```\n{content[:400]}\n```")
            answer = self._generate_cloud_performance_answer(content, dataset_name)
        
        return {"prompt": prompt, "answer": answer}
    
    def _generate_k8s_answer(self, content: str, dataset_name: str) -> str:
        """生成K8s相关问题的答案"""
        return f"""**K8s诊断**:
基于{dataset_name}数据分析，检测到Kubernetes集群异常行为。

**解决方案**:
```bash
# 检查Pod状态
kubectl get pods -o wide
kubectl describe pod <pod-name>

# 查看集群资源
kubectl top nodes
kubectl top pods

# 检查事件
kubectl get events --sort-by='.lastTimestamp'

# 如果是资源问题，调整资源限制
kubectl edit deployment <deployment-name>
```

**预防措施**: 
- 设置合适的资源请求和限制
- 配置健康检查探针
- 实施Pod Disruption Budget
- 建立监控告警机制"""
    
    def _generate_microservice_answer(self, content: str, dataset_name: str) -> str:
        """生成微服务相关问题的答案"""
        return f"""**服务分析**: 
基于{dataset_name}微服务架构分析，发现服务间通信存在异常。

**依赖关系问题**: 
- 服务调用链路异常
- 可能存在循环依赖
- 服务发现配置问题

**微服务治理建议**:
1. **服务网格**: 实施Istio或Linkerd进行流量管理
2. **熔断器**: 使用Circuit Breaker模式防止级联失败
3. **监控**: 部署分布式追踪系统(如Jaeger)
4. **限流**: 实施API网关进行流量控制
5. **健康检查**: 配置服务健康检查端点"""
    
    def _generate_cloud_performance_answer(self, content: str, dataset_name: str) -> str:
        """生成云性能相关问题的答案"""
        return f"""**性能分析**: 
基于{dataset_name}云基础设施日志，识别出性能瓶颈。

**云原生优化**:
1. **自动扩缩容**: 配置HPA和VPA实现动态资源调整
2. **缓存策略**: 实施Redis集群提升数据访问速度
3. **负载均衡**: 优化Ingress控制器配置
4. **资源调度**: 使用Node Affinity优化Pod调度
5. **存储优化**: 选择合适的StorageClass和持久卷

**预期效果**:
- 响应延迟降低30-50%
- 系统吞吐量提升2-3倍  
- 资源利用率优化20-40%
- 成本节省15-25%"""
    
    def process_all_datasets(self, output_dir: str = "data/cloud_aiops", target_samples: int = 10000) -> Dict[str, str]:
        """处理所有数据集的主函数"""
        
        logger.info("🚀 开始云原生AIOps数据集快速下载...")
        logger.info("特点: AWS、K8s、微服务相关真实数据，快速并行下载")
        
        # 异步下载所有数据集
        successful_downloads = asyncio.run(self.download_all_async(output_dir))
        
        if not successful_downloads:
            logger.error("❌ 没有成功下载任何数据集")
            return {}
        
        logger.info(f"✅ 成功下载: {', '.join(successful_downloads)}")
        
        # 并行处理数据集
        all_samples = []
        samples_per_dataset = target_samples // len(successful_downloads)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for dataset_name in successful_downloads:
                future = executor.submit(
                    self.extract_logs_fast, 
                    dataset_name, 
                    output_dir, 
                    samples_per_dataset
                )
                futures.append((dataset_name, future))
            
            for dataset_name, future in futures:
                try:
                    samples = future.result(timeout=120)  # 2分钟超时
                    all_samples.extend(samples)
                    logger.info(f"✅ {dataset_name} 处理完成: {len(samples)} 样本")
                except Exception as e:
                    logger.error(f"❌ {dataset_name} 处理失败: {e}")
        
        # 随机采样到目标数量
        if len(all_samples) > target_samples:
            all_samples = random.sample(all_samples, target_samples)
        
        random.shuffle(all_samples)
        
        # 分割数据集
        train_size = int(len(all_samples) * 0.8)
        train_samples = all_samples[:train_size]
        eval_samples = all_samples[train_size:]
        
        # 保存数据
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_file = output_path / "cloud_aiops_train_data.jsonl"
        eval_file = output_path / "cloud_aiops_eval_data.jsonl"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            for sample in train_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        with open(eval_file, 'w', encoding='utf-8') as f:
            for sample in eval_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"🎉 云原生AIOps数据集处理完成！")
        logger.info(f"📊 总样本数: {len(all_samples)}")
        logger.info(f"📈 训练集: {len(train_samples)} 样本")
        logger.info(f"📉 验证集: {len(eval_samples)} 样本")
        logger.info(f"💾 数据源: {', '.join(successful_downloads)}")
        
        return {
            "train_file": str(train_file),
            "eval_file": str(eval_file),
            "total_samples": len(all_samples),
            "train_samples": len(train_samples),
            "eval_samples": len(eval_samples),
            "datasets_used": successful_downloads
        }

def main():
    """主函数"""
    downloader = CloudAIOpsDatasetDownloader()
    
    random.seed(42)
    
    result = downloader.process_all_datasets(
        output_dir="data/cloud_aiops",
        target_samples=10000
    )
    
    logger.info("\n📋 处理结果:")
    for key, value in result.items():
        logger.info(f"   {key}: {value}")

if __name__ == "__main__":
    main()