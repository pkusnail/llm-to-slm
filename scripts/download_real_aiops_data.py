#!/usr/bin/env python3
"""
下载并处理真实的AIOps数据集

数据来源：
1. LogHub - 真实系统日志数据集合 (github.com/logpai/loghub)
2. HDFS, BGL, Thunderbird等标准AIOps基准数据集
3. 包含异常检测标签的真实生产日志

特点：
- 完全真实数据，非合成生成
- 包含标注的异常和正常案例
- 多种系统类型（分布式文件系统、超级计算机等）
- 适合知识蒸馏训练
"""

import os
import json
import random
import requests
import zipfile
import tarfile
import gzip
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path
import logging
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealAIOpsDatasetDownloader:
    """真实AIOps数据集下载和处理器"""
    
    def __init__(self):
        # LogHub 数据集URLs - 正确的Zenodo链接
        self.datasets = {
            "HDFS": {
                "description": "Hadoop分布式文件系统日志 - 真实生产环境",
                "url": "https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1",
                "size": "1.47GB",
                "log_lines": "11,175,629",
                "labeled": True,
                "type": "distributed_filesystem",
                "format": "zip"
            },
            "BGL": {
                "description": "Blue Gene/L超级计算机日志 - Sandia实验室",
                "url": "https://zenodo.org/records/8196385/files/BGL.zip?download=1", 
                "size": "708MB",
                "log_lines": "4,747,963",
                "labeled": True,
                "type": "supercomputer",
                "format": "zip"
            },
            "Thunderbird": {
                "description": "Thunderbird超级计算机日志 - Sandia实验室",
                "url": "https://zenodo.org/records/8196385/files/Thunderbird.tar.gz?download=1",
                "size": "29.6GB", 
                "log_lines": "211,212,192",
                "labeled": True,
                "type": "supercomputer",
                "format": "tar.gz"
            },
            "Hadoop": {
                "description": "Hadoop集群日志 - 46核5机器集群",
                "url": "https://zenodo.org/records/8196385/files/Hadoop.zip?download=1",
                "size": "16.3MB",
                "log_lines": "394,308", 
                "labeled": True,
                "type": "big_data_cluster",
                "format": "zip"
            },
            "OpenStack": {
                "description": "OpenStack基础设施日志",
                "url": "https://zenodo.org/records/8196385/files/OpenStack.tar.gz?download=1",
                "size": "58.6MB",
                "log_lines": "207,820",
                "labeled": True,
                "type": "cloud_infrastructure",
                "format": "tar.gz"
            }
        }
        
        # AIOps问答模板 - 基于真实系统运维场景
        self.qa_templates = [
            {
                "category": "anomaly_detection",
                "template": "在{system_type}系统中，发现以下日志模式：\n\n{log_pattern}\n\n这是正常行为还是异常？请分析原因并提供处理建议。",
                "response_template": "**状态分析**: {status}\n\n**分析原因**: {analysis}\n\n**处理建议**:\n{recommendations}"
            },
            {
                "category": "root_cause_analysis", 
                "template": "系统{system_type}出现异常，相关日志如下：\n\n{error_logs}\n\n请进行根因分析并提供解决方案。",
                "response_template": "**根因分析**:\n{root_cause}\n\n**影响范围**: {impact}\n\n**解决方案**:\n{solution}\n\n**预防措施**: {prevention}"
            },
            {
                "category": "performance_diagnosis",
                "template": "在{system_type}中观察到性能下降，关键日志信息：\n\n{performance_logs}\n\n请诊断性能瓶颈并给出优化建议。",
                "response_template": "**性能诊断**: {diagnosis}\n\n**瓶颈识别**: {bottleneck}\n\n**优化建议**:\n{optimization}\n\n**预期效果**: {expected_improvement}"
            }
        ]
    
    def download_dataset(self, dataset_name: str, output_dir: str = "data/real_aiops") -> bool:
        """下载指定数据集"""
        if dataset_name not in self.datasets:
            logger.error(f"未知数据集: {dataset_name}")
            return False
        
        dataset_info = self.datasets[dataset_name]
        output_path = Path(output_dir) / dataset_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        url = dataset_info["url"]
        file_format = dataset_info.get("format", "zip")
        if file_format == "tar.gz":
            archive_file = output_path / f"{dataset_name}.tar.gz"
        else:
            archive_file = output_path / f"{dataset_name}.zip"
        
        logger.info(f"下载 {dataset_name} 数据集...")
        logger.info(f"描述: {dataset_info['description']}")
        logger.info(f"大小: {dataset_info['size']}, 日志行数: {dataset_info['log_lines']}")
        
        try:
            # 检查是否已下载
            if archive_file.exists():
                logger.info(f"{dataset_name} 已存在，跳过下载")
                return True
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(archive_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"{dataset_name} 下载完成: {archive_file}")
            return True
            
        except Exception as e:
            logger.error(f"下载 {dataset_name} 失败: {e}")
            return False
    
    def extract_and_parse_logs(self, dataset_name: str, output_dir: str = "data/real_aiops", sample_limit: int = 2000) -> List[Dict[str, Any]]:
        """解压并解析日志文件，提取样本"""
        dataset_path = Path(output_dir) / dataset_name
        dataset_info = self.datasets[dataset_name]
        file_format = dataset_info.get("format", "zip")
        
        if file_format == "tar.gz":
            archive_file = dataset_path / f"{dataset_name}.tar.gz"
        else:
            archive_file = dataset_path / f"{dataset_name}.zip"
        
        if not archive_file.exists():
            logger.error(f"数据集文件不存在: {archive_file}")
            return []
        
        samples = []
        
        try:
            # 根据文件格式选择解压方式
            if file_format == "tar.gz":
                with tarfile.open(archive_file, 'r:gz') as tar_ref:
                    tar_ref.extractall(dataset_path)
            else:
                with zipfile.ZipFile(archive_file, 'r') as zip_ref:
                    zip_ref.extractall(dataset_path)
            
            # 查找日志文件
            log_files = list(dataset_path.rglob("*.log")) + list(dataset_path.rglob("*.txt"))
            label_files = list(dataset_path.rglob("*label*")) + list(dataset_path.rglob("*abnormal*"))
            
            if not log_files:
                logger.warning(f"在 {dataset_path} 中未找到日志文件")
                return []
            
            # 解析主日志文件
            main_log_file = log_files[0]  # 取第一个日志文件
            logger.info(f"解析日志文件: {main_log_file}")
            
            # 读取标签（如果存在）
            labels = {}
            if label_files:
                label_file = label_files[0]
                logger.info(f"读取标签文件: {label_file}")
                try:
                    with open(label_file, 'r') as f:
                        for line_no, line in enumerate(f):
                            if line.strip():
                                labels[line_no] = line.strip()
                except Exception as e:
                    logger.warning(f"读取标签文件失败: {e}")
            
            # 解析日志内容
            log_entries = []
            try:
                with open(main_log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_no, line in enumerate(f):
                        if len(log_entries) >= sample_limit:
                            break
                        
                        line = line.strip()
                        if line:
                            log_entries.append({
                                "line_no": line_no,
                                "content": line,
                                "label": labels.get(line_no, "Normal")
                            })
            
            except Exception as e:
                logger.error(f"解析日志文件失败: {e}")
                return []
            
            # 随机采样并生成QA对
            sampled_entries = random.sample(log_entries, min(sample_limit, len(log_entries)))
            
            for i, entry in enumerate(sampled_entries):
                qa_pair = self.generate_qa_pair(
                    dataset_name, 
                    entry["content"], 
                    entry["label"],
                    self.datasets[dataset_name]
                )
                
                samples.append({
                    "id": f"{dataset_name.lower()}_real_{i+1}",
                    "domain": f"aiops_{dataset_name.lower()}",
                    "prompt": qa_pair["prompt"],
                    "expected_answer": qa_pair["answer"],
                    "metadata": {
                        "dataset": dataset_name,
                        "source": "real_logs",
                        "system_type": self.datasets[dataset_name]["type"],
                        "original_label": entry["label"],
                        "line_no": entry["line_no"]
                    }
                })
            
            logger.info(f"从 {dataset_name} 生成了 {len(samples)} 个样本")
            return samples
            
        except Exception as e:
            logger.error(f"处理数据集 {dataset_name} 失败: {e}")
            return []
    
    def generate_qa_pair(self, dataset_name: str, log_content: str, label: str, dataset_info: Dict) -> Dict[str, str]:
        """基于真实日志生成问答对"""
        
        # 根据标签判断异常类型
        is_anomaly = label.lower() not in ["normal", "0", "-", ""]
        system_type = dataset_info["type"].replace("_", " ")
        
        # 选择合适的模板
        if is_anomaly:
            if "error" in log_content.lower() or "fail" in log_content.lower():
                template = random.choice([t for t in self.qa_templates if t["category"] == "root_cause_analysis"])
            else:
                template = random.choice([t for t in self.qa_templates if t["category"] == "anomaly_detection"])
        else:
            template = random.choice([t for t in self.qa_templates if t["category"] == "performance_diagnosis"])
        
        # 生成问题
        if template["category"] == "anomaly_detection":
            prompt = template["template"].format(
                system_type=system_type,
                log_pattern=f"```\n{log_content}\n```"
            )
        elif template["category"] == "root_cause_analysis":
            prompt = template["template"].format(
                system_type=system_type,
                error_logs=f"```\n{log_content}\n```"
            )
        else:  # performance_diagnosis
            prompt = template["template"].format(
                system_type=system_type,
                performance_logs=f"```\n{log_content}\n```"
            )
        
        # 生成回答
        if is_anomaly:
            answer = self._generate_anomaly_response(log_content, system_type, dataset_name)
        else:
            answer = self._generate_normal_response(log_content, system_type, dataset_name)
        
        return {
            "prompt": prompt,
            "answer": answer
        }
    
    def _generate_anomaly_response(self, log_content: str, system_type: str, dataset_name: str) -> str:
        """生成异常情况的回答"""
        
        # 分析日志内容中的关键词
        keywords = {
            "error": "系统错误",
            "fail": "操作失败", 
            "timeout": "超时问题",
            "memory": "内存问题",
            "disk": "磁盘问题",
            "network": "网络问题",
            "crash": "系统崩溃"
        }
        
        detected_issues = []
        for keyword, description in keywords.items():
            if keyword.lower() in log_content.lower():
                detected_issues.append(description)
        
        if not detected_issues:
            detected_issues = ["系统异常行为"]
        
        # 根据数据集类型提供特定建议
        system_specific_advice = {
            "distributed_filesystem": [
                "检查HDFS DataNode状态",
                "验证网络连通性", 
                "检查磁盘空间和权限",
                "重新平衡数据块分布"
            ],
            "supercomputer": [
                "检查计算节点资源状态",
                "验证作业调度配置",
                "监控系统负载和内存使用",
                "检查硬件故障日志"
            ],
            "big_data_cluster": [
                "检查集群资源分配",
                "验证MapReduce任务配置", 
                "监控网络和I/O性能",
                "优化数据本地性"
            ],
            "cloud_infrastructure": [
                "检查云服务状态",
                "验证API调用限制",
                "监控资源配额使用",
                "检查服务间依赖关系"
            ]
        }
        
        advice_key = self.datasets[dataset_name]["type"]
        recommendations = system_specific_advice.get(advice_key, ["进行系统健康检查", "查看相关监控指标"])
        
        return f"""**状态分析**: 异常 - 检测到{', '.join(detected_issues)}

**分析原因**: 
基于{system_type}日志分析，发现以下异常模式：
- 日志内容显示系统运行异常
- 可能涉及：{', '.join(detected_issues)}
- 需要立即关注和处理

**处理建议**:
1. {recommendations[0]}
2. {recommendations[1]}
3. {recommendations[2] if len(recommendations) > 2 else '执行系统诊断检查'}
4. 如问题持续，考虑{recommendations[3] if len(recommendations) > 3 else '联系技术支持'}

**监控重点**:
- 持续观察类似日志模式
- 设置相关告警规则
- 记录问题处理过程以供后续参考"""
    
    def _generate_normal_response(self, log_content: str, system_type: str, dataset_name: str) -> str:
        """生成正常情况的回答"""
        
        return f"""**状态分析**: 正常 - 系统运行正常

**分析原因**: 
基于{system_type}日志分析：
- 日志内容显示系统正常运行
- 无异常错误或警告信息
- 符合系统预期行为模式

**性能诊断**: 
- 当前操作执行成功
- 系统响应正常
- 无性能瓶颈迹象

**优化建议**:
1. 继续监控系统运行状态
2. 保持当前配置和运维策略
3. 定期检查系统性能指标
4. 建立基线用于异常检测

**预期效果**:
- 系统将持续稳定运行
- 性能指标保持在正常范围
- 用户体验不受影响"""

    def download_and_process_all(self, output_dir: str = "data/real_aiops", target_samples: int = 10000) -> Dict[str, str]:
        """下载并处理所有数据集"""
        
        # 按数据集大小分配样本数量
        dataset_priorities = {
            "HDFS": 3000,      # 最大且最常用
            "Hadoop": 2000,    # 适中大小，Hadoop相关
            "BGL": 2000,       # 超级计算机，高质量
            "OpenStack": 2000, # 云基础设施，现代化
            "Thunderbird": 1000 # 太大，少量采样
        }
        
        all_samples = []
        successful_downloads = []
        
        for dataset_name, sample_count in dataset_priorities.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"处理数据集: {dataset_name}")
            logger.info(f"目标样本数: {sample_count}")
            
            # 下载数据集
            if self.download_dataset(dataset_name, output_dir):
                # 解析并生成样本
                samples = self.extract_and_parse_logs(dataset_name, output_dir, sample_count)
                all_samples.extend(samples)
                successful_downloads.append(dataset_name)
                
                if len(all_samples) >= target_samples:
                    logger.info(f"已达到目标样本数 {target_samples}")
                    break
        
        # 如果样本不足，截取到目标数量
        if len(all_samples) > target_samples:
            all_samples = random.sample(all_samples, target_samples)
        
        # 随机打乱
        random.shuffle(all_samples)
        
        # 分割训练和验证集
        train_size = int(len(all_samples) * 0.8)
        train_samples = all_samples[:train_size]
        eval_samples = all_samples[train_size:]
        
        # 保存数据集
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_file = output_path / "real_aiops_train_data.jsonl"
        eval_file = output_path / "real_aiops_eval_data.jsonl"
        
        # 写入训练数据
        with open(train_file, 'w', encoding='utf-8') as f:
            for sample in train_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # 写入验证数据
        with open(eval_file, 'w', encoding='utf-8') as f:
            for sample in eval_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"\n{'='*50}")
        logger.info("真实AIOps数据集处理完成！")
        logger.info(f"成功下载数据集: {', '.join(successful_downloads)}")
        logger.info(f"总样本数: {len(all_samples)}")
        logger.info(f"训练集: {train_file} ({len(train_samples)} 样本)")
        logger.info(f"验证集: {eval_file} ({len(eval_samples)} 样本)")
        
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
    downloader = RealAIOpsDatasetDownloader()
    
    # 设置随机种子确保可重现性
    random.seed(42)
    
    logger.info("开始下载和处理真实AIOps数据集...")
    logger.info("数据来源: LogHub (github.com/logpai/loghub)")
    logger.info("数据特点: 完全真实的生产环境日志，包含异常标注")
    
    result = downloader.download_and_process_all(
        output_dir="data/real_aiops",
        target_samples=10000
    )
    
    logger.info("\n处理结果:")
    for key, value in result.items():
        logger.info(f"  {key}: {value}")

if __name__ == "__main__":
    main()