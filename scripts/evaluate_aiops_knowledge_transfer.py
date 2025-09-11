#!/usr/bin/env python3
"""
AIOps Knowledge Transfer Evaluation Script
评估AIOps领域知识强化效果的综合脚本

测试维度：
1. 领域特化性能 (Domain Specialization)
2. 知识密度提升 (Knowledge Intensification) 
3. 对比基线模型 (Baseline Comparison)
"""

import json
import torch
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AIOpsKnowledgeEvaluator:
    """AIOps领域知识转移评估器"""
    
    def __init__(self):
        self.aiops_test_cases = self._load_aiops_test_cases()
        self.general_test_cases = self._load_general_test_cases()
        
    def _load_aiops_test_cases(self) -> List[Dict]:
        """加载AIOps专业测试用例"""
        return [
            # Kubernetes相关
            {
                "category": "kubernetes",
                "question": "如何诊断Kubernetes Pod处于CrashLoopBackOff状态的根因？请提供详细的排查步骤。",
                "keywords": ["kubectl describe", "kubectl logs", "liveness probe", "readiness probe", "OOMKilled"],
                "complexity": "expert"
            },
            {
                "category": "kubernetes", 
                "question": "解释Kubernetes中Service Mesh的工作原理，以及它如何解决微服务间的通信问题。",
                "keywords": ["istio", "envoy", "sidecar", "circuit breaker", "load balancing"],
                "complexity": "advanced"
            },
            
            # 监控告警相关
            {
                "category": "monitoring",
                "question": "设计一个完整的微服务监控体系，包括指标收集、告警策略和故障定位。",
                "keywords": ["prometheus", "grafana", "alertmanager", "SLI", "SLO", "error budget"],
                "complexity": "expert"
            },
            {
                "category": "monitoring",
                "question": "什么是Golden Signals？请解释它们在AIOps中的重要性。",
                "keywords": ["latency", "traffic", "errors", "saturation", "observability"],
                "complexity": "intermediate"
            },
            
            # 云原生架构
            {
                "category": "cloud_native",
                "question": "在AWS环境下，如何实现多区域的容灾备份策略？",
                "keywords": ["multi-az", "cross-region", "RDS", "S3", "Route53", "disaster recovery"],
                "complexity": "expert"
            },
            {
                "category": "cloud_native",
                "question": "比较容器化部署和传统虚拟机部署在运维方面的差异。",
                "keywords": ["docker", "orchestration", "resource utilization", "scaling", "maintenance"],
                "complexity": "intermediate"
            },
            
            # 故障处理
            {
                "category": "incident_response",
                "question": "描述一个典型的P0级别生产故障的应急响应流程。",
                "keywords": ["incident commander", "war room", "rollback", "post-mortem", "RCA"],
                "complexity": "expert" 
            },
            {
                "category": "incident_response",
                "question": "如何设计有效的告警降噪机制，避免告警疲劳？", 
                "keywords": ["alert fatigue", "deduplication", "correlation", "threshold tuning"],
                "complexity": "advanced"
            },
            
            # 性能优化
            {
                "category": "performance",
                "question": "Java微服务出现内存泄漏时的诊断和解决步骤？",
                "keywords": ["heap dump", "GC analysis", "memory profiling", "JVM tuning"],
                "complexity": "expert"
            },
            {
                "category": "performance", 
                "question": "数据库查询慢的常见原因和优化方法有哪些？",
                "keywords": ["query plan", "index optimization", "connection pooling", "caching"],
                "complexity": "intermediate"
            }
        ]
    
    def _load_general_test_cases(self) -> List[Dict]:
        """加载通用知识测试用例"""
        return [
            {
                "category": "general_programming",
                "question": "请解释什么是设计模式，并举例说明单例模式的实现。",
                "keywords": ["design pattern", "singleton", "thread safety"],
                "complexity": "intermediate"
            },
            {
                "category": "general_ai",
                "question": "什么是机器学习中的过拟合现象？如何预防？",
                "keywords": ["overfitting", "regularization", "cross-validation"],
                "complexity": "intermediate"
            },
            {
                "category": "general_math",
                "question": "计算一个边长为5的正方形的面积和周长。",
                "keywords": ["area", "perimeter", "calculation"],
                "complexity": "basic"
            }
        ]
    
    def evaluate_model(self, model, tokenizer, model_name: str) -> Dict:
        """评估单个模型的AIOps知识水平"""
        logger.info(f"评估模型: {model_name}")
        
        results = {
            "model_name": model_name,
            "aiops_performance": {},
            "general_performance": {},
            "knowledge_specialization_score": 0.0,
            "detailed_results": []
        }
        
        # 评估AIOps知识
        aiops_scores = []
        for test_case in self.aiops_test_cases:
            score, response = self._evaluate_single_case(model, tokenizer, test_case)
            aiops_scores.append(score)
            
            results["detailed_results"].append({
                "type": "aiops",
                "category": test_case["category"],
                "question": test_case["question"],
                "score": score,
                "response": response,
                "expected_keywords": test_case["keywords"]
            })
        
        # 评估通用知识
        general_scores = []
        for test_case in self.general_test_cases:
            score, response = self._evaluate_single_case(model, tokenizer, test_case)
            general_scores.append(score)
            
            results["detailed_results"].append({
                "type": "general", 
                "category": test_case["category"],
                "question": test_case["question"],
                "score": score,
                "response": response,
                "expected_keywords": test_case["keywords"]
            })
        
        # 计算综合得分
        results["aiops_performance"] = {
            "average_score": np.mean(aiops_scores),
            "max_score": np.max(aiops_scores),
            "min_score": np.min(aiops_scores),
            "std_score": np.std(aiops_scores)
        }
        
        results["general_performance"] = {
            "average_score": np.mean(general_scores),
            "max_score": np.max(general_scores), 
            "min_score": np.min(general_scores),
            "std_score": np.std(general_scores)
        }
        
        # 计算知识特化得分 (AIOps相对于General的提升)
        aiops_avg = np.mean(aiops_scores)
        general_avg = np.mean(general_scores)
        
        if general_avg > 0:
            results["knowledge_specialization_score"] = (aiops_avg - general_avg) / general_avg
        else:
            results["knowledge_specialization_score"] = 0.0
            
        return results
    
    def _evaluate_single_case(self, model, tokenizer, test_case: Dict) -> Tuple[float, str]:
        """评估单个测试用例"""
        prompt = self._format_prompt(test_case["question"])
        
        # 生成回答
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # 评估回答质量
        score = self._score_response(response, test_case["keywords"], test_case["complexity"])
        
        return score, response
    
    def _format_prompt(self, question: str) -> str:
        """格式化prompt"""
        return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    def _score_response(self, response: str, expected_keywords: List[str], complexity: str) -> float:
        """评估回答质量 (0-1分数)"""
        if not response or len(response.strip()) < 20:
            return 0.0
        
        response_lower = response.lower()
        
        # 关键词匹配得分
        keyword_matches = sum(1 for keyword in expected_keywords 
                            if keyword.lower() in response_lower)
        keyword_score = keyword_matches / len(expected_keywords)
        
        # 长度和详细程度得分
        length_score = min(len(response) / 500, 1.0)  # 最多500字符满分
        
        # 复杂度权重
        complexity_weights = {"basic": 1.0, "intermediate": 1.2, "advanced": 1.5, "expert": 2.0}
        complexity_weight = complexity_weights.get(complexity, 1.0)
        
        # 综合得分
        base_score = (keyword_score * 0.7 + length_score * 0.3)
        final_score = base_score * complexity_weight
        
        return min(final_score, 1.0)  # 限制在0-1范围内
    
    def compare_models(self, results_list: List[Dict]) -> Dict:
        """比较多个模型的结果"""
        if len(results_list) < 2:
            return {"error": "需要至少2个模型进行比较"}
        
        comparison = {
            "model_ranking": [],
            "aiops_knowledge_comparison": {},
            "specialization_analysis": {},
            "improvement_analysis": {}
        }
        
        # 按AIOps性能排序
        sorted_results = sorted(results_list, 
                              key=lambda x: x["aiops_performance"]["average_score"], 
                              reverse=True)
        
        comparison["model_ranking"] = [
            {
                "rank": i+1,
                "model_name": result["model_name"],
                "aiops_score": result["aiops_performance"]["average_score"],
                "general_score": result["general_performance"]["average_score"],
                "specialization_score": result["knowledge_specialization_score"]
            }
            for i, result in enumerate(sorted_results)
        ]
        
        # 找到baseline (通常是原始8B模型)
        baseline_result = None
        distilled_result = None
        
        for result in results_list:
            if "baseline" in result["model_name"].lower() or "8B" in result["model_name"]:
                baseline_result = result
            elif "distilled" in result["model_name"].lower() or "kd" in result["model_name"].lower():
                distilled_result = result
        
        if baseline_result and distilled_result:
            # 计算知识强化效果
            aiops_improvement = (distilled_result["aiops_performance"]["average_score"] - 
                               baseline_result["aiops_performance"]["average_score"])
            general_change = (distilled_result["general_performance"]["average_score"] - 
                            baseline_result["general_performance"]["average_score"])
            
            comparison["improvement_analysis"] = {
                "aiops_knowledge_gain": aiops_improvement,
                "general_knowledge_change": general_change,
                "net_specialization": aiops_improvement - general_change,
                "relative_aiops_improvement": aiops_improvement / baseline_result["aiops_performance"]["average_score"] * 100,
                "conclusion": self._generate_improvement_conclusion(aiops_improvement, general_change)
            }
        
        return comparison
    
    def _generate_improvement_conclusion(self, aiops_improvement: float, general_change: float) -> str:
        """生成改进结论"""
        if aiops_improvement > 0.1 and abs(general_change) < 0.05:
            return "成功实现AIOps知识强化，通用知识保持稳定"
        elif aiops_improvement > 0.05 and general_change < -0.05:
            return "AIOps知识有所提升，但通用知识略有下降"
        elif aiops_improvement > 0.1 and general_change > 0.05:
            return "同时提升了AIOps和通用知识，效果显著"
        elif abs(aiops_improvement) < 0.05:
            return "AIOps知识转移效果不明显"
        else:
            return "需要进一步分析知识转移模式"


def load_model_and_tokenizer(model_path: str, base_model: str = "Qwen/Qwen3-8B") -> Tuple:
    """加载模型和tokenizer"""
    logger.info(f"加载模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # 加载基础模型
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 如果提供了LoRA路径，加载适配器
    if model_path and Path(model_path).exists() and (Path(model_path) / "adapter_config.json").exists():
        logger.info(f"加载LoRA适配器: {model_path}")
        model = PeftModel.from_pretrained(base_model_obj, model_path)
        model = model.merge_and_unload()
    else:
        if model_path:
            logger.warning(f"模型路径无效，使用基础模型: {base_model}")
        model = base_model_obj
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="评估AIOps知识转移效果")
    parser.add_argument("--models", nargs="+", required=True, help="模型路径列表")
    parser.add_argument("--model_names", nargs="+", help="模型名称列表")
    parser.add_argument("--base_model", default="Qwen/Qwen3-8B", help="基础模型名称") 
    parser.add_argument("--output_dir", default="outputs/aiops_evaluation", help="输出目录")
    parser.add_argument("--save_detailed", action="store_true", help="保存详细结果")
    
    args = parser.parse_args()
    
    if args.model_names and len(args.model_names) != len(args.models):
        raise ValueError("模型名称数量必须与模型路径数量一致")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化评估器
    evaluator = AIOpsKnowledgeEvaluator()
    
    # 评估所有模型
    all_results = []
    
    for i, model_path in enumerate(args.models):
        model_name = args.model_names[i] if args.model_names else f"Model_{i+1}"
        
        try:
            # 加载模型
            model, tokenizer = load_model_and_tokenizer(model_path, args.base_model)
            
            # 评估模型
            result = evaluator.evaluate_model(model, tokenizer, model_name)
            all_results.append(result)
            
            # 清理GPU内存
            del model
            torch.cuda.empty_cache()
            
            logger.info(f"模型 {model_name} 评估完成")
            logger.info(f"  AIOps平均得分: {result['aiops_performance']['average_score']:.3f}")
            logger.info(f"  通用知识平均得分: {result['general_performance']['average_score']:.3f}")
            logger.info(f"  知识特化得分: {result['knowledge_specialization_score']:.3f}")
            
        except Exception as e:
            logger.error(f"评估模型 {model_name} 时出错: {e}")
    
    if len(all_results) < 1:
        logger.error("没有成功评估任何模型")
        return
    
    # 比较分析
    if len(all_results) > 1:
        comparison = evaluator.compare_models(all_results)
    else:
        comparison = {"note": "只有一个模型，无法进行比较分析"}
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存汇总结果
    summary_file = output_dir / f"aiops_evaluation_summary_{timestamp}.json"
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "evaluation_config": {
            "base_model": args.base_model,
            "models_evaluated": len(all_results)
        },
        "summary_results": [
            {
                "model_name": r["model_name"],
                "aiops_avg_score": r["aiops_performance"]["average_score"],
                "general_avg_score": r["general_performance"]["average_score"], 
                "specialization_score": r["knowledge_specialization_score"]
            }
            for r in all_results
        ],
        "comparison_analysis": comparison
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"汇总结果已保存: {summary_file}")
    
    # 保存详细结果
    if args.save_detailed:
        detailed_file = output_dir / f"aiops_evaluation_detailed_{timestamp}.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"详细结果已保存: {detailed_file}")
    
    # 打印结果摘要
    logger.info("\n=== AIOps知识转移评估结果 ===")
    
    if len(all_results) > 1:
        logger.info(f"模型排名 (按AIOps性能):")
        for rank_info in comparison["model_ranking"]:
            logger.info(f"  {rank_info['rank']}. {rank_info['model_name']}: "
                       f"AIOps={rank_info['aiops_score']:.3f}, "
                       f"General={rank_info['general_score']:.3f}, "
                       f"Specialization={rank_info['specialization_score']:.3f}")
        
        if "improvement_analysis" in comparison:
            imp = comparison["improvement_analysis"]
            logger.info(f"\n知识强化分析:")
            logger.info(f"  AIOps知识提升: {imp['aiops_knowledge_gain']:.3f}")
            logger.info(f"  通用知识变化: {imp['general_knowledge_change']:.3f}")
            logger.info(f"  相对提升: {imp['relative_aiops_improvement']:.1f}%")
            logger.info(f"  结论: {imp['conclusion']}")
    
    logger.info("评估完成！")


if __name__ == "__main__":
    main()