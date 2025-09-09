"""
模型评估器
支持多种任务的自动评估：数学推理、代码生成、AIOps分析
"""
import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import subprocess
import tempfile
import signal
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table

from utils.common import load_jsonl, save_jsonl

logger = logging.getLogger(__name__)
console = Console()


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        self.model_path = model_path
        
        logger.info(f"Loading model for evaluation: {model_path}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.95,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """生成单个响应"""
        
        # 构建消息
        messages = [{"role": "user", "content": prompt}]
        
        # 应用聊天模板
        if hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            text = f"Human: {prompt}\n\nAssistant:"
        
        # 分词
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
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
        input_length = inputs['input_ids'].shape[1]
        response_ids = outputs[0][input_length:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        return response.strip()
    
    def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 4,
        **generation_kwargs
    ) -> List[str]:
        """批量生成响应"""
        
        responses = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_responses = []
            
            for prompt in batch_prompts:
                response = self.generate_response(prompt, **generation_kwargs)
                batch_responses.append(response)
            
            responses.extend(batch_responses)
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Generated {i + len(batch_prompts)}/{len(prompts)} responses")
        
        return responses


class GSM8KEvaluator:
    """GSM8K数学推理评估器"""
    
    @staticmethod
    def extract_answer(response: str) -> Optional[float]:
        """从响应中提取最终答案"""
        
        # 寻找 "Final Answer:", "答案是", "答案为" 等模式
        patterns = [
            r"(?:Final Answer|最终答案|答案)[：:]\s*([+-]?\d+(?:\.\d+)?)",
            r"#### ([+-]?\d+(?:\.\d+)?)",  # GSM8K格式
            r"答案是\s*([+-]?\d+(?:\.\d+)?)",
            r"答案为\s*([+-]?\d+(?:\.\d+)?)",
            r"=\s*([+-]?\d+(?:\.\d+)?)$"  # 行末等号
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.MULTILINE | re.IGNORECASE)
            if matches:
                try:
                    return float(matches[-1])  # 取最后一个匹配
                except ValueError:
                    continue
        
        # 回退：提取最后一个数字
        numbers = re.findall(r"([+-]?\d+(?:\.\d+)?)", response)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        
        return None
    
    @staticmethod
    def extract_reference_answer(reference: str) -> Optional[float]:
        """从参考答案中提取数值"""
        # GSM8K的答案格式通常是 "#### 1640"
        match = re.search(r"#### ([+-]?\d+(?:\.\d+)?)", reference)
        if match:
            return float(match.group(1))
        
        # 回退到提取数字
        numbers = re.findall(r"([+-]?\d+(?:\.\d+)?)", reference)
        if numbers:
            return float(numbers[-1])
        
        return None
    
    @classmethod
    def evaluate(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """评估GSM8K准确率"""
        
        correct = 0
        total = 0
        no_answer_count = 0
        
        results = []
        
        for pred, ref in zip(predictions, references):
            pred_answer = self.extract_answer(pred)
            ref_answer = self.extract_reference_answer(ref)
            
            if pred_answer is None:
                no_answer_count += 1
                results.append({
                    "prediction": pred,
                    "reference": ref,
                    "pred_answer": None,
                    "ref_answer": ref_answer,
                    "correct": False
                })
                total += 1
                continue
            
            if ref_answer is None:
                logger.warning(f"Cannot extract reference answer: {ref}")
                continue
            
            is_correct = abs(pred_answer - ref_answer) < 1e-6
            correct += is_correct
            total += 1
            
            results.append({
                "prediction": pred,
                "reference": ref,
                "pred_answer": pred_answer,
                "ref_answer": ref_answer,
                "correct": is_correct
            })
        
        accuracy = correct / total if total > 0 else 0.0
        
        metrics = {
            "exact_match": accuracy,
            "total_samples": total,
            "correct_samples": correct,
            "no_answer_samples": no_answer_count,
            "no_answer_rate": no_answer_count / len(predictions) if predictions else 0.0
        }
        
        return metrics, results


class HumanEvalEvaluator:
    """HumanEval代码生成评估器"""
    
    @staticmethod
    def extract_code(response: str, entry_point: str) -> str:
        """从响应中提取代码"""
        
        # 寻找代码块
        code_patterns = [
            r"```python\s*\n(.*?)\n```",
            r"```\s*\n(.*?)\n```",
            r"def\s+" + entry_point + r".*?(?=\n\n|\nclass|\ndef|\Z)"
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.MULTILINE)
            if matches:
                code = matches[0].strip()
                # 确保代码包含函数定义
                if f"def {entry_point}" in code:
                    return code
        
        # 回退：返回完整响应
        return response
    
    @staticmethod
    def execute_code(code: str, test_case: str, timeout: int = 5) -> Tuple[bool, str]:
        """安全执行代码"""
        
        @contextmanager
        def timeout_handler(seconds):
            def timeout_alarm(signum, frame):
                raise TimeoutError(f"Code execution timed out after {seconds}s")
            
            signal.signal(signal.SIGALRM, timeout_alarm)
            signal.alarm(seconds)
            try:
                yield
            finally:
                signal.alarm(0)
        
        try:
            # 创建临时执行环境
            exec_globals = {"__builtins__": __builtins__.copy()}
            
            with timeout_handler(timeout):
                # 执行代码定义
                exec(code, exec_globals)
                
                # 执行测试
                exec(test_case, exec_globals)
                
            return True, "Success"
            
        except Exception as e:
            return False, str(e)
    
    @classmethod
    def evaluate(
        self,
        predictions: List[str],
        test_cases: List[str],
        entry_points: List[str],
        k_list: List[int] = [1, 5, 10],
        n_samples: int = 1
    ) -> Dict[str, float]:
        """评估HumanEval pass@k"""
        
        if n_samples == 1:
            # 单样本评估：pass@1
            passed = []
            results = []
            
            for pred, test, entry_point in zip(predictions, test_cases, entry_points):
                code = self.extract_code(pred, entry_point)
                success, error_msg = self.execute_code(code, test)
                
                passed.append(success)
                results.append({
                    "prediction": pred,
                    "extracted_code": code,
                    "passed": success,
                    "error": error_msg if not success else None
                })
            
            pass_at_1 = sum(passed) / len(passed) if passed else 0.0
            
            metrics = {
                "pass@1": pass_at_1,
                "total_samples": len(predictions),
                "passed_samples": sum(passed)
            }
            
            return metrics, results
        
        else:
            # 多样本评估：pass@k
            # TODO: 实现多样本pass@k评估
            raise NotImplementedError("Multi-sample evaluation not implemented yet")


class AIOpsEvaluator:
    """AIOps分析评估器"""
    
    @staticmethod
    def extract_keywords(text: str) -> List[str]:
        """提取关键词"""
        # 定义AIOps相关关键词
        keywords_patterns = [
            r"连接池|connection pool",
            r"内存泄漏|memory leak",
            r"磁盘空间|disk space",
            r"网络延迟|network latency",
            r"数据库|database",
            r"超时|timeout",
            r"熔断|circuit breaker",
            r"负载均衡|load balancer",
            r"重启|restart",
            r"配置|config",
        ]
        
        found_keywords = []
        text_lower = text.lower()
        
        for pattern in keywords_patterns:
            if re.search(pattern, text_lower):
                found_keywords.extend(re.findall(pattern, text_lower))
        
        return found_keywords
    
    @classmethod
    def evaluate(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """评估AIOps分析质量"""
        
        keyword_matches = 0
        substring_matches = 0
        total = len(predictions)
        
        results = []
        
        for pred, ref in zip(predictions, references):
            pred_keywords = set(self.extract_keywords(pred))
            ref_keywords = set(self.extract_keywords(ref))
            
            # 关键词匹配
            keyword_overlap = len(pred_keywords & ref_keywords)
            keyword_match = keyword_overlap > 0
            keyword_matches += keyword_match
            
            # 子串匹配
            substring_match = any(key_ref in pred.lower() for key_ref in ref.lower().split())
            substring_matches += substring_match
            
            results.append({
                "prediction": pred,
                "reference": ref,
                "pred_keywords": list(pred_keywords),
                "ref_keywords": list(ref_keywords),
                "keyword_match": keyword_match,
                "substring_match": substring_match
            })
        
        metrics = {
            "keyword_match_rate": keyword_matches / total if total > 0 else 0.0,
            "substring_match_rate": substring_matches / total if total > 0 else 0.0,
            "total_samples": total
        }
        
        return metrics, results


def run_evaluation(
    model_path: str,
    eval_data_path: str,
    output_dir: str,
    generation_kwargs: Optional[Dict] = None,
    **kwargs
) -> Dict[str, Any]:
    """运行完整评估"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"开始评估模型: {model_path}")
    
    # 加载评估数据
    eval_data = load_jsonl(eval_data_path)
    logger.info(f"评估样本数: {len(eval_data)}")
    
    # 加载模型
    evaluator = ModelEvaluator(model_path)
    
    # 默认生成参数
    default_gen_kwargs = {
        "max_new_tokens": 512,
        "temperature": 0.1,
        "top_p": 0.95,
        "do_sample": True
    }
    if generation_kwargs:
        default_gen_kwargs.update(generation_kwargs)
    
    # 生成响应
    logger.info("生成模型响应...")
    prompts = [item['prompt'] for item in eval_data]
    predictions = evaluator.batch_generate(
        prompts,
        batch_size=kwargs.get('batch_size', 4),
        **default_gen_kwargs
    )
    
    # 按任务分组评估
    domain_results = {}
    domain_data = {}
    
    for domain in ['math', 'code', 'aiops']:
        # 过滤该领域的数据
        domain_indices = [i for i, item in enumerate(eval_data) if item['domain'] == domain]
        if not domain_indices:
            continue
        
        domain_predictions = [predictions[i] for i in domain_indices]
        domain_eval_data = [eval_data[i] for i in domain_indices]
        
        logger.info(f"评估 {domain} 任务，样本数: {len(domain_indices)}")
        
        if domain == 'math':
            references = [item['expected_answer'] for item in domain_eval_data]
            metrics, results = GSM8KEvaluator.evaluate(domain_predictions, references)
        
        elif domain == 'code':
            test_cases = [item['metadata']['test_cases'] for item in domain_eval_data]
            entry_points = [item['metadata']['entry_point'] for item in domain_eval_data]
            metrics, results = HumanEvalEvaluator.evaluate(domain_predictions, test_cases, entry_points)
        
        elif domain == 'aiops':
            references = [item['expected_answer'] for item in domain_eval_data]
            metrics, results = AIOpsEvaluator.evaluate(domain_predictions, references)
        
        domain_results[domain] = metrics
        domain_data[domain] = results
        
        # 保存详细结果
        save_jsonl(results, output_dir / f"{domain}_detailed_results.json")
    
    # 计算总体指标
    total_samples = sum(metrics['total_samples'] for metrics in domain_results.values())
    
    overall_results = {
        "model_path": model_path,
        "eval_data_path": eval_data_path,
        "total_samples": total_samples,
        "domain_results": domain_results,
        "generation_config": default_gen_kwargs
    }
    
    # 保存结果
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(overall_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"评估完成，结果已保存到: {results_file}")
    
    # 打印结果摘要
    _print_evaluation_summary(overall_results)
    
    return overall_results


def _print_evaluation_summary(results: Dict[str, Any]):
    """打印评估结果摘要"""
    
    table = Table(title="评估结果摘要")
    table.add_column("任务", style="cyan")
    table.add_column("样本数", justify="right")
    table.add_column("主要指标", justify="right", style="green")
    table.add_column("指标值", justify="right", style="bold green")
    
    for domain, metrics in results["domain_results"].items():
        if domain == "math":
            main_metric = "exact_match"
            metric_name = "准确率"
        elif domain == "code":
            main_metric = "pass@1"
            metric_name = "通过率"
        elif domain == "aiops":
            main_metric = "keyword_match_rate"
            metric_name = "关键词匹配率"
        
        value = metrics[main_metric]
        table.add_row(
            domain.upper(),
            str(metrics['total_samples']),
            metric_name,
            f"{value:.1%}"
        )
    
    console.print(table)