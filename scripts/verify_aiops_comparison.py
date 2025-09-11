#!/usr/bin/env python3
"""
Direct AIOps comparison between 30B Teacher and 8B Student
验证8B在AIOps方面是否真的比30B强
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time
from pathlib import Path

def load_model(model_path, base_model=None):
    """加载模型"""
    print(f"加载模型: {model_path}")
    
    if base_model is None:
        base_model = model_path
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.eval()
    return model, tokenizer

def test_aiops_knowledge(model, tokenizer, model_name):
    """AIOps专业知识测试"""
    test_cases = [
        {
            "category": "Kubernetes",
            "question": "如何查看Kubernetes集群中所有Pod的状态？",
            "keywords": ["kubectl", "get", "pods", "状态", "all-namespaces"]
        },
        {
            "category": "监控告警",
            "question": "Prometheus中如何配置CPU使用率超过80%的告警规则？",
            "keywords": ["alert", "cpu", "usage", "80", "prometheus", "rules"]
        },
        {
            "category": "日志分析",
            "question": "在微服务架构中，如何实现分布式链路追踪？",
            "keywords": ["tracing", "jaeger", "zipkin", "span", "distributed", "微服务"]
        },
        {
            "category": "故障处理",
            "question": "当应用响应时间过慢时，应该从哪些方面排查？",
            "keywords": ["cpu", "内存", "数据库", "网络", "缓存", "索引"]
        },
        {
            "category": "容器优化",
            "question": "Docker容器内存使用过高的常见原因和解决方案？",
            "keywords": ["内存泄漏", "limit", "jvm", "资源限制", "gc", "heap"]
        },
        {
            "category": "云原生",
            "question": "Service Mesh的主要作用是什么？",
            "keywords": ["istio", "envoy", "traffic", "security", "observability", "服务网格"]
        }
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\n测试类别: {case['category']}")
        print(f"问题: {case['question']}")
        
        # 生成回答
        inputs = tokenizer(f"问题: {case['question']}\n回答: ", return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(f"问题: {case['question']}\n回答: ", "").strip()
        
        print(f"回答: {response}")
        
        # 关键词匹配评分
        matched_keywords = []
        for keyword in case['keywords']:
            if keyword.lower() in response.lower():
                matched_keywords.append(keyword)
        
        score = len(matched_keywords) / len(case['keywords'])
        
        results.append({
            "category": case['category'],
            "question": case['question'],
            "response": response,
            "expected_keywords": case['keywords'],
            "matched_keywords": matched_keywords,
            "score": score
        })
        
        print(f"匹配关键词: {matched_keywords}")
        print(f"得分: {score:.3f} ({len(matched_keywords)}/{len(case['keywords'])})")
    
    overall_score = sum(r['score'] for r in results) / len(results)
    print(f"\n📊 {model_name} AIOps总体得分: {overall_score:.3f}")
    
    return overall_score, results

def main():
    """主函数：直接对比30B Teacher和8B Student的AIOps能力"""
    print("🔍 Direct AIOps Comparison: 30B Teacher vs 8B Student")
    print("="*60)
    
    models_to_test = [
        {
            "name": "Teacher_30B",
            "model_path": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507"
        },
        {
            "name": "Student_8B_Baseline", 
            "model_path": "Qwen/Qwen3-8B",
            "base_model": "Qwen/Qwen3-8B"
        }
    ]
    
    all_results = {}
    
    for model_config in models_to_test:
        print(f"\n{'='*60}")
        print(f"🧪 测试模型: {model_config['name']}")
        print(f"📁 路径: {model_config['model_path']}")
        print(f"{'='*60}")
        
        try:
            model, tokenizer = load_model(model_config['model_path'], model_config['base_model'])
            overall_score, detailed_results = test_aiops_knowledge(model, tokenizer, model_config['name'])
            
            all_results[model_config['name']] = {
                'overall_score': overall_score,
                'detailed_results': detailed_results
            }
            
            # 清理GPU内存
            del model, tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ 测试 {model_config['name']} 时出错: {str(e)}")
            all_results[model_config['name']] = {'overall_score': 0.0, 'error': str(e)}
    
    # 输出对比结果
    print(f"\n{'='*60}")
    print("📊 AIOps能力直接对比")
    print(f"{'='*60}")
    
    teacher_score = all_results.get('Teacher_30B', {}).get('overall_score', 0.0)
    student_score = all_results.get('Student_8B_Baseline', {}).get('overall_score', 0.0)
    
    print(f"Teacher_30B:          {teacher_score:.3f}")
    print(f"Student_8B_Baseline:  {student_score:.3f}")
    
    if student_score > teacher_score:
        print(f"✅ 确认: 8B Student在AIOps方面确实比30B Teacher强 ({student_score:.3f} > {teacher_score:.3f})")
        print("   可能原因: 30B模型过于通用化，8B模型在特定领域表现更集中")
    elif teacher_score > student_score:
        print(f"❌ 结果: 30B Teacher在AIOps方面比8B Student强 ({teacher_score:.3f} > {student_score:.3f})")
        print("   这与之前的评测结果不一致，需要检查评测方法")
    else:
        print("📊 两个模型在AIOps方面表现相当")
    
    # 保存结果
    output_file = "outputs/aiops_direct_comparison.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "comparison_summary": {
                "teacher_30b_score": teacher_score,
                "student_8b_score": student_score,
                "winner": "Student_8B" if student_score > teacher_score else "Teacher_30B" if teacher_score > student_score else "Tie"
            },
            "detailed_results": all_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 详细结果已保存到: {output_file}")
    return all_results

if __name__ == "__main__":
    main()