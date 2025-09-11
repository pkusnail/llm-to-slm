#!/usr/bin/env python3
"""
Student Model Generalization Test
测试Student模型在非AIOps领域的泛化能力
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import time
from pathlib import Path

def load_model(model_path, base_model="Qwen/Qwen3-8B"):
    """加载模型（支持LoRA适配器）"""
    print(f"加载模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 如果是checkpoint路径，加载LoRA适配器
    if Path(model_path).exists() and "checkpoint" in model_path:
        print(f"加载LoRA适配器: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        
    model.eval()
    return model, tokenizer

def test_general_capabilities(model, tokenizer):
    """测试通用能力"""
    test_cases = [
        # 数学推理
        {
            "category": "数学",
            "prompt": "计算：25 × 4 + 18 ÷ 3 = ?",
            "expected_keywords": ["100", "6", "106"]
        },
        # 逻辑推理
        {
            "category": "逻辑",
            "prompt": "如果所有的猫都是动物，而小黑是一只猫，那么小黑是什么？",
            "expected_keywords": ["动物", "animal"]
        },
        # 语言理解
        {
            "category": "语言",
            "prompt": "请解释'画蛇添足'这个成语的含义。",
            "expected_keywords": ["多余", "适得其反", "画蛇", "足"]
        },
        # 常识推理
        {
            "category": "常识", 
            "prompt": "为什么冬天会下雪而不是下雨？",
            "expected_keywords": ["温度", "冰点", "水蒸气", "凝固"]
        },
        # 创意写作
        {
            "category": "创意",
            "prompt": "请写一首关于春天的小诗（4行）。",
            "expected_keywords": ["春", "花", "绿", "暖"]
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n测试类别: {test_case['category']}")
        print(f"问题: {test_case['prompt']}")
        
        # 生成回答
        inputs = tokenizer(test_case['prompt'], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(test_case['prompt'], '').strip()
        
        print(f"回答: {response}")
        
        # 简单关键词匹配评估
        keyword_matches = 0
        for keyword in test_case['expected_keywords']:
            if keyword.lower() in response.lower():
                keyword_matches += 1
        
        relevance_score = keyword_matches / len(test_case['expected_keywords'])
        
        results.append({
            "category": test_case['category'],
            "prompt": test_case['prompt'],
            "response": response,
            "keyword_matches": keyword_matches,
            "total_keywords": len(test_case['expected_keywords']),
            "relevance_score": relevance_score
        })
        
        print(f"相关度得分: {relevance_score:.2f}")
    
    return results

def test_multiple_models():
    """测试多个模型的泛化能力"""
    models_to_test = [
        {
            "name": "Teacher_30B",
            "model_path": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "output_file": "outputs/generalization_teacher_30b.json"
        },
        {
            "name": "Student_8B_Baseline", 
            "model_path": "Qwen/Qwen3-8B",
            "base_model": "Qwen/Qwen3-8B",
            "output_file": "outputs/generalization_student_8b_baseline.json"
        },
        {
            "name": "KD_Student_Checkpoint500",
            "model_path": "outputs/experiment/ultra_optimized_kd/checkpoint-500",
            "base_model": "Qwen/Qwen3-8B", 
            "output_file": "outputs/generalization_kd_student.json"
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
            results = test_general_capabilities(model, tokenizer)
            
            # 计算总体得分
            avg_relevance = sum(r['relevance_score'] for r in results) / len(results)
            
            print(f"\n📈 {model_config['name']} 泛化能力测试结果:")
            print(f"平均相关度得分: {avg_relevance:.3f}")
            
            # 按类别显示结果
            for result in results:
                print(f"{result['category']:>6}: {result['relevance_score']:.3f} ({result['keyword_matches']}/{result['total_keywords']})")
            
            # 保存结果
            Path(model_config['output_file']).parent.mkdir(parents=True, exist_ok=True)
            
            summary = {
                "model_name": model_config['name'],
                "model_path": model_config['model_path'], 
                "base_model": model_config['base_model'],
                "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "overall_score": avg_relevance,
                "detailed_results": results
            }
            
            with open(model_config['output_file'], 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            all_results[model_config['name']] = {
                'overall_score': avg_relevance,
                'detailed_results': results
            }
            
            # 评估结论
            if avg_relevance > 0.6:
                print("✅ 模型保持了良好的泛化能力")
            elif avg_relevance > 0.4:
                print("⚠️ 模型泛化能力有所下降，但仍在可接受范围")
            else:
                print("❌ 模型泛化能力明显下降，需要调整训练策略")
                
            print(f"💾 结果已保存到: {model_config['output_file']}")
            
            # 清理GPU内存
            del model, tokenizer
            import torch
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ 测试 {model_config['name']} 时出错: {str(e)}")
            all_results[model_config['name']] = {'overall_score': 0.0, 'error': str(e)}
    
    # 打印对比总结
    print(f"\n{'='*60}")
    print("📊 所有模型泛化能力对比总结:")
    print(f"{'='*60}")
    for name, result in all_results.items():
        if 'error' not in result:
            print(f"{name:>25}: {result['overall_score']:.3f}")
        else:
            print(f"{name:>25}: 测试失败")
    
    return all_results

def main():
    print("🧪 Multi-Model Generalization Test")
    print("测试Teacher(30B)、Student Baseline(8B)、KD Student泛化能力对比")
    
    results = test_multiple_models()
    
    # 保存对比结果
    comparison_file = "outputs/generalization_comparison.json" 
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "comparison_results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 对比结果已保存到: {comparison_file}")

if __name__ == "__main__":
    main()