#!/usr/bin/env python3
"""
轻量标准评测脚本
集成常用的标准评测任务，但只选取重点的几个维度
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import time
from pathlib import Path
import re
import math

def load_model(model_path, base_model="Qwen/Qwen3-8B"):
    """加载模型"""
    print(f"加载模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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

def test_gsm8k_math(model, tokenizer):
    """GSM8K风格数学推理测试 (轻量版)"""
    test_cases = [
        {
            "question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day?",
            "answer": "18"  # 16-3-4=9, 9*2=18
        },
        {
            "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts are needed?", 
            "answer": "3"   # 2 + 1 = 3
        },
        {
            "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increases the value of the house by 150%. How much profit did he make?",
            "answer": "70000"  # Original: 80k+50k=130k, New value: 130k*1.5=195k, Profit: 195k-130k=65k
        }
    ]
    
    results = []
    for case in test_cases:
        inputs = tokenizer(f"Question: {case['question']}\nAnswer: Let me solve this step by step.\n", 
                          return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,  # 低温度保证数学准确性
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(f"Question: {case['question']}\nAnswer: Let me solve this step by step.\n", "").strip()
        
        # 提取数字答案
        numbers = re.findall(r'\d+', response.split('.')[-1])  # 从最后一句提取数字
        predicted = numbers[-1] if numbers else "0"
        
        is_correct = predicted == case['answer']
        results.append({
            "question": case['question'][:50] + "...",
            "predicted": predicted,
            "expected": case['answer'],
            "correct": is_correct,
            "response": response[:100] + "..."
        })
    
    accuracy = sum(r['correct'] for r in results) / len(results)
    return accuracy, results

def test_hellaswag_common_sense(model, tokenizer):
    """HellaSwag风格常识推理测试 (轻量版)"""
    test_cases = [
        {
            "context": "A woman is outside with a bucket and a dog. The dog is running around trying to avoid getting a bath. She...",
            "options": [
                "starts to pet the dog",
                "chases the dog to give it a bath", 
                "gets a drink of water",
                "sits down and watches"
            ],
            "answer": 1  # chases the dog to give it a bath
        },
        {
            "context": "A man is holding a microphone and speaking to a large audience. The audience is listening carefully. He...",
            "options": [
                "starts singing a song",
                "continues his speech",
                "throws the microphone away", 
                "leaves the stage"
            ],
            "answer": 1  # continues his speech
        },
        {
            "context": "Children are playing in the playground. One child falls and starts crying. Another child...",
            "options": [
                "runs away quickly",
                "helps the crying child up",
                "starts crying too",
                "continues playing alone"
            ],
            "answer": 1  # helps the crying child up
        }
    ]
    
    results = []
    for case in test_cases:
        correct_count = 0
        option_scores = []
        
        for i, option in enumerate(case['options']):
            full_text = case['context'] + " " + option
            inputs = tokenizer(full_text, return_tensors="pt", max_length=256, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                # 计算perplexity作为评分
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    loss = outputs.loss
                else:
                    loss = torch.nn.functional.cross_entropy(
                        outputs.logits[:, :-1].contiguous().view(-1, outputs.logits.size(-1)),
                        inputs['input_ids'][:, 1:].contiguous().view(-1)
                    )
                perplexity = torch.exp(loss).item()
                option_scores.append(perplexity)
        
        # 选择perplexity最低的选项
        predicted = option_scores.index(min(option_scores))
        is_correct = predicted == case['answer']
        
        results.append({
            "context": case['context'][:50] + "...",
            "predicted": predicted,
            "expected": case['answer'],
            "correct": is_correct,
            "scores": option_scores
        })
    
    accuracy = sum(r['correct'] for r in results) / len(results)
    return accuracy, results

def test_mmlu_knowledge(model, tokenizer):
    """MMLU风格知识测试 (计算机科学子集)"""
    test_cases = [
        {
            "question": "In Python, what does the 'yield' keyword do?",
            "options": [
                "Returns a value and exits the function",
                "Creates a generator function", 
                "Raises an exception",
                "Imports a module"
            ],
            "answer": 1
        },
        {
            "question": "What is the time complexity of binary search?",
            "options": [
                "O(n)",
                "O(log n)",
                "O(n^2)", 
                "O(1)"
            ],
            "answer": 1
        },
        {
            "question": "In Kubernetes, what is a Pod?",
            "options": [
                "A cluster of nodes",
                "The smallest deployable unit",
                "A network policy",
                "A storage volume"
            ],
            "answer": 1
        }
    ]
    
    results = []
    for case in test_cases:
        # 构建多选题prompt
        prompt = f"Question: {case['question']}\n"
        for i, option in enumerate(case['options']):
            prompt += f"{chr(65+i)}. {option}\n"
        prompt += "Answer: The correct answer is"
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        # 提取答案选项
        predicted = -1
        for i, letter in enumerate(['A', 'B', 'C', 'D']):
            if letter in response[:3]:  # 只看前3个字符
                predicted = i
                break
        
        is_correct = predicted == case['answer']
        results.append({
            "question": case['question'][:40] + "...",
            "predicted": predicted,
            "expected": case['answer'],
            "correct": is_correct,
            "response": response[:20]
        })
    
    accuracy = sum(r['correct'] for r in results) / len(results)
    return accuracy, results

def run_lightweight_standard_eval():
    """运行轻量标准评测"""
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
        },
        {
            "name": "KD_Student",
            "model_path": "outputs/experiment/ultra_optimized_kd/checkpoint-500",
            "base_model": "Qwen/Qwen3-8B"
        }
    ]
    
    all_results = {}
    
    for model_config in models_to_test:
        print(f"\n{'='*60}")
        print(f"🧮 标准评测: {model_config['name']}")
        print(f"{'='*60}")
        
        try:
            model, tokenizer = load_model(model_config['model_path'], model_config['base_model'])
            
            # 运行三个核心测试
            print("\n📊 GSM8K数学推理测试...")
            math_acc, math_results = test_gsm8k_math(model, tokenizer)
            
            print("\n🧠 HellaSwag常识推理测试...")
            cs_acc, cs_results = test_hellaswag_common_sense(model, tokenizer)
            
            print("\n📚 MMLU知识测试...")
            know_acc, know_results = test_mmlu_knowledge(model, tokenizer)
            
            # 汇总结果
            overall_score = (math_acc + cs_acc + know_acc) / 3
            
            print(f"\n📈 {model_config['name']} 标准评测结果:")
            print(f"   数学推理 (GSM8K): {math_acc:.3f}")
            print(f"   常识推理 (HellaSwag): {cs_acc:.3f}")
            print(f"   知识测试 (MMLU): {know_acc:.3f}")
            print(f"   综合得分: {overall_score:.3f}")
            
            all_results[model_config['name']] = {
                'math_accuracy': math_acc,
                'common_sense_accuracy': cs_acc, 
                'knowledge_accuracy': know_acc,
                'overall_score': overall_score,
                'detailed_results': {
                    'math': math_results,
                    'common_sense': cs_results,
                    'knowledge': know_results
                }
            }
            
            # 清理内存
            del model, tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ 测试 {model_config['name']} 时出错: {str(e)}")
            all_results[model_config['name']] = {'error': str(e)}
    
    # 保存结果
    output_file = "outputs/lightweight_standard_eval_results.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_description": "轻量标准评测: GSM8K数学 + HellaSwag常识 + MMLU知识",
            "results": all_results
        }, f, ensure_ascii=False, indent=2)
    
    # 打印对比总结
    print(f"\n{'='*60}")
    print("📊 标准评测对比总结:")
    print(f"{'='*60}")
    print(f"{'模型':<20} {'数学':<8} {'常识':<8} {'知识':<8} {'综合':<8}")
    print("-" * 60)
    for name, result in all_results.items():
        if 'error' not in result:
            print(f"{name:<20} {result['math_accuracy']:<8.3f} {result['common_sense_accuracy']:<8.3f} "
                  f"{result['knowledge_accuracy']:<8.3f} {result['overall_score']:<8.3f}")
    
    print(f"\n💾 详细结果已保存到: {output_file}")
    return all_results

if __name__ == "__main__":
    print("🎯 轻量标准评测启动")
    print("测试维度: GSM8K数学推理 + HellaSwag常识推理 + MMLU知识测试")
    
    results = run_lightweight_standard_eval()