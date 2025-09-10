#!/usr/bin/env python3
"""
Knowledge Distillation Evaluation Script
评估知识蒸馏效果的综合脚本

Usage:
    python scripts/evaluate_distillation.py --model_path outputs/qwen3_adapter_fixed_kd/final_model
"""

import json
import torch
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path, base_model="Qwen/Qwen3-8B"):
    """Load student model with LoRA adapters"""
    logger.info(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Load base student model
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapters if path exists and is not empty
    if model_path and Path(model_path).exists() and (Path(model_path) / "adapter_config.json").exists():
        logger.info(f"Loading LoRA adapters from: {model_path}")
        model = PeftModel.from_pretrained(base_model_obj, model_path)
        model = model.merge_and_unload()  # Merge LoRA weights
    else:
        if model_path:
            logger.warning(f"Model path {model_path} not found or invalid, using base model")
        else:
            logger.info("No model path provided, using base model")
        model = base_model_obj
    
    return model, tokenizer

def load_teacher_model(teacher_model="Qwen/Qwen3-30B-A3B-Instruct-2507"):
    """Load teacher model for comparison"""
    logger.info(f"Loading teacher model: {teacher_model}")
    tokenizer = AutoTokenizer.from_pretrained(teacher_model)
    model = AutoModelForCausalLM.from_pretrained(
        teacher_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

def evaluate_perplexity(model, tokenizer, test_texts, max_length=512):
    """Evaluate model perplexity on test texts"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            total_loss += loss.item() * inputs["input_ids"].numel()
            total_tokens += inputs["input_ids"].numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity, avg_loss

def evaluate_generation_quality(model, tokenizer, prompts, max_new_tokens=256):
    """Evaluate generation quality"""
    model.eval()
    results = []
    
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model.generate(**inputs, **generation_config)
            generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            results.append({
                "prompt": prompt,
                "generated": generated_text,
                "length": len(generated_text)
            })
    
    return results

def load_evaluation_data(data_path="data/processed/eval_dataset.jsonl"):
    """Load evaluation dataset"""
    eval_data = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                # Handle different data formats
                if "prompt" in data:
                    # Current format: {"prompt": "...", "teacher_response": "..."}
                    eval_data.append({
                        "instruction": data["prompt"],
                        "input": "",
                        "expected_answer": data.get("teacher_response", data.get("original_answer", ""))
                    })
                elif "instruction" in data:
                    # Standard format
                    eval_data.append(data)
                else:
                    logger.warning(f"Skipping unknown data format: {list(data.keys())}")
        logger.info(f"Loaded {len(eval_data)} evaluation samples")
    except FileNotFoundError:
        logger.warning(f"Evaluation data not found at {data_path}, using default prompts")
        eval_data = [
            {"instruction": "请解释什么是人工智能？", "input": "", "expected_answer": ""},
            {"instruction": "写一首关于春天的诗", "input": "", "expected_answer": ""},
            {"instruction": "解释量子计算的基本原理", "input": "", "expected_answer": ""},
            {"instruction": "编写一个Python函数来计算斐波那契数列", "input": "", "expected_answer": ""},
            {"instruction": "描述机器学习和深度学习的区别", "input": "", "expected_answer": ""}
        ]
    
    return eval_data

def format_prompt(instruction, input_text=""):
    """Format prompt for Qwen models"""
    if input_text:
        return f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    else:
        return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

def main():
    parser = argparse.ArgumentParser(description="Evaluate knowledge distillation results")
    parser.add_argument("--model_path", required=True, help="Path to the distilled model")
    parser.add_argument("--teacher_model", default="Qwen/Qwen3-30B-A3B-Instruct-2507", help="Teacher model name")
    parser.add_argument("--student_base", default="Qwen/Qwen3-8B", help="Student base model name")
    parser.add_argument("--eval_data", default="data/processed/eval_dataset.jsonl", help="Evaluation data path")
    parser.add_argument("--output_dir", default="outputs/evaluation", help="Output directory for results")
    parser.add_argument("--sample_size", type=int, default=50, help="Number of samples to evaluate")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load evaluation data
    eval_data = load_evaluation_data(args.eval_data)
    eval_sample = eval_data[:args.sample_size]
    
    logger.info("=== KNOWLEDGE DISTILLATION EVALUATION ===")
    
    # Prepare evaluation prompts and texts
    prompts = [format_prompt(item["instruction"], item.get("input", "")) for item in eval_sample]
    texts_for_perplexity = [item.get("expected_answer", item.get("teacher_response", "")) for item in eval_sample if item.get("expected_answer") or item.get("teacher_response")]
    
    results = {
        "experiment": Path(args.model_path).parent.name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_path": args.model_path,
            "teacher_model": args.teacher_model,
            "student_base": args.student_base,
            "sample_size": len(eval_sample)
        },
        "results": {}
    }
    
    # 1. Evaluate Student Model (Distilled)
    logger.info("1. Evaluating Student Model (Distilled)")
    try:
        student_model, student_tokenizer = load_model_and_tokenizer(args.model_path, args.student_base)
        
        # Perplexity evaluation
        if texts_for_perplexity:
            student_ppl, student_loss = evaluate_perplexity(student_model, student_tokenizer, texts_for_perplexity[:10])
            logger.info(f"Student Perplexity: {student_ppl:.2f} (Loss: {student_loss:.4f})")
        else:
            student_ppl, student_loss = None, None
            logger.warning("No reference texts found for perplexity evaluation")
        
        # Generation evaluation
        student_generations = evaluate_generation_quality(student_model, student_tokenizer, prompts[:5])
        
        results["results"]["student"] = {
            "perplexity": student_ppl,
            "loss": student_loss,
            "generations": student_generations
        }
        
        # Clean up GPU memory
        del student_model
        torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"Student model evaluation failed: {e}")
        results["results"]["student"] = {"error": str(e)}
    
    # 2. Evaluate Baseline Student Model (No distillation)
    logger.info("2. Evaluating Baseline Student Model")
    try:
        baseline_model, baseline_tokenizer = load_model_and_tokenizer("", args.student_base)  # No LoRA adapters
        
        if texts_for_perplexity:
            baseline_ppl, baseline_loss = evaluate_perplexity(baseline_model, baseline_tokenizer, texts_for_perplexity[:10])
            logger.info(f"Baseline Perplexity: {baseline_ppl:.2f} (Loss: {baseline_loss:.4f})")
        else:
            baseline_ppl, baseline_loss = None, None
        
        baseline_generations = evaluate_generation_quality(baseline_model, baseline_tokenizer, prompts[:5])
        
        results["results"]["baseline"] = {
            "perplexity": baseline_ppl,
            "loss": baseline_loss,
            "generations": baseline_generations
        }
        
        del baseline_model
        torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"Baseline model evaluation failed: {e}")
        results["results"]["baseline"] = {"error": str(e)}
    
    # 3. Optional: Teacher Model Evaluation (if resources allow)
    if torch.cuda.device_count() >= 6:  # Only if we have enough GPUs
        logger.info("3. Evaluating Teacher Model")
        try:
            teacher_model, teacher_tokenizer = load_teacher_model(args.teacher_model)
            
            if texts_for_perplexity:
                teacher_ppl, teacher_loss = evaluate_perplexity(teacher_model, teacher_tokenizer, texts_for_perplexity[:5])
                logger.info(f"Teacher Perplexity: {teacher_ppl:.2f} (Loss: {teacher_loss:.4f})")
            else:
                teacher_ppl, teacher_loss = None, None
            
            teacher_generations = evaluate_generation_quality(teacher_model, teacher_tokenizer, prompts[:3])
            
            results["results"]["teacher"] = {
                "perplexity": teacher_ppl,
                "loss": teacher_loss,
                "generations": teacher_generations
            }
            
            del teacher_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Teacher model evaluation failed: {e}")
            results["results"]["teacher"] = {"error": str(e)}
    else:
        logger.info("Skipping teacher evaluation (insufficient GPU memory)")
        results["results"]["teacher"] = {"skipped": "insufficient_gpu_memory"}
    
    # 4. Calculate Improvement Metrics
    logger.info("4. Calculating Improvement Metrics")
    if "student" in results["results"] and "baseline" in results["results"]:
        student_res = results["results"]["student"]
        baseline_res = results["results"]["baseline"]
        
        improvements = {}
        if student_res.get("perplexity") and baseline_res.get("perplexity"):
            ppl_improvement = (baseline_res["perplexity"] - student_res["perplexity"]) / baseline_res["perplexity"] * 100
            improvements["perplexity_improvement_pct"] = ppl_improvement
            logger.info(f"Perplexity Improvement: {ppl_improvement:+.2f}%")
        
        if student_res.get("loss") and baseline_res.get("loss"):
            loss_improvement = (baseline_res["loss"] - student_res["loss"]) / baseline_res["loss"] * 100
            improvements["loss_improvement_pct"] = loss_improvement
            logger.info(f"Loss Improvement: {loss_improvement:+.2f}%")
        
        results["improvements"] = improvements
    
    # 5. Save Results
    output_file = output_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Evaluation results saved to: {output_file}")
    
    # 6. Print Summary
    logger.info("\n=== EVALUATION SUMMARY ===")
    if "student" in results["results"]:
        student_res = results["results"]["student"]
        if student_res.get("perplexity"):
            logger.info(f"Student Model Perplexity: {student_res['perplexity']:.2f}")
    
    if "baseline" in results["results"]:
        baseline_res = results["results"]["baseline"]
        if baseline_res.get("perplexity"):
            logger.info(f"Baseline Model Perplexity: {baseline_res['perplexity']:.2f}")
    
    if "improvements" in results:
        for metric, value in results["improvements"].items():
            logger.info(f"{metric}: {value:+.2f}%")

if __name__ == "__main__":
    main()