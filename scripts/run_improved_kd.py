#!/usr/bin/env python3
"""
Improved Knowledge Distillation Script
Based on evaluation: Current KD shows 0% improvement over baseline!

Key Issues Found:
1. Training loss diverged (13.52 → 16.04)
2. Student = Baseline (2.99 perplexity both)
3. Need much more conservative hyperparameters
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the KD function from src
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.distillation.kd import run_kd_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Improved KD with conservative hyperparameters")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Very low LR (default: 1e-5)")
    parser.add_argument("--temperature", type=float, default=1.2, help="Lower temperature (default: 1.2)")
    parser.add_argument("--alpha", type=float, default=0.8, help="High KL weight (default: 0.8)")
    parser.add_argument("--max_steps", type=int, default=100, help="Shorter training (default: 100)")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    parser.add_argument("--train_data", type=str, default=None, help="Path to training data")
    parser.add_argument("--eval_data", type=str, default=None, help="Path to evaluation data")
    
    args = parser.parse_args()
    
    # Create experiment
    experiment_name = f"qwen3_conservative_kd_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # VERY conservative config to ensure convergence
    config = {
        "experiment_name": experiment_name,
        "problem_analysis": {
            "current_issues": [
                "Training loss diverged (13.52 → 16.04)",
                "Student model = Baseline (0% improvement)",
                "KD completely ineffective"
            ],
            "solutions": [
                "Much lower learning rate (1e-5 vs 1.5e-4)",
                "Lower temperature for sharper teacher distributions",
                "Higher alpha to focus more on KL divergence",
                "Shorter training to avoid overfitting"
            ]
        },
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "temperature": args.temperature, 
            "alpha": args.alpha,
            "max_steps": args.max_steps,
            "gradient_accumulation_steps": 64,  # Very smooth gradients
            "warmup_steps": 20,
            "save_steps": 20,
            "logging_steps": 5
        }
    }
    
    # Save config
    with open(output_dir / "conservative_config.json", 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info("🎯 CONSERVATIVE KNOWLEDGE DISTILLATION")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Problem: Current KD shows 0% improvement over baseline!")
    
    logger.info("\n🔧 CONSERVATIVE FIXES:")
    logger.info(f"  Learning Rate: {args.learning_rate:.1e} (MUCH lower - was 1.5e-4)")
    logger.info(f"  Temperature: {args.temperature} (sharper - was 2.0)")
    logger.info(f"  Alpha (KL weight): {args.alpha} (higher - was 0.5)")
    logger.info(f"  Max Steps: {args.max_steps} (shorter - was ~200)")
    logger.info(f"  Grad Accumulation: 64 (smoother - was 16)")
    
    # Use provided data paths or default to existing teacher-generated data
    if args.train_data:
        train_file = Path(args.train_data)
    else:
        train_file = Path("outputs/experiment/qwen3_30b_to_8b_ultrabatch_512/sft/sft_train_data_clean.jsonl")
    
    if args.eval_data:
        eval_file = Path(args.eval_data)
    else:
        eval_file = Path("outputs/experiment/qwen3_30b_to_8b_ultrabatch_512/sft/sft_eval_data_clean.jsonl")
    
    if not train_file.exists():
        logger.error(f"Training data file not found: {train_file}")
        logger.error("Please run: python scripts/prepare_training_data.py")
        return
    
    if not eval_file.exists():
        logger.error(f"Evaluation data file not found: {eval_file}")
        logger.error("Please run: python scripts/prepare_training_data.py")
        return
    
    logger.info(f"\n📁 Data verified:")
    logger.info(f"  Train: {train_file.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info(f"  Eval: {eval_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    logger.info("\n🚀 Starting conservative training...")
    
    try:
        results = run_kd_pipeline(
            teacher_model_path="Qwen/Qwen3-30B-A3B-Instruct-2507",
            student_model_path="Qwen/Qwen3-8B",
            train_data_path=str(train_file),
            eval_data_path=str(eval_file),
            output_dir=str(output_dir),
            
            # Conservative training parameters
            num_train_epochs=1,
            max_steps=args.max_steps,
            per_device_train_batch_size=1,
            learning_rate=args.learning_rate,  # MUCH lower
            gradient_accumulation_steps=64,    # Much smoother
            max_length=2048,
            
            # Learning rate scheduling
            warmup_steps=20,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            max_grad_norm=0.5,  # Lower gradient clipping
            
            # Frequent monitoring
            save_steps=20,
            eval_steps=25,
            logging_steps=5,
            
            # Conservative KD parameters
            temperature=args.temperature,  # Lower for sharper distributions
            alpha=args.alpha,             # Higher KL weight
            
            # Hardware
            bf16=True,
            gradient_checkpointing=True,
            dataloader_num_workers=4,
            seed=42
        )
        
        # Save results with analysis
        results["config"] = config
        results_path = output_dir / "conservative_results.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✅ Training completed: {results_path}")
        
        # Analyze training progress
        if "training_history" in results and results["training_history"]:
            history = results["training_history"]
            if len(history) >= 2:
                losses = [h.get("total_loss") for h in history if "total_loss" in h]
                if len(losses) >= 2:
                    init_loss = losses[0]
                    final_loss = losses[-1]
                    improvement = (init_loss - final_loss) / init_loss * 100
                    
                    logger.info(f"\n📊 TRAINING ANALYSIS:")
                    logger.info(f"  Initial loss: {init_loss:.3f}")
                    logger.info(f"  Final loss: {final_loss:.3f}")
                    logger.info(f"  Improvement: {improvement:+.1f}%")
                    
                    if improvement > 10:
                        logger.info("🎉 Excellent! Much better convergence!")
                    elif improvement > 5:
                        logger.info("✅ Good improvement!")
                    elif improvement > 0:
                        logger.info("📈 Positive trend - on the right track")
                    else:
                        logger.warning("⚠️ Still diverging - need even more conservative settings")
        
        logger.info(f"\n🔍 Next: Evaluate with:")
        logger.info(f"python scripts/evaluate_distillation.py --model_path {output_dir}/final_model")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.info("\n🔧 Even more conservative options:")
        logger.info("  --learning_rate 5e-6")
        logger.info("  --temperature 1.0") 
        logger.info("  --max_steps 50")
        raise

if __name__ == "__main__":
    main()