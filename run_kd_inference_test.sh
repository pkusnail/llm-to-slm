#!/bin/bash
# KD模型评估测试启动脚本

echo "🔥 KD模型推理对比测试 - GPU优化版本"
echo "======================================="
echo "可用测试模式:"
echo "  quick  - 快速验证 (10样本, ~3分钟, batch=8)"
echo "  medium - 中等规模 (50样本, ~15分钟, batch=16)" 
echo "  full   - 完整评估 (350样本, ~30分钟, batch=24)"
echo "======================================="

# 激活虚拟环境
source l2s/bin/activate

# 检查参数
if [ $# -eq 0 ]; then
    echo "❌ 请指定测试模式!"
    echo "使用方法:"
    echo "  ./run_tests.sh quick   # 快速测试"
    echo "  ./run_tests.sh medium  # 中等测试"  
    echo "  ./run_tests.sh full    # 完整测试"
    exit 1
fi

MODE=$1

# 验证模式
if [[ "$MODE" != "quick" && "$MODE" != "medium" && "$MODE" != "full" ]]; then
    echo "❌ 无效的测试模式: $MODE"
    echo "有效模式: quick, medium, full"
    exit 1
fi

echo "🚀 启动 $MODE 模式测试..."
echo "⚡ GPU优化配置:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits | head -8

echo ""
echo "开始测试..."

# 获取可选的temperature参数
TEMPERATURE=${2:-0.7}  # 默认0.7

echo "🌡️ 推理温度: $TEMPERATURE"

# 运行测试
python test_kd_inference_v2.py --mode $MODE --temperature $TEMPERATURE

echo ""
echo "✅ 测试完成! 请查看生成的结果文件。"