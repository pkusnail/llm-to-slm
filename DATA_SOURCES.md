# AIOps Real Data Sources

## 📊 Available Dataset Scripts

### 🟢 Primary: Real Production AIOps Data
**`scripts/download_real_aiops_data.py`** ⭐ **RECOMMENDED**
- **LogHub Benchmark Data**: HDFS, BGL, Thunderbird, OpenStack
- **Scale**: Millions of log lines from real production systems
- **Focus**: Anomaly detection, system failures, performance issues
- **Industries**: Telecommunications, Cloud Infrastructure, Distributed Systems
- **Usage**: `python scripts/download_real_aiops_data.py`

### 🟡 Secondary: Cloud-Native Specific Data  
**`scripts/download_cloud_aiops_data.py`**
- **OpenStack Logs**: 207K+ real cloud infrastructure logs (58.6MB)
- **Hadoop Cluster**: 394K+ distributed computing logs (16.3MB)  
- **Kubernetes Security**: ~50K K8s network traffic & anomaly data
- **Microservices**: 20 real microservice project dependency graphs
- **Focus**: AWS, K8s, Cloud-native applications
- **Usage**: `python scripts/download_cloud_aiops_data.py`

## 🎯 Perfect for Your Use Case
- ✅ **Real Production Data** (no synthetic)
- ✅ **Cloud & K8s Focus** (AWS, OpenStack, Kubernetes)
- ✅ **System Operations** (logging, monitoring, troubleshooting)
- ✅ **Scalable** (millions of real log entries)
- ✅ **Industry Standard** (LogHub benchmark datasets)

## 🚀 Quick Start
```bash
# Download core AIOps production data
python scripts/download_real_aiops_data.py

# Download additional cloud-native data  
python scripts/download_cloud_aiops_data.py
```

Both scripts output training-ready JSONL files for knowledge distillation.