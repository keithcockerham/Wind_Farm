# GPU Acceleration for XGBoost Pipeline

## Quick Start

### Enable GPU Acceleration

```python
from autonomous_feature_selection import run_autonomous_xgboost_pipeline

# Enable GPU acceleration
results = run_autonomous_xgboost_pipeline(
    farm='C',
    scada_data=scada,
    event_info=event_info,
    use_gpu=True,       # Enable GPU
    gpu_id=0            # GPU device ID (default: 0)
)
```

**Output:**
```
================================================================================
AUTONOMOUS XGBOOST PIPELINE - FARM C
================================================================================
🚀 GPU ACCELERATION ENABLED (Device: cuda:0)

PHASE 1: AUTOMATIC FEATURE SELECTION
...
```

---

## Requirements

### 1. Install CUDA-enabled XGBoost

```bash
pip install xgboost[cuda]
```

Or if already installed:
```bash
pip uninstall xgboost
pip install xgboost[cuda]
```

### 2. Verify GPU is Available

```python
import xgboost as xgb

# Check XGBoost sees your GPU
print(f"XGBoost version: {xgb.__version__}")

# Test GPU training
from xgboost import XGBClassifier
import numpy as np

X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)

model = XGBClassifier(tree_method='hist', device='cuda:0')
model.fit(X, y)
print("✓ GPU training successful!")
```

### 3. Check NVIDIA GPU

```bash
# Windows
nvidia-smi

# Output should show your GPU
```

---

## Performance Comparison

### Without GPU (CPU)

```python
import time

start = time.time()
results_cpu = run_autonomous_xgboost_pipeline(
    farm='C',
    scada_data=scada,
    event_info=event_info,
    use_gpu=False  # CPU
)
cpu_time = time.time() - start
print(f"CPU Training Time: {cpu_time:.1f}s")
```

### With GPU (CUDA)

```python
start = time.time()
results_gpu = run_autonomous_xgboost_pipeline(
    farm='C',
    scada_data=scada,
    event_info=event_info,
    use_gpu=True  # GPU
)
gpu_time = time.time() - start
print(f"GPU Training Time: {gpu_time:.1f}s")
print(f"Speedup: {cpu_time/gpu_time:.1f}x")
```

**Expected Results (Farm C, 27 events):**
- CPU: ~300-600 seconds
- GPU: ~60-120 seconds
- **Speedup: 3-5x faster**

---

## Multi-GPU Setup

If you have multiple GPUs:

```python
# Use GPU 0
results = run_autonomous_xgboost_pipeline(
    farm='A', scada=scada_a, event_info=event_a,
    use_gpu=True, gpu_id=0
)

# Use GPU 1
results = run_autonomous_xgboost_pipeline(
    farm='B', scada=scada_b, event_info=event_b,
    use_gpu=True, gpu_id=1
)
```

Or process multiple farms in parallel:

```python
from concurrent.futures import ThreadPoolExecutor

def train_farm(farm_id, gpu_id):
    scada = get_farm_scada_chunked(farm=farm_id)
    event_info = get_event_info(farm=farm_id)
    
    return run_autonomous_xgboost_pipeline(
        farm=farm_id,
        scada_data=scada,
        event_info=event_info,
        use_gpu=True,
        gpu_id=gpu_id
    )

# Train 3 farms on 3 GPUs simultaneously
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(train_farm, 'A', 0),
        executor.submit(train_farm, 'B', 1),
        executor.submit(train_farm, 'C', 2)
    ]
    results_all = [f.result() for f in futures]
```

---

## Troubleshooting

### Error: "No CUDA-capable device is detected"

**Problem:** XGBoost can't find GPU

**Solutions:**
1. Check GPU is working:
   ```bash
   nvidia-smi
   ```

2. Verify CUDA drivers installed:
   ```bash
   nvcc --version
   ```

3. Reinstall CUDA-enabled XGBoost:
   ```bash
   pip uninstall xgboost
   pip install xgboost[cuda]
   ```

4. Check CUDA version compatibility:
   - XGBoost requires CUDA 11.x or 12.x
   - Check: `nvidia-smi` shows CUDA version

### Error: "tree_method='hist' is not available"

**Problem:** Installed CPU-only XGBoost

**Solution:**
```bash
pip uninstall xgboost
pip install xgboost[cuda]
```

### Error: Out of memory

**Problem:** GPU doesn't have enough memory

**Solutions:**
1. Reduce batch size (XGBoost handles automatically)
2. Use smaller max_features:
   ```python
   results = run_autonomous_xgboost_pipeline(
       farm='C', scada=scada, event_info=event_info,
       use_gpu=True,
       max_features=10  # Reduce from 15
   )
   ```

3. Use CPU for large datasets:
   ```python
   results = run_autonomous_xgboost_pipeline(
       farm='C', scada=scada, event_info=event_info,
       use_gpu=False  # Fall back to CPU
   )
   ```

### Slower with GPU than CPU?

**Possible reasons:**
1. **Small dataset:** GPU overhead > speedup for small data
   - Farm B (6 failures): CPU might be faster
   - Farm C (27 failures): GPU should be faster

2. **Data transfer overhead:** Moving data to GPU takes time
   - More noticeable with many small CV folds

3. **Old GPU:** Older GPUs (e.g., GTX 10-series) may not be faster

**When to use GPU:**
- Large datasets (>50K rows)
- Many features (>20)
- Many trees (n_estimators > 100)
- Multiple farms to train

**When CPU is fine:**
- Small datasets (<10K rows)
- Few features (<10)
- Quick experiments

---

## Best Practices

### 1. Always Verify Results Match

```python
# Train on CPU
results_cpu = run_autonomous_xgboost_pipeline(
    farm='C', scada=scada, event_info=event_info, use_gpu=False
)

# Train on GPU
results_gpu = run_autonomous_xgboost_pipeline(
    farm='C', scada=scada, event_info=event_info, use_gpu=True
)

# Verify same results (within numerical precision)
assert results_cpu['results']['detected'].equals(results_gpu['results']['detected'])
print("✓ CPU and GPU results match")
```

### 2. Profile Training Time

```python
import time

def benchmark_training(use_gpu=False):
    start = time.time()
    results = run_autonomous_xgboost_pipeline(
        farm='C', scada=scada, event_info=event_info,
        use_gpu=use_gpu
    )
    elapsed = time.time() - start
    return elapsed, results

cpu_time, results_cpu = benchmark_training(use_gpu=False)
gpu_time, results_gpu = benchmark_training(use_gpu=True)

print(f"CPU: {cpu_time:.1f}s")
print(f"GPU: {gpu_time:.1f}s")
print(f"Speedup: {cpu_time/gpu_time:.1f}x")
```

### 3. Monitor GPU Usage

```python
# In separate terminal
watch -n 1 nvidia-smi

# Look for:
# - GPU utilization % (should be high during training)
# - Memory usage
# - Temperature
```

---

## Expected Performance

### Farm A (12 failures, 81 sensors, ~15 features)
- **CPU:** ~60-120s
- **GPU:** ~30-60s
- **Speedup:** 2x

### Farm B (6 failures, 257 sensors, ~10 features)
- **CPU:** ~30-60s
- **GPU:** ~20-40s
- **Speedup:** 1.5x (small dataset, GPU overhead)

### Farm C (27 failures, 952 sensors, ~15 features)
- **CPU:** ~300-600s
- **GPU:** ~60-120s
- **Speedup:** 4-5x

---

## Hardware Requirements

**Minimum:**
- NVIDIA GPU with CUDA support (GTX 10-series or newer)
- 4GB GPU memory
- CUDA 11.x or 12.x drivers

**Recommended:**
- NVIDIA RTX 20-series or newer
- 8GB+ GPU memory
- Latest CUDA drivers

**Your Setup (Example):**
Check your GPU:
```python
import torch  # If you have PyTorch installed
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

## Summary

**To enable GPU acceleration:**

```python
results = run_autonomous_xgboost_pipeline(
    farm='C',
    scada_data=scada,
    event_info=event_info,
    use_gpu=True,      # ← Add this
    gpu_id=0           # ← Optional, default is 0
)
```

**That's it!** The pipeline automatically configures XGBoost for GPU training.

**Expected speedup:** 2-5x faster depending on dataset size and GPU.
