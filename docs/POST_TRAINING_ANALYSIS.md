# Post-Training Analysis and Available Commands

This document describes all post-training analysis tools and commands available in the microwakeword_trainer framework.

---

## 🚨 Your Training Results Analysis

Your results show **suspiciously perfect metrics**:

```
Accuracy:  0.9999
Precision: 1.0000
Recall:    0.9997
F1 Score:  0.9998
FA/Hour:   0.00
```

### ⚠️ Why This Is Suspicious

**Statistical Reality:**
- **FAH = 0.00** with 83,720 negative samples is **virtually impossible**
- Only **5 false negatives** out of 14,299 positives is suspiciously low
- **Zero false positives** is statistically improbable

**Likely Causes:**

1. **Data Leakage (Most Likely)**
   - Same audio files in both train and validation sets
   - Speaker overlap (same person in train and val)
   - Augmented versions of same files counted as different samples

2. **Validation Set Too Small**
   - Your validation set might be too small to represent real-world diversity
   - Check: `ls -la dataset/positive/` vs `ls -la dataset/negative/`

3. **Overfitting**
   - Model memorized the validation set
   - 70,000 steps is a lot - model might have overfitted

4. **Incorrect Labeling**
   - Negative samples might actually contain wake word instances
   - Background noise might be too clean

### 🔍 How to Verify

```bash
# 1. Check dataset splits
python -c "
from src.data.ingestion import load_clips, Split
clips = load_clips('config/presets/standard.yaml')
print(f'Train: {len(clips.get_split(Split.TRAIN))}')
print(f'Val:   {len(clips.get_split(Split.VAL))}')
print(f'Test:  {len(clips.get_split(Split.TEST))}')
"

# 2. Evaluate on completely separate test data
python scripts/evaluate_model.py \
    --checkpoint models/checkpoints/best_weights.weights.h5 \
    --config standard \
    --split test \
    --analyze

# 3. Export and verify TFLite compatibility
python scripts/verify_esphome.py models/exported/wake_word.tflite --verbose
```

---

## 📊 Post-Training Commands

### 1. Model Export
```bash
# Export to TFLite
mww-export \
    --checkpoint models/checkpoints/best_weights.weights.h5 \
    --output models/exported/

# Without quantization (for debugging)
mww-export \
    --checkpoint models/checkpoints/best.ckpt \
    --output models/exported/ \
    --no-quantize
```

### 2. ESPHome Compatibility Verification
```bash
# Basic check
python scripts/verify_esphome.py models/exported/wake_word.tflite

# Verbose output
python scripts/verify_esphome.py models/exported/wake_word.tflite --verbose

# JSON output for CI/CD
python scripts/verify_esphome.py models/exported/wake_word.tflite --json
```

### 3. Model Evaluation (NEW)
```bash
# Evaluate on test set
python scripts/evaluate_model.py \
    --checkpoint models/checkpoints/best.ckpt \
    --config standard \
    --split test \
    --analyze

# Evaluate TFLite model
python scripts/evaluate_model.py \
    --tflite models/exported/wake_word.tflite \
    --config standard \
    --split test

# Output as JSON
python scripts/evaluate_model.py \
    --checkpoint models/checkpoints/best.ckpt \
    --config standard \
    --json
```

### 4. Auto-Tuning (Fine-tuning)
```bash
# Fine-tune for better FAH/recall
mww-autotune \
    --checkpoint models/checkpoints/best.ckpt \
    --config standard \
    --target-fah 0.2 \
    --target-recall 0.95

# With custom iterations
mww-autotune \
    --checkpoint models/checkpoints/best.ckpt \
    --config standard \
    --max-iterations 50
```

### 5. Model Analysis
```bash
# Detailed model report
python -c "
from src.export.model_analyzer import analyze_model_architecture
results = analyze_model_architecture('models/exported/wake_word.tflite')
print(results)
"
```

---

## 📁 Test Dataset Usage

### Current State
The framework **does define** a TEST split, but it's **not actively used** in training:

```python
# From src/data/ingestion.py
class Split(Enum):
    TRAIN = "train"
    VAL = "validation"
    TEST = "test"
```

**During Training:**
- Only TRAIN and VAL splits are used
- TEST split is set aside but not evaluated

**Post-Training:**
- You should manually evaluate on TEST split
- This gives unbiased performance estimate

### How to Use Test Split

```bash
# Evaluate on test set (NEW script)
python scripts/evaluate_model.py \
    --checkpoint models/checkpoints/best.ckpt \
    --config standard \
    --split test \
    --analyze
```

Or programmatically:

```python
from src.data.dataset import WakeWordDataset
from src.evaluation.metrics import MetricsCalculator
import numpy as np

# Load dataset
dataset = WakeWordDataset(config)
dataset.build()

# Get test generator
test_gen = dataset.test_generator_factory(max_time_frames)()

# Evaluate
y_true = []
y_scores = []

for features, labels in test_gen:
    predictions = model.predict(features)
    y_true.extend(labels)
    y_scores.extend(predictions)

# Calculate metrics
calc = MetricsCalculator(y_true=np.array(y_true), y_score=np.array(y_scores))
metrics = calc.compute_all_metrics(ambient_duration_hours=10.0)
```

---

## 🎯 Recommended Post-Training Workflow

### Step 1: Verify Export
```bash
python scripts/verify_esphome.py models/exported/wake_word.tflite --verbose
```

### Step 2: Evaluate on Test Set
```bash
python scripts/evaluate_model.py \
    --checkpoint models/checkpoints/best.ckpt \
    --config standard \
    --split test \
    --analyze
```

### Step 3: Check for Data Leakage
```bash
# Compare train/val/test speaker overlap
python cluster-Test.py --config standard --dataset all
```

### Step 4: Real-World Testing
- Export model to ESP32
- Test with real audio recordings
- Check actual false activation rate

---

## 🐛 Debugging Suspicious Results

### If metrics are too good:

1. **Check Data Leakage:**
   ```bash
   # Compare file lists
   find dataset/positive/train -name "*.wav" | sort > train_files.txt
   find dataset/positive/val -name "*.wav" | sort > val_files.txt
   comm -12 train_files.txt val_files.txt  # Should be empty
   ```

2. **Check Speaker Overlap:**
   ```bash
   python cluster-Test.py --config standard
   # Review cluster_output/*_cluster_report.txt
   ```

3. **Visualize Predictions:**
   ```python
   import matplotlib.pyplot as plt
   
   # Plot prediction distribution
   plt.hist(y_scores[y_true == 0], bins=50, alpha=0.5, label='Negative')
   plt.hist(y_scores[y_true == 1], bins=50, alpha=0.5, label='Positive')
   plt.legend()
   plt.savefig('prediction_distribution.png')
   ```

4. **Check Augmentation:**
   - If augmentation is too aggressive, model might see "easy" versions
   - Verify augmentation parameters in config

---

## 📋 Summary of Available Commands

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `mww-train` | Train model | Initial training |
| `mww-export` | Export to TFLite | After training |
| `mww-autotune` | Fine-tune model | If metrics need improvement |
| `scripts/verify_esphome.py` | Verify TFLite compatibility | After export |
| `scripts/evaluate_model.py` | Evaluate on test set | Post-training validation |
| `cluster-Test.py` | Speaker clustering | Data preparation |
| `scripts/generate_test_dataset.py` | Generate synthetic data | Testing pipeline |

---

## ⚠️ Your Next Steps

Given your suspicious results:

1. **Run verification:**
   ```bash
   python scripts/verify_esphome.py models/exported/wake_word.tflite
   ```

2. **Check test set performance:**
   ```bash
   python scripts/evaluate_model.py \
       --checkpoint models/checkpoints/best.ckpt \
       --config standard \
       --split test \
       --analyze
   ```

3. **If still suspicious, re-train with:**
   - Verified speaker separation
   - Smaller training steps (e.g., 20k instead of 70k)
   - More aggressive data augmentation

4. **Real-world test:**
   - Deploy to ESP32
   - Test with actual microphone input
   - Count real false activations per hour
