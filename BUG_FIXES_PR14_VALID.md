# Valid Bug Fix Proposals from PR #14 - V2.0.0

Generated: March 17, 2026
PR: https://github.com/sarpel/microwakeword_trainer/pull/14
Status: **NOT STARTED** - Review required before implementing

---

## Summary

- **🔴 Critical Issues:** 6
- **🟠 Major Issues:** 34
- **🟡 Minor Issues:** 4
- **✅ Resolved Issues:** 8 (excluded from this list)
- **Total Valid Unresolved:** 44 issues

**Key Findings:**
- Multiple issues in `src/tuning/orchestrator.py` indicate auto-tuning is incomplete/non-functional
- GPU memory management in `src/data/spec_augment_gpu.py` is completely broken
- CLI parameter propagation from `src/tuning/cli.py` doesn't work
- Data pipeline issues in `src/data/dataset.py` affect both training and validation

---

## 🔴 CRITICAL ISSUES (6)

### 1. GPU Memory Cleanup Completely Broken
**File:** `src/data/spec_augment_gpu.py` (lines 162-165)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
MemoryAsyncPool is configured at startup for CuPy, but the cleanup code calls `cp.get_default_memory_pool().free_all_blocks()` which is completely ineffective. The MemoryAsyncPool instance lives in a different allocation context (CUDA stream-ordered allocator), so calling the default memory pool's cleanup doesn't release the GPU memory that was actually allocated.

**Impact:**
- Memory leaks during GPU augmentation operations
- Possible OOM (out of memory) errors during training
- GPU memory not released between augmentation calls

**Evidence from PR:**
```
GPU belleğinin temizliğinde hata: MemoryAsyncPool çalışırken default pool temizlemesi işe yaramıyor. Bundan sonra GPU'da veri ayırdığında, o bellek MemoryAsyncPool kutusuna gidiyor, default kutuya değil.
```

**Suggested Fix:**
Replace the default-pool cleanup with a call to the actual MemoryAsyncPool instance, or remove it entirely (similar to `spec_augment_gpu()` which removed it).

---

### 2. CLI Silently Ignores Most Tuner Flags
**File:** `src/tuning/cli.py` (lines 267-270)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
The CLI writes tuner parameters to a flat dict structure called `at`, but `MicroAutoTuner` and its `_get_cfg()` method expects those options to be nested under the `auto_tuning_expert` namespace. As a result, critical flags like `population_size`, `micro_burst_steps`, `knob_cycle`, and `max_iterations` are silently ignored.

**Impact:**
- Users cannot configure auto-tuning parameters via CLI
- Critical tuner functionality is unavailable
- All expert-level knobs are ignored

**Evidence from PR:**
```
CLI şu an tuner ayarlarının büyük kısmını sessizce yok sayıyor. Burada `at` düz bir `dict` olarak `MicroAutoTuner`'a gidiyor, ama `src/tuning/orchestrator.py` içindeki `_get_cfg()` ayarları `getattr(...)` ile okuyor.
```

**Suggested Fix:**
Structure the CLI parameters to nest them under `auto_tuning_expert`:
```python
at = {
    **config_dict.get("auto_tuning", {}),
    **config_dict.get("auto_tuning_expert", {}),
}
if getattr(args, "population_size", None) is not None:
    at["auto_tuning_expert"]["population_size"] = args.population_size
# ... same for other flags
```

---

### 3. Help Panel Crashes Without Error Handling
**File:** `src/tools/help_panel.py` (lines 65-72)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
The import of `RichTrainingLogger` is wrapped in try-except, but the instantiation and `log_next_steps()` call are outside of it. Any error during instantiation or method execution will cause the CLI to crash instead of falling back to `_print_fallback_next_steps()`.

**Impact:**
- Unreliable post-training guidance
- CLI crashes prevent users from getting next steps
- Poor user experience after training

**Evidence from PR:**
```
Satır 65-69 sadece `ImportError` yakalıyor. Satır 71'de `RichTrainingLogger()` oluşturulması veya satır 72'de `log_next_steps()` çağrısı başarısız olursa, CLI direkt crash olur ve fallback'e ulaşamaz.
```

**Suggested Fix:**
Wrap the entire sequence in a single try-except:
```python
try:
    from src.training.rich_logger import RichTrainingLogger
    logger = RichTrainingLogger()
    logger.log_next_steps(checkpoint, args.config)
except Exception:
    _print_fallback_next_steps(checkpoint, args.config)
    return
```

---

### 4. Verification Script Crashes on Artifact Path
**File:** `scripts/verify_esphome.py` (lines 1238, 1357)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
`datetime` is imported inside a conditional block (line 1238), but `datetime.now()` is called unconditionally later (line 1357). If artifacts exist, the import never happens and the code raises a `NameError` when trying to use `datetime`.

**Impact:**
- Verification pipeline completely broken when artifacts exist
- Cannot verify exported models
- Makes verification workflow unusable

**Evidence from PR:**
```
Line 1238'de `datetime` import edilir AMA sadece artifact zaten varsa. Ama Line 1357'de ise daima `datetime.now()` kullanılır. Eğer artifact yoksa? İşte o zaman kod "datetime'ı tanımıyorum" diye hata verir.
```

**Suggested Fix:**
Move the datetime import to the top of the file (or outside all conditionals):
```python
from datetime import datetime  # Move to top level
# ... rest of code with conditional imports
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
```

---

### 5. Auto-Cutoff Resolver Bypasses Canonical Metadata
**File:** `src/export/manifest.py` (lines 103-149)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
The current auto-cutoff logic scans `*.metadata.json` files using `os.path.commonprefix` to find a probability cutoff. However, `src/export/tflite.py` already uses the canonical source `config['checkpoint_metadata']` for reading cutoff values. This bypass can:
1. Pick the wrong sidecar file if multiple metadata files exist
2. Fail to use the canonical metadata field
3. Cause cutoff mismatches between manifest and evaluation

**Impact:**
- Wrong probability cutoff used in verification
- Incorrect manifest generation
- Inconsistent with `scripts/evaluate_model.py` which uses `evaluation.default_threshold`

**Evidence from PR:**
```
Auto-cutoff çözümü canonical metadata kaynağını bypass ediyor. Bu resolver `config['checkpoint_metadata']`ı hiç okumadan `tflite_path` çevresinde `*.metadata.json` arıyor ve karakter-bazlı `commonprefix` ile eşleştiriyor.
```

**Suggested Fix:**
Check canonical `config['checkpoint_metadata']` first, only fall back to filesystem scan:
```python
def resolve_probability_cutoff(config: dict, tflite_path: Path) -> float:
    # First check canonical source
    checkpoint_metadata = config.get('checkpoint_metadata', {})
    cutoff = checkpoint_metadata.get('tuned_probability_cutoff') or checkpoint_metadata.get('probability_cutoff')
    if cutoff is not None:
        try:
            cutoff = float(cutoff)
            if 0.0 < cutoff <= 1.0:
                return cutoff
        except (ValueError, TypeError):
            pass

    # Only fall back to filesystem if canonical missing
    # existing fallback logic...
```

---

### 6. Pipeline Can Select Unloadable Checkpoint Shards
**File:** `src/pipeline.py` (lines 67-128)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
The candidate selection logic includes `.ckpt.data-*` shard files and `.ckpt.index` files directly in the candidate list. These are not directly loadable as checkpoint files. Additionally, `src/export/tflite.py` metadata scanning expects HDF5 weight files. This can cause:
1. Selection of incomplete checkpoint shards
2. Export failures with "newest" checkpoint selection
3. Incompatibility with downstream export logic

**Impact:**
- Export/auto-tune can fail when "newest" artifact is a shard
- Pipeline selects unloadable files
- Breaks export workflow

**Evidence from PR:**
```
Aday listesine `.ckpt.data-*` ve `.ckpt.index` dosyalarını ekleyip en yenisini doğrudan "checkpoint" diye döndürüyorsunuz. Özellikle `.ckpt.data-*` tek başına yüklenebilir checkpoint değil; ayrıca `src/export/tflite.py` metadata taramasında HDF5 ağırlık dosyası bekliyor.
```

**Suggested Fix:**
Only consider supported, loadable artifacts:
1. Include HDF5 weight files (`*.weights.h5`)
2. Include checkpoint prefixes only when matching `.ckpt.index` exists
3. Explicitly ignore `.ckpt.data-*` shards when building candidates
4. Centralize `.index -> prefix` normalization

---

## 🟠 MAJOR ISSUES (34)

### 7. Causal Padding Applied Wrong Side (Dynamic Rank)
**File:** `src/model/streaming.py` (line 403)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
For dynamic-rank tensors, when `pad_time_dim == "causal"`, the code applies padding to index `[1][1]` (right side) instead of `[1][0]` (left side). The static-rank path correctly applies left padding. This inconsistency causes streaming inference to produce wrong results with dynamic batches.

**Additional Issue:**
The static-rank code (lines 404-420) also allows `pad_time_dim == "same"` in TRAINING mode, which violates project rules forbidding "same" on time axis.

**Impact:**
- Streaming inference produces incorrect results
- Causal convolution semantics broken for dynamic batches
- Inconsistency between static and dynamic rank handling

**Evidence from PR:**
```
Dinamik yol (satır 403): `pad[1][1] = pad_total` → sağ padding `[0, pad_total]`
Statik yol (satır 415): `pad[1] = [pad_total, 0]` → sol padding `[pad_total, 0]`
```

**Suggested Fix:**
Change dynamic-rank causal padding to match static-rank:
```python
if self.pad_time_dim == "causal":
    pad = tf.tensor_scatter_nd_update(pad, [[1, 0]], [pad_total])  # Left padding, not right
```

Also disallow `pad_time_dim == "same"` unconditionally:
```python
elif self.pad_time_dim == "same":
    raise ValueError("pad_time_dim='same' is not allowed on time axis in this project. Use 'causal'.")
```

---

### 8. Generator Creates Infinite Loop on Empty Batches
**File:** `src/data/dataset.py` (lines 1238-1265)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
When `infinite=False` and every sample in the epoch fails (e.g., corrupted files, decoding errors), `batch_idx` stays at 0. The `continue` statement on line 1242 prevents reaching the `break` statement. This causes the generator to loop forever without yielding any data.

**Impact:**
- Validation/test hangs indefinitely on data errors
- Training may never complete if all samples fail
- Cannot detect or recover from data issues
- Makes training unreliable

**Evidence from PR:**
```
Eğer bir epoch'taki tüm örnekler hata verirse (dosya bozuksa mesela), `batch_idx` sıfırda kalır. Sonra 142. satırdaki `continue` komutu programı döngünün başına geri gönderiyor ve 145. satırdaki `break` hiç çalışmıyor.
```

**Suggested Fix:**
When batch is empty and not infinite, break instead of continue:
```python
if batch_idx == 0:
    if infinite:
        continue  # Skip empty batches for infinite streams
    else:
        break  # Exit if finite and nothing to yield
# Continue to yield partial batch logic...
```

---

### 9. Held-Out Test Evaluates Wrong Weights
**File:** `src/training/trainer.py` (lines 2231-2239)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
The test block evaluates `self.model` after restoring training state (line 2206). However, the actual artifacts on disk (`final_weights.weights.h5` or `best_weights.weights.h5`) may use EMA weights or have other differences. The logged test metrics don't reflect the actual saved model.

**Impact:**
- Test metrics in logs don't match exported model
- Cannot trust evaluation results
- Inconsistency between logged metrics and actual model performance

**Evidence from PR:**
```
Line 2206'dan sonra raw weights geri yüklenmiş oluyor; bu blok `TestEvaluator(self.model, ...)` ile devam ettiği için test özeti `final_weights.weights.h5` içindeki EMA ağırlıklarını veya `best_weights.weights.h5`i temsil etmiyor.
```

**Suggested Fix:**
Load the actual artifact weights before running TestEvaluator:
```python
# After checkpoint selection:
test_weights_path = checkpoint_dir / selected_checkpoint_file
self.model.load_weights(test_weights_path)
test_evaluator = TestEvaluator(self.model, ...)
# ... run test evaluation
```

---

### 10. Periodic Checkpoints Don't Save EMA Snapshot
**File:** `src/training/trainer.py` (lines 1187-1192)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
When EMA is enabled (`training.ema_decay` configured), the `checkpoint_step_*.weights.h5` files are saved with raw training weights instead of the EMA snapshot. The `weights_snapshot` mechanism only works for `best_weights.weights.h5`. Periodic checkpoints thus contain non-EMA weights.

**Impact:**
- Checkpoints use different weights than displayed metrics
- Test/evaluation from checkpoints is incorrect
- Inconsistent with EMA-based `best_weights.weights.h5`
- Violates EMA contract

**Evidence from PR:**
```
`weights_snapshot` sadece `best_weights.weights.h5` yolunda kullanılıyor. Line 1192'de model doğrudan kaydedildiği için, EMA açıkken `checkpoint_step_*.weights.h5` dosyaları validate edilen EMA artefact'ı değil ham training state oluyor.
```

**Suggested Fix:**
Apply `weights_snapshot` or swap to EMA before saving periodic checkpoints:
```python
if training.ema_decay is not None:
    # Use weights_snapshot or call _swap_to_ema_weights()
    checkpoint_weights = weights_snapshot if weights_snapshot else self.model.get_weights()
    # ... save checkpoint_weights to checkpoint_step_{step}.weights.h5
```

---

### 11. CLI --num-positive and --num-negative Ignore Input
**File:** `scripts/generate_test_dataset.py` (lines 165-176)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
The CLI arguments `--num-positive` and `--num-negative` are parsed and shown in help, but the functions `create_positive_samples()` and `create_negative_samples()` still use hardcoded values (10 and 50 respectively).

**Impact:**
- User cannot control sample generation counts
- CLI interface is misleading
- Sample size always the same regardless of flags

**Evidence from PR:**
```
Argümanlar parse ediliyor ama üretim fonksiyonlarına hiç geçilmiyor; `create_positive_samples()` hâlâ sabit 10, `create_negative_samples()` hâlâ sabit 50 örnek üretiyor.
```

**Suggested Fix:**
Pass parsed values to generation functions:
```python
num_positive = args.num_positive
num_negative = args.num_negative
create_positive_samples(num_positive)
create_negative_samples(num_negative)
```

---

### 12. Script Fails: Global Used Before Declaration
**File:** `scripts/generate_test_dataset.py` (lines 161-188)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
The `main()` function uses `DATASET_DIR` at line 161-162 before the `global DATASET_DIR` declaration at line 188. Python raises a SyntaxError: "name 'DATASET_DIR' is used prior to global declaration".

**Impact:**
- Script cannot run at all
- Fails immediately on import/execution
- Complete script non-functional

**Evidence from PR:**
```
`main()` fonksiyonunda `DATASET_DIR` değişkenini parser ayarlarında kullanıyorsun (161. ve 162. satırlar), ama `global` olarak ilan ettiğiniz yer daha sonra geliyor (188. satır).
```

**Suggested Fix:**
Move global declaration before first use:
```python
def main(args: argparse.Namespace | None = -> int:
    """Main entry point with CLI argument parsing."""
    global DATASET_DIR, POSITIVE_DIR, NEGATIVE_DIR, HARD_NEGATIVE_DIR  # Move to top
    # ... rest of code
```

---

### 13. Exit Code 4 Should Be Exit Code 2
**File:** `scripts/verify_streaming.py` (help text)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
The help/documentation text says `4=incompatible`, but the actual scripts (`check_esphome_compat.py` and `verify_esphome.py`) use exit code `2` for incompatibility failures. This creates confusion and violates the standard exit code contract.

**Impact:**
- Misleading documentation
- CI/automation may expect wrong exit codes
- Difficulty using help to understand behavior

**Evidence from PR:**
```
Gerçek scriptlerde (`check_esphome_compat.py` ve `verify_esphome.py`) uyumluluk hatası için çıkış kodu `2` kullanılıyor, ama yardım metni `4=incompatible` diyor.
```

**Standard Exit Code Contract:**
- `0` = success
- `1` = runtime/internal error
- `2` = validation/compatibility failure
- `4` = incorrect (should not be used)

**Suggested Fix:**
Update help text to use correct exit code:
```python
# Change "exit 4 (incompatible)" to "exit 2 (incompatible)"
# Standardize exit codes across all scripts
```

---

### 14. Exit Code 1 for Validation Failure Should Be 2
**File:** `scripts/verify_streaming.py` (lines 267-275)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
The validation failure path returns `1`, which conflates validation failures with runtime errors. According to coding guidelines, validation failures should return `2` to distinguish from actual bugs.

**Impact:**
- Cannot distinguish validation failures from bugs in CI
- Incorrect error categorization
- Makes debugging harder

**Evidence from PR:**
```
Check'ler fail olduğunda burada `1` dönüyor. Bu da hata (runtime hatası) ile uyumsuzluk hatasını birbirine karıştırıyor.
```

**Suggested Fix:**
Return `2` for validation failures:
```python
# Change:
return 0 if results["passed"] else 1
# To:
return 0 if results["passed"] else 2
```

---

### 15. Hardcoded Absolute Path Breaks CI
**File:** `tests/unit/test_tuning_integration.py` (line 27)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
The test uses a hardcoded absolute path `/home/sarpel/microwakeword_trainer-micro-autotuner-redesign` instead of computing it dynamically. This will fail in CI/CD environments where that path doesn't exist.

**Impact:**
- Tests fail in CI environments
- Not portable across different machines
- Blocks automated testing workflow

**Evidence from PR:**
```
`cwd` argümanı literal `/home/sarpel/microwakeword_trainer-micro-autotuner-redesign` yolu kullanıyor. Bu yol sadece geliştiricinin makinende var, CI'de çalışmaz.
```

**Suggested Fix:**
Use `tmp_path` fixture or compute path dynamically:
```python
cwd=tmp_path  # Or use dynamic calculation
cwd=Path(__file__).resolve().parents[2]  # Repository root
```

---

### 16. Test Uses Path.cwd() Instead of tmp_path Fixture
**File:** `tests/unit/test_tuning_integration.py` (line 14)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
The test constructs `missing_checkpoint` using `Path.cwd()` instead of using the pytest `tmp_path` fixture. This means the test runs in the current working directory instead of an isolated temp directory.

**Impact:**
- Tests don't run in isolation
- Can pollute other tests or user directories
- Non-deterministic test behavior

**Evidence from PR:**
```
`Path.cwd()` çalışma klasörünü kullanıyor; pytest'ın sağladığı izole `tmp_path` fixture'ı kullanmıyor. Bu testlerin birbirbirini etkileyebilir.
```

**Suggested Fix:**
Use the `tmp_path` fixture:
```python
def test_dry_run_exits_cleanly(tmp_path) -> None:  # Add fixture
    missing_checkpoint = tmp_path / "missing_checkpoint.h5"  # Use fixture instead
```

---

### 17. Subprocess Call Missing Timeout
**File:** `tests/unit/test_tuning_integration.py` (line 27)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
The `subprocess.run()` call to execute the CLI has no timeout parameter. If the CLI hangs (infinite loop, deadlock, etc.), the test will run forever, blocking CI.

**Impact:**
- Tests can hang indefinitely
- Blocks CI/CD pipeline
- Makes debugging difficult (no timeout output)

**Evidence from PR:**
```
CLI komutunun hiç timeout'u yok; yani eğer program takılırsa test sonsuza kadar bekler ve CI'ı durdurur.
```

**Suggested Fix:**
Add timeout to subprocess call:
```python
result = subprocess.run(
    [...],
    capture_output=True,
    text=True,
    timeout=60,  # Add timeout (60 seconds)
    cwd=...
)
```

---

### 18. Random Seed Creates Identical Perturbations
**File:** `src/tuning/population.py` (lines 108-112)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
The `exploit_explore()` method creates a new `np.random.RandomState(42)` instance every time it's called. Since a new RandomState object is created each call with the same seed, it produces the exact same noise pattern. No actual "exploration" of different perturbation directions occurs.

**Called from:** `src/tuning/orchestrator.py`, potentially ~6 iterations per tuning run.

**Impact:**
- Auto-tuner cannot explore different perturbation directions
- Weight perturbations are always identical
- Exploration strategy is ineffective

**Evidence from PR:**
```
Satır 108'de `np.random.RandomState(42)` hardcoded olduğu için, bu metod her çağrıldığında aynı rastgele gürültü paternini üretiyor. Keşif için farklı yönleri denemesi gerekir.
```

**Suggested Fix:**
Use iteration count or shared RNG:
```python
def exploit_explore(..., iteration: int = 0, ...):
    seed = 42 + iteration  # Different seed each call
    rng = np.random.RandomState(seed)
    # or use shared RNG passed in as argument
```

---

### 19. _run_burst is Stub - No Gradient Updates
**File:** `src/tuning/orchestrator.py` (lines 153-174)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
The `_run_burst()` function contains only `losses.append(0.0)` in its loop. It accepts `model` and `optimizer` parameters but never:
1. Calls sampler to get batches
2. Runs forward pass through model
3. Uses tf.GradientTape to compute gradients
4. Calls optimizer.apply_gradients()

It's a complete dummy stub that doesn't actually train the model.

**Impact:**
- Surgical gradient bursts (20-200 step range) don't train
- Auto-tuning weight perturbation is non-functional
- Model parameters never updated during bursts

**Evidence from PR:**
```
Bu fonksiyon gradient güncellemesi uygulamaması gerekiyor ama şu anda sadece `losses.append(0.0)` yapıyor. Parametreleri veriliyor (model, optimizer) ama hiç kullanılmıyor.
```

**Suggested Fix:**
Implement actual gradient update loop:
```python
def _run_burst(model: tf.keras.Model, optimizer, ...):
    losses = []
    for step in range(micro_burst_steps):
        batch = sampler.sample_batch()
        with tf.GradientTape() as tape:
            outputs = model(batch.features, training=True)
            loss = compute_loss(outputs, batch.labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        losses.append(float(loss))
    # Keep existing lr scheduling and _freeze_bn/_unfreeze_bn calls
    return losses
```

---

### 20. _evaluate_candidate Ignores Model Predictions
**File:** `src/tuning/orchestrator.py` (lines 114-124)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
The `_evaluate_candidate()` function computes metrics from labels only (`search_eval_partition[1]`). It accepts a `model` parameter but never calls it to get predictions. The features (`search_eval_partition[0]`) are completely ignored.

**Impact:**
- Every candidate gets the same score regardless of model
- Auto-tuner cannot differentiate between models
- Optimization is entirely useless

**Evidence from PR:**
```
Bu fonksiyon tam olarak bunu yapıyor: `model` parametresini alıyor ama hiç kullanmıyor (çağırıp tahminler aldığı yok). Sadece etiketler üzerinde istatistik hesaplıyor.
```

**Suggested Fix:**
Call model to get predictions:
```python
def _evaluate_candidate(model, search_eval_partition, ...):
    features, labels = search_eval_partition[0], search_eval_partition[1]
    predictions = model.predict(features)  # or model(features) call
    # Compute metrics comparing predictions to labels
    positives = ...
    negatives = ...
    recall = ...
    fah = ...
    return TuneMetrics(...)
```

---

### 21. Fine Sweep Returns Arbitrary 0.5 When Target Unattainable
**File:** `src/tuning/metrics.py` (lines 613-645)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
The `_fine_sweep()` function initializes `best_threshold = 0.5` and only updates it when `fah <= target_fah`. If no threshold meets the target FAH, it returns the arbitrary hardcoded 0.5 instead of finding the most conservative threshold (lowest FAH).

**Impact:**
- Incorrect threshold selection when target is unattainable
- Most conservative threshold not returned
- Suboptimal model performance

**Evidence from PR:**
```
`_fine_sweep()` `best_threshold`'u `0.5` ile başlatıp sadece `fah <= target_fah` dalında güncelliyor. Yani hedef FAH imkânsızsa en düşük-FAH eşiğini seçmek yerine sabit `0.5` dönüyor.
```

**Suggested Fix:**
Track minimal FAH threshold alongside best recall:
```python
def _fine_sweep(...):
    best_recall = -1.0
    min_fah = float("inf")
    min_fah_threshold = None
    best_threshold = None

    for t in unique_scores:
        preds = (y_scores >= t).astype(int)
        fp = ...
        tp = ...
        fah = fp / max(ambient_hours, 1e-8)
        recall = tp / max(n_pos, 1) if n_pos > 0 else 0.0

        # Track minimal FAH across all thresholds
        if fah < min_fah:
            min_fah = fah
            min_fah_threshold = float(t)
        elif math.isclose(fah, min_fah) and recall > min_recall:
            min_fah = fah
            min_recall = recall
            min_fah_threshold = float(t)

        # Update best threshold for target FAH
        if fah <= target_fah and recall > best_recall:
            best_recall = recall
            best_threshold = float(t)

    return min_fah_threshold if best_threshold is None else best_threshold
```

---

### 22. Pareto Archive Duplicates Same Candidate ID
**File:** `src/tuning/metrics.py` (lines 95-115)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
The `try_add()` method doesn't filter out old entries for the same `candidate_id` before adding the new one. It also doesn't call `_is_diverse()`. This allows the same candidate to be added multiple times to the archive, causing:
1. Archive bloat
2. `max_size` fills with duplicates
3. Crowding prune can't find real frontier points

**Impact:**
- Archive contains duplicate candidates
- Storage waste
- Exploration effectiveness reduced

**Evidence from PR:**
```
`try_add()` mevcut `candidate_id` için eski girdiyi atmıyor ve `_is_diverse()` hiç devreye girmiyor. Orchestrator aynı adayı her iterasyonda yeniden eklediği için arşiv kopyalarla şişip `max_size` dolunca gerçekten farklı frontier noktalarını silebilir.
```

**Suggested Fix:**
Filter duplicates before appending:
```python
def try_add(self, metrics: TuneMetrics, candidate_id: str) -> bool:
    # Filter out old entry for same candidate_id
    new_archive = [
        (m, cid)
        for (m, cid) in self._archive
        if cid != candidate_id and not metrics.dominates(m)
    ]

    # Check if new metrics is dominated by any remaining
    is_dominated = any(m.dominates(metrics) for (m, _) in new_archive)

    if is_dominated:
        return False

    # Check diversity before adding
    if not self._is_diverse(metrics, new_archive):
        return False

    # Not dominated — add to archive
    new_archive.append((metrics, candidate_id))
    # ... prune by crowding
```

---

### 23. Test for FocusedSampler Doesn't Verify Correctness
**File:** `tests/unit/test_tuning_knobs.py` (lines 150-154)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
The test only asserts that `FocusedSampler.build_batch()` returns `None`. It doesn't create a deterministic dataset, apply the `search_eval_fraction` split (default 0.30) to produce `search_train` and `search_eval` partitions, or verify that the sampler only trains on `search_train` data.

**Impact:**
- Implementation bugs in sampler would pass silently
- Core sampler constraint not verified
- No actual correctness testing

**Evidence from PR:**
```
`build_batch() is None` beklentisi, `FocusedSampler`'ın hiç batch üretmemesini başarı sayıyor. Böyle olunca asıl kritik şey — örneklerin sadece `search_train` partition'ından gelmesi — hiç test edilmiyor.
```

**Suggested Fix:**
Create proper test with data split:
```python
def test_focused_sampler_correctness():
    # Create deterministic dataset
    dataset = create_deterministic_dataset()

    # Apply same split logic as auto-tuner
    search_train, search_eval = split_dataset(dataset, fraction=0.30)

    sampler = FocusedSampler(search_train)
    batch = sampler.build_batch()

    # Verify batch contains only search_train items
    assert all(item in search_train for item in batch)
    assert not any(item in search_eval for item in batch)
```

---

### 24. _random_from Returns Less Samples Than Requested
**File:** `src/tuning/autotuner.py` (lines 595-603)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
The `_random_from()` function uses `size=min(n, len(pool))` instead of `size=n` with `replace=True`. When the pool has fewer items than requested, it returns fewer samples without oversampling. This affects strategy mixing ratios.

**Impact:**
- Batch sizes inconsistent with request
- Strategy mixing ratios broken
- Fewer samples than expected

**Evidence from PR:**
```
`_random_from()` şu anda `size=min(n, len(pool))` kullandığı için eksik havuzda oversample etmiyorsunuz; sadece daha az örnek döndürüyorsunuz.
```

**Suggested Fix:**
Always request exact size with replace:
```python
def _random_from(pool: list, n: int, ...):
    indices = np.random.choice(pool, size=n, replace=len(pool) < n)
    return np.asarray(indices)
```

---

### 25. Pareto try_add Missing Diversity Check (Also affects line 117-128)
**File:** `src/tuning/metrics.py` (lines 95-115)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
The `try_add()` method never calls `_is_diverse()`. Near-duplicate candidates (similar metrics) are added to the archive without diversity filtering. This causes the archive to accumulate similar candidates, preventing proper exploration.

**Impact:**
- Archive accumulates similar candidates
- Early exploration is ineffective
- Crowding prune can't find diverse frontier

**Evidence from PR:**
```
`try_add()` `_is_diverse()`'i hiç çağrımıyor. Yakın-duplicate'lar eklenmeden önce çeşitlilik kontrolü yapılmıyor.
```

**Suggested Fix:**
Call diversity check before appending (same fix as issue 22).

---

### 26. Fallback Threshold Uses Wrong Default
**File:** `src/export/manifest.py` (lines 150-155)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
The manifest uses a hardcoded `0.5` as fallback when probability cutoff metadata is missing. However, `scripts/evaluate_model.py` uses `evaluation.default_threshold` (likely `0.5` but centralized). Having inconsistent defaults causes mismatches.

**Impact:**
- Inconsistent thresholds between manifest and evaluation
- May use wrong default if evaluation.default_threshold changes
- Confusing behavior across tools

**Evidence from PR:**
```
Metadata miss olduğunda burada doğrudan `0.5`e düşüyorsunuz. Ama `scripts/evaluate_model.py` aynı sentinelde önce `evaluation.default_threshold` kullanıyor.
```

**Suggested Fix:**
Use same constant or import from evaluation module:
```python
# Import from evaluation module or use shared constant
from src.evaluation.config import DEFAULT_THRESHOLD
# Or use:
evaluation_default = config.get('evaluation', {}).get('default_threshold', 0.5)
```

---

### 27. ROC and PR-AUC Use Different Sample Sets
**File:** `src/evaluation/metrics.py` (lines 634-649)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
PR-AUC filters labels with `y_true != 2` mask, but `auc_roc` uses the full arrays. This means the two AUC metrics are computed over different sample sets, making them incomparable and potentially misleading.

**Impact:**
- AUCs not comparable within same report
- Model ranking can be misleading
- Inconsistent with PR-AUC calculation

**Evidence from PR:**
```
Burada PR-AUC `y_true != 2` filtresiyle hesaplanıyor, ama `auc_roc` tüm diziyi gönderip `label == 2` örneklerini negatif sınıfa katıyor.
```

**Suggested Fix:**
Apply same filtering to both metrics:
```python
valid_mask = self.y_true != 2
auc_roc = compute_roc_auc(self.y_true[valid_mask], self.y_score[valid_mask])

y_true_pr = _binarize_labels(self.y_true[valid_mask])
auc_pr = compute_pr_auc(y_true_pr, self.y_score[valid_mask])
```

---

### 28. test_split Duration Scaling Inverted (5 locations)
**File:** `src/evaluation/test_evaluator.py` (lines 258-261)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
The code divides `self.ambient_duration_hours / self.test_split` instead of multiplying. For a 10% test split, this gives 10x the duration instead of 0.1x. FAH becomes overly optimistic by factor of 100.

**Additional locations:** Lines 289-293, 400-403, 444-447, 736-739

**Impact:**
- FAH overly optimistic (100x inflated)
- Incorrect evaluation metrics
- Misleading model comparison

**Evidence from PR:**
```
Aynı `training.ambient_duration_hours` kavramı `src/tuning/autotuner.py` içinde alt kümelere inerken `* split_fraction` ile küçültülüyor; burada ise `/ self.test_split` ile büyütülüyor.
```

**Suggested Fix:**
Create centralized helper and multiply:
```python
def scale_ambient_duration(hours: float, split_fraction: float) -> float:
    return hours * split_fraction  # Not divide

# Replace all occurrences:
scaled_duration = scale_ambient_duration(self.ambient_duration_hours, self.test_split)
```

---

### 29. Ring Buffer Can Return None State
**File:** `src/model/streaming.py` (lines 377-386)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
When `ring_buffer_size_in_time_dim` is falsy (0 or None), the code returns `input_state` directly which may be `None`. External state calls expect a 6-tensor tuple named `stream` through `stream_5`.

**Impact:**
- External state crashes when expecting 6-tensor tuple
- Incompatible state API
- Breaks streaming inference

**Evidence from PR:**
```
Buffer boyutu 0 iken ve runtime `state` verilmemişse bu değer `None` olabilir; external-state akışında state tensörü bekleyen yolu kırabilir.
```

**Suggested Fix:**
Ensure output_state is never None:
```python
if ring_buffer_size_in_time_dim:
    output = self.cell(inputs)
    state_update = input_state if input_state is not None else inputs[:, :0, ...]
    self.output_state = state_update
    return output, state_update
else:
    # Existing logic...
```

---

### 30. Verifier Uses Hardcoded Config
**File:** `scripts/verify_esphome.py` (lines 132-148)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
The verifier always loads the hardcoded "standard" preset to compute expected streaming state shapes. Models trained/exported with non-standard model parameters (different stride, kernel sizes, temporal_frames, etc.) will be reported as incompatible due to shape mismatches.

**Impact:**
- Cannot verify models trained with non-standard configs
- False incompatibility failures
- Blocks models with custom architectures

**Evidence from PR:**
```
`verify_esphome.py` her zaman "standard" preset'ini yükliyor; farklı model/hardware ayarıyla eğitilen modeller yanlışlıkla uyumsuz raporlanıyor.
```

**Suggested Fix:**
Add `--config/--override` argument to specify preset:
```python
parser.add_argument("--config", type=str, default="standard",
                   help="Config preset name (default: standard)")
config = load_full_config(args.config)
```

---

### 31. Lockfile Cleanup is Nonsensical
**File:** `src/data/tfdata_pipeline.py` (lines 102-126)
**Bot:** coderabbitai, copilot-pull-request-reviewer
**Status:** Unresolved

**Issue Description:**
Both branches of an if/else statement call `stale.append(lf)`. The conditional logic is meaningless - every lockfile is marked stale regardless of the condition. This deletes active lockfiles too, potentially corrupting concurrent tf.data caches.

**Impact:**
- Corrupts concurrent process caches
- Risk of race conditions
- Inefficient cleanup logic

**Evidence from PR:**
```
Hem elif dalı da `stale.append(lf)` çağırıyor - conditional'ın anlamı yok. Bütün lockfile'leri "stale" olarak işaretleniyor, aktif olanlar da dahil.
```

**Suggested Fix:**
Add age threshold or active check:
```python
for lf in lockfiles:
    # Check if lockfile is old enough
    if age(lf) > STALE_THRESHOLD_HOURS:
        stale.append(lf)
    # Or check for active process (more complex)
```

---

### 32. Generator Termination Never Reaches Break
**File:** `src/data/dataset.py` (train_generator_factory)
**Bot:** copilot-pull-request-reviewer
**Status:** Unresolved

**Issue Description:**
`train_generator_factory()` calls `_iter_split_batches(..., infinite=False)`. The non-tf.data training path (`use_tfdata: false`) passes this factory directly into `Trainer.train()`, which expects a repeating (infinite) stream. A finite generator will exhaust and can stop training early or raise `StopIteration`.

**Impact:**
- Non-tf.data training stops early
- Generator exhausts after one "epoch"
- Incompatible with training loop expectations

**Evidence from PR:**
```
`train_generator_factory()` şimdi `_iter_split_batches(..., infinite=False)` çağırıyor. Non-tf.data path (`use_tfdata: false`) bu factory'yi doğrudan `Trainer.train()`'e veriyor, tekrarlayan stream bekliyor. Sonuç: training erken durabilir.
```

**Suggested Fix:**
Add `repeat()` or create infinite wrapper:
```python
def train_generator_factory(...):
    base_gen = _iter_split_batches(..., infinite=False)

    if use_tfdata:
        return base_gen  # TF datasets are already repeating
    else:
        # Make it infinite for compatibility
        return itertools.cycle(base_gen)
```

---

### 33. Test Doesn't Assert Op Validation
**File:** `tests/unit/test_esphome_op_whitelist.py` (around line 72-82)
**Bot:** copilot-pull-request-reviewer, coderabbitai
**Status:** Unresolved

**Issue Description:**
The test only calls `inspect.signature(verify_tflite_model)` but doesn't assert anything about:
1. Expected parameters existing
2. Whitelist enforcement behavior
3. Validation result includes ops check

It only asserts that the function is callable, which doesn't test the actual whitelist functionality.

**Impact:**
- Whitelist regressions go undetected
- Test passes without actual validation
- No confidence in whitelist implementation

**Evidence from PR:**
```
Sadece fonksiyon callable olup olmadığını test ediyor; whitelist davranışı veya op kontrolü için hiç assertion yok.
```

**Suggested Fix:**
Add concrete assertions:
```python
sig = inspect.signature(verify_tflite_model)
assert "model_path" in sig.parameters
assert "allowed_ops" in sig.parameters  # or similar

# Create model with forbidden op and verify it raises
result = verify_tflite_model(mock_model, allowed_ops={...})
assert result.failed_operations is not None
```

---

### 34. Exit Code Tests Are Unconditionally Skipped
**File:** `tests/unit/test_verify_esphome_exit_codes.py`
**Bot:** copilot-pull-request-reviewer
**Status:** Unresolved

**Issue Description:**
Both tests call `pytest.skip()` unconditionally. The exit code contract for success (0) and failure (2/4) paths is never exercised. This provides zero test coverage for the actual exit behavior.

**Impact:**
- Exit contract never validated
- Regressions possible
- False confidence in test suite

**Evidence from PR:**
```
Her iki test de `pytest.skip()` çağırıyor; başarısı/başarısı davranışı test edilmiyor.
```

**Suggested Fix:**
Use pytest fixtures or model mocks:
```python
# Create minimal TFLite model fixture
@pytest.fixture
def simple_compatible_model():
    return create_minimal_tflite_model()

def test_exit_code_0_for_compatible_model(simple_compatible_model):
    # Now we can actually test
    ...
```

---

### 35. probability_cutoff Resolution is Non-Deterministic
**File:** `src/export/manifest.py` (lines 103-149)
**Bot:** copilot-pull-request-reviewer
**Status:** Unresolved

**Issue Description:**
The resolver scans for `*.metadata.json` files and returns the first one found (using `os.path.commonprefix`). If a directory has multiple checkpoints with metadata files, the selection is non-deterministic and depends on filesystem iteration order.

**Impact:**
- Wrong cutoff for multi-checkpoint directories
- Non-deterministic behavior
- Race condition in selection

**Evidence from PR:**
```
Bu resolver `tflite_path` çevresinde `*.metadata.json` arıyor ve karakter-bazlı `commonprefix` ile eşleştiriyor. Tek alakasız dosya varsa yanlış olan seçilebilir.
```

**Suggested Fix:**
Use exact stem matching or canonical metadata first (same as issue 5).

---

### 36. log_false_predictions Uses Wrong Logger API
**File:** `src/training/mining.py` (around line calling `log_false_predictions_to_json`)
**Bot:** copilot-pull-request-reviewer
**Status:** Unresolved

**Issue Description:**
The code calls `_log.info(...)` but `RichTrainingLogger` exposes `log_info()` and `log_warning()` methods. It does not implement the generic `_log.info()` interface, so an `AttributeError` is raised.

**Impact:**
- Mining crashes when logging false predictions
- Training workflow interrupted
- Incompatible with logger implementations

**Evidence from PR:**
```
Kod `_log.info()` çağırıyor ama `RichTrainingLogger` `log_info()`/`log_warning()` metodlarını sunuyor. `AttributeError` fırlatıyor.
```

**Suggested Fix:**
Use RichTrainingLogger's correct API:
```python
# Change:
_log.info(...)
# To:
self.training_logger.log_info(...)
# Or use logging module
import logging
logger = logging.getLogger(...)
logger.info(...)
```

---

### 37. Annealing Expert Settings Ignored
**File:** `src/tuning/autotuner.py` (lines 1375-1378)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
`AnnealingController()` is called with default parameters, ignoring the expert-level knobs that were set (`self.initial_annealing_temp`, `self.cooling_rate`, `self.reheat_after`, `self.reheat_factor`). The annealing behavior doesn't reflect the configured expert settings.

**Impact:**
- Auto-tuning doesn't use annealing settings
- Temperature schedule ignored
- Ineffective annealing strategy

**Evidence from PR:**
```
Üstte `self.initial_annealing_temp`, `self.cooling_rate`, `self.reheat_after`, `self.reheat_factor` dolduruluyor; ama `AnnealingController()` default'larla kuruluyor.
```

**Suggested Fix:**
Pass expert parameters to AnnealingController:
```python
self.annealing = AnnealingController(
    initial_temperature=self.initial_annealing_temp,
    cooling_rate=self.cooling_rate,
    reheat_factor=self.reheat_factor,
    reheat_after=self.reheat_after,
)
```

---

### 38. Base Model Uses Different Weighting
**File:** `src/tuning/autotuner.py` (lines 3169-3175)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
`base_metrics` is computed without `sample_weights`, but later evaluations pass `sample_weights=search_eval_weights`. This creates an "apples-to-oranges" comparison between the baseline and optimized candidates - metric scores are not comparable.

**Impact:**
- Before→After comparison is invalid
- Misleading optimization results
- Incorrect baseline selection

**Evidence from PR:**
```
`base_metrics` ağırlıksız hesaplanıyor; aşağıdaki iterasyon değerlendirmesi ise `sample_weights=search_eval_weights` kullanıyor.
```

**Suggested Fix:**
Pass sample_weights to base_metrics computation:
```python
base_metrics = self._evaluate_scores(
    y_scores=base_eval_scores,
    labels=search_eval_labels,
    ambient_hours=ambient_hours_search_eval,
    fold_indices=fold_indices,
    sample_weights=search_eval_weights,  # Add this
)
```

---

### 39. Test Guardrail Silently False Passes
**File:** `tests/integration/test_pipeline_regression.py` (line 147)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
If `src/tuning` directory is missing or contains no `.py` files, the test calls `_assert_token_absent_in_python_files()` on an empty directory. Since no files to scan, the test passes without actually checking for forbidden tokens (`_evaluate_int8`, `int8_shadow`, `eval_results_int8`).

**Impact:**
- INT8 shadow artifact regressions go undetected
- False confidence in guardrails
- Silent test failures

**Evidence from PR:**
```
`src/tuning` dizini yoksa (veya içinde `*.py` yoksa) bu test otomatik geçiyor; yani yasak token kontrolü fiilen çalışmamış oluyor.
```

**Suggested Fix:**
Assert directory and files exist before scanning:
```python
tuning_root = SRC_ROOT / "tuning"
assert tuning_root.is_dir(), "src/tuning directory not found"

# Find Python files first
python_files = list(tuning_root.glob("*.py"))
assert len(python_files) > 0, "No Python files in tuning directory"

# Now scan for tokens
_assert_token_absent_in_python_files(tuning_root, forbidden)
```

---

### 40. find_rich_tables.py Not a Valid Script
**File:** `scripts/find_rich_tables.py`
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
The script lacks:
1. No `main()` function
2. No `argparse` CLI interface
3. No error handling for file not found
4. Hardcoded file path

This violates the `scripts/` directory contract that scripts should be self-contained CLI tools.

**Impact:**
- Cannot be used as standalone tool
- Violates repository conventions
- Not portable

**Evidence from PR:**
```
Bu script CLI değil: `main()` yok, `argparse` yok, hata yakalama yok ve sabitlenmiş yol var. `scripts/` sözleşmesini ihlal ediyor.
```

**Suggested Fix:**
Add CLI interface and error handling:
```python
def main():
    parser = argparse.ArgumentParser(description="Find Rich tables in training logs")
    parser.add_argument("file", type=str, default="src/training/rich_logger.py")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    try:
        path = Path(args.file)
        if not path.exists():
            print(f"Error: File not found: {path}")
            return 1
        # ... search and print
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

---

## 🟡 MINOR ISSUES (4)

### 41. Documentation Table Formatting Issues
**File:** Various documentation files
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
Several markdown formatting issues:
- Missing blank lines around tables (violates MD058)
- Fenced code blocks missing language tags (violates MD040)

**Files affected:**
- `docs/compliance_audit_2026-03-15.md` (lines 85-90, 158-160)
- `BUG_REPORT.md` (line 312-314)

**Impact:**
- Markdown linter warnings
- Poor documentation rendering
- Inconsistent formatting

**Suggested Fix:**
```
# Add blank lines before and after tables
| Column 1 | Column 2 |
|---------|----------|
| Data    | Value    |

# Add language tags to fence blocks
```python
# Instead of:
```
input_value = (feature * 256) / 666 - 128
```

# Use:
```text
input_value = (feature * 256) / 666 - 128
```

or
```bash
input_value = (feature * 256) / 666 - 128
```
```

---

### 42. Typo in AGENTS.md
**File:** `src/tuning/AGENTS.md` (line 122)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
The list uses `temperatureKnob` (lowercase "t") instead of `TemperatureKnob` (uppercase "T"). All other knobs use correct PascalCase (LRKnob, ThresholdKnob, SamplingMixKnob, WeightPerturbationKnob, LabelSmoothingKnob).

**Impact:**
- Inconsistent naming
- Documentation inaccuracy
- Doesn't match actual class name

**Suggested Fix:**
Fix casing:
```markdown
temperatureKnob     # Wrong
TemperatureKnob     # Correct
```

---

### 43. Command Templates Have Unquoted Paths
**File:** `src/training/rich_logger.py` (lines 390-554)
**Bot:** coderabbitai
**Status:** Unresolved

**Issue Description:**
Command templates embed `best_path` and `config_preset` variables directly into f-strings without shell quoting. If a path contains spaces, copying and pasting the command will fail because the path gets split into multiple arguments.

**Example failure:**
```
# Path: checkpoints/best model.weights.h5
# Generated command (broken):
mww-autotune --checkpoint checkpoints/best model.weights.h5 --config fast test
# Shell interprets as: "checkpoints/best", "model.weights.h5", "fast", "test"
```

**Impact:**
- Copied commands fail with spaces in paths
- Poor user experience
- Frustration with copy-paste

**Suggested Fix:**
Use subprocess-safe quoting or shlex.quote:
```python
# In command templates, use quoting:
cmd = f"mww-autotune --checkpoint {shlex.quote(best_path)} --config {shlex.quote(config_preset)}"

# Or use helper function:
def _cmd(cmd: str) -> str:
    parts = shlex.split(cmd)
    return subprocess.list2cmdline(parts)
```

---

## ❌ RESOLVED / DEPRECATED ISSUES (8 - Excluded)

These issues were already marked as resolved in the PR or fixed in commits. Excluded from the todo list to avoid confusion.

1. ~~Causal padding applied wrong side (dynamic rank)~~ - Fixed in commit a90405b5
2. ~~Generator never terminates~~ - Marked resolved by qodo-code-review
3. ~~Test JSON output flag~~ - Marked resolved by copilot
4. ~~Hardcoded test path in test_tuning_integration.py~~ - Fixed in commit 95986b9
5. ~~Static path test~~ - Fixed in commit 95986b9
6. ~~Best weights resume semantics confusion~~ - Fixed in commit a90405b5
7. ~~Checkpoint selection documentation confusion~~ - Fixed in commit a90405b5
8. ~~Verification date inconsistency~~ - Fixed in commit a90405b5

---

## Verification Guidelines

Before starting fixes, consider verifying issues in this priority order:

### 🔴 Priority 1 - Critical Issues
Verify any of these before implementation:
1. **Issue 1: GPU Memory Cleanup** (`src/data/spec_augment_gpu.py`)
2. **Issue 2: CLI Flags Ignored** (`src/tuning/cli.py`)
3. **Issue 3: Help Panel Crash** (`src/tools/help_panel.py`)
4. **Issue 4: Verification Script Crash** (`scripts/verify_esphome.py`)

### 🟠 Priority 2 - High Impact Issues
After critical issues, verify:
5. **Issue 7: Causal Padding Wrong Side** (`src/model/streaming.py`)
6. **Issue 8: Generator Infinite Loop** (`src/data/dataset.py`)
7. **Issue 9: Test Wrong Weights** (`src/training/trainer.py`)
8. **Issue 19: _run_burst Stub** (`src/tuning/orchestrator.py`)
9. **Issue 20: _evaluate_candidate Ignores Model** (`src/tuning/orchestrator.py`)

### ⚙️ Priority 3 - Implementation Completeness
Focus on auto-tuning orchestration:
10. **Issue 18: Random Seed Identical** (`src/tuning/population.py`)
11. **Issue 37: Annealing Settings Ignored** (`src/tuning/autotuner.py`)

---

## Next Steps

1. Review this document and prioritize issues based on severity
2. For each issue, verify the current code matches the problem description
3. Create targeted fixes for highest-priority issues first
4. Test fixes thoroughly to avoid regressions
5. Update this document with fix status and commit references

---

**Generated from:** PR #14 Bot Comments
**Last Updated:** March 17, 2026
**Total Issues Listed:** 44 unresolved
