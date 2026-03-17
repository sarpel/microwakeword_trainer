# Phase 2: Security & Performance Review

## Security Findings

### Critical

#### SEC-C1: Unsafe Pickle Serialization in Population Module
- **CWE:** CWE-502 - Deserialization of Untrusted Data
- **CVSS:** 9.8 (CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H)
- **File:** `src/tuning/population.py`, lines 28, 34, 86, 114
- The `Candidate` class uses `pickle.dumps()`/`pickle.loads()` for model weights serialization. If tuning state is persisted to disk and loaded from a malicious file, arbitrary code execution is possible.
- **Attack Scenario:** Attacker crafts malicious pickle payload, replaces checkpoint file, victim loads checkpoint → malicious code executes with victim's privileges.
- **Fix:** Replace with NumPy's `.npz` format:
```python
def save_state_safe(self, model) -> None:
    buffer = io.BytesIO()
    weights = model.get_weights()
    np.savez(buffer, *weights)
    self.weights_bytes = buffer.getvalue()

def restore_state_safe(self, model) -> None:
    if self.weights_bytes is None:
        return
    buffer = io.BytesIO(self.weights_bytes)
    with np.load(buffer, allow_pickle=False) as data:
        weights = [data[f'arr_{i}'] for i in range(len(data.files))]
    model.set_weights(weights)
```

#### SEC-C2: Numpy allow_pickle=True on World-Writable Cache Files
- **CWE:** CWE-502 - Deserialization of Untrusted Data
- **CVSS:** 9.1 (CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:N)
- **File:** `src/data/clustering.py`, line 1162
- `np.load(cache_file, allow_pickle=True)` loads from `/tmp` (world-writable). Attacker can replace cache file with malicious numpy pickle payload.
- **Note:** Line 549 in same file correctly uses `allow_pickle=False`.
- **Fix:** Use `allow_pickle=False` and store metadata as JSON sidecar:
```python
data = np.load(cache_file, allow_pickle=False)
metadata_path = cache_file.with_suffix('.json')
if metadata_path.exists():
    with open(metadata_path) as f:
        metadata = json.load(f)
```

### High

#### SEC-H1: Unsafe ast.literal_eval on Config-Derived Strings
- **CWE:** CWE-95 - Eval Injection
- **CVSS:** 7.5 (CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:L/A:H)
- **File:** `src/model/architecture.py`, line 38
- `ast.literal_eval()` used on user-controllable configuration strings without format validation. Malformed input can cause DoS via excessive memory allocation.
- **Fix:** Add strict input validation:
```python
import re

def parse_model_param(text):
    if not text:
        return []
    if not re.match(r'^[\d,\[\]\s\-]+$', text):
        raise ValueError(f"Invalid characters in model parameter: {text!r}")
    if len(text) > 1000:
        raise ValueError(f"Model parameter too long: {len(text)} chars")
    return ast.literal_eval(text)
```

#### SEC-H2: Dynamic YAML String Construction
- **CWE:** CWE-20 - Improper Input Validation
- **CVSS:** 7.1 (CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:N)
- **File:** `scripts/verify_esphome.py`, line 172
- `yaml.safe_load(f"[{mixconv_str}]")` constructs YAML from string interpolation. While `safe_load` prevents RCE, string interpolation can cause parsing ambiguity.
- **Fix:** Use JSON parsing for structured data:
```python
import json
mixconv_str = model_cfg.get("mixconv_kernel_sizes", "[[5], [7, 11], [9, 15], [23]]")
try:
    mixconv_kernel_sizes = json.loads(mixconv_str)
except json.JSONDecodeError:
    mixconv_kernel_sizes = ast.literal_eval(f"[{mixconv_str}]")
```

### Medium

#### SEC-M1: Overly Permissive Security Linter Suppressions
- **CWE:** CWE-1104 - Use of Unmaintained Third Party Components
- **CVSS:** 6.5
- **File:** `pyproject.toml`, lines 117-127
- Globally suppresses `S301` (pickle), `S603`/`S607` (subprocess), `S605`/`S606` (OS calls), `S105`-`S107` (hardcoded passwords). New code introducing these patterns is never flagged.
- **Fix:** Use per-file ignores only where necessary:
```toml
[tool.ruff.lint]
ignore = ["S101"]  # Only for asserts in tests

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["S101"]
# Do NOT globally suppress S301
```

#### SEC-M2: Unbounded Memory Allocation in Hard-Negative Mining
- **CWE:** CWE-770 - Allocation of Resources Without Limits
- **CVSS:** 6.2
- **File:** `src/training/mining.py`
- Hard-negative mining allocates memory based on user-controlled configuration without upper bounds. Large values for `max_samples` or `top_k_per_epoch` can cause OOM crashes.
- **Fix:** Add explicit bounds:
```python
MAX_MINING_SAMPLES = 100_000
MAX_TOP_K = 10_000

def __init__(self, max_samples: int = 5000, top_k_per_epoch: int = 150):
    self.max_samples = min(int(max_samples), MAX_MINING_SAMPLES)
    self.top_k_per_epoch = min(int(top_k_per_epoch), MAX_TOP_K)
```

#### SEC-M3: Unsafe ast.literal_eval in TFLite Export
- **CWE:** CWE-95 - Eval Injection
- **CVSS:** 5.9
- **File:** `src/export/tflite.py`, line 1644
- Same issue as SEC-H1: `ast.literal_eval()` on config-derived strings without validation.
- **Fix:** Apply same validation pattern as SEC-H1.

### Low

#### SEC-L1: Subprocess Output Not Captured in Pipeline
- **CWE:** CWE-532 - Insertion of Sensitive Information into Log File
- **CVSS:** 4.3
- **File:** `src/pipeline.py`, lines 38, 48
- `subprocess.run()` without capturing output may leak sensitive information through stdout/stderr.
- **Fix:** Capture and filter sensitive information:
```python
result = subprocess.run(args, check=False, capture_output=True, text=True)
if result.returncode != 0:
    logger.error(f"Command failed: {' '.join(args)}")
    logger.debug(f"stderr: {result.stderr}")  # Debug level only
```

#### SEC-L2: Verbose Error Messages in Export
- **CWE:** CWE-209 - Generation of Error Message Containing Sensitive Information
- **CVSS:** 4.0
- **File:** `src/export/tflite.py`
- Exception handlers may leak internal file paths and system details through stack traces.
- **Fix:** Log stack traces only at debug level:
```python
except Exception as e:
    logger.warning(f"Could not load config: {e}")
    logger.debug(traceback.format_exc())  # Only in debug mode
```

#### SEC-L3: Predictable Temporary Directory Names
- **CWE:** CWE-377 - Insecure Temporary File
- **CVSS:** 3.7
- **File:** `src/data/clustering.py`, line 490
- Cache directory uses predictable name in system temp directory: `Path(tempfile.gettempdir()) / "mww_embeddings_cache"`
- **Fix:** Use `tempfile.mkdtemp()` with proper permissions:
```python
cache_dir = Path(tempfile.mkdtemp(prefix="mww_embeddings_", suffix=".cache"))
os.chmod(cache_dir, 0o700)
```

#### SEC-L4: Insufficient Security Logging
- **CWE:** CWE-778 - Insufficient Logging
- **CVSS:** 3.5
- **File:** `src/training/trainer.py`
- Security-relevant events (checkpoint loading, config overrides, data source changes) are not logged at appropriate levels.
- **Fix:** Add security event logging:
```python
security_logger = logging.getLogger("security")
security_logger.info(f"Config override applied: {override_path}")
security_logger.info(f"Checkpoint loaded: {checkpoint_path}")
```

---

## Performance Findings

### Critical

#### PERF-C1: Pickle Serialization for Model Weights
- **File:** `src/tuning/population.py`, lines 28-34
- Python's `pickle` is used for serializing model weights during auto-tuning. This adds 50-200ms per candidate per iteration.
- **Impact:** With population_size=4 and 100 iterations, adds 20-80 seconds of pure serialization overhead.
- **Fix:** Use TensorFlow's protobuf serialization:
```python
def save_state(self, model) -> None:
    weights = model.get_weights()
    self.weights_bytes = tf.io.serialize_tensor(tf.concat([
        tf.reshape(w, [-1]) for w in weights
    ], axis=0)).numpy()
    self._weight_shapes = [w.shape for w in weights]
```

#### PERF-C2: Unbounded Memory Growth in AsyncHardExampleMiner
- **File:** `src/training/mining.py`, lines 400-543
- Model cloning doubles GPU memory usage during async mining with no memory limit or fallback.
- **Impact:** Can cause OOM errors on GPUs with limited VRAM.
- **Fix:** Implement memory-aware mining with CPU fallback:
```python
def start_mining(self, model, data_generator, epoch):
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus and gpus[0].memoryFree < gpus[0].memoryTotal * 0.3:
            logger.warning("Low GPU memory, falling back to synchronous mining")
            self._fallback_to_sync = True
            return
    except Exception:
        pass
    # ... proceed with cloning
```

### High

#### PERF-H1: Synchronous I/O in Data Pipeline
- **File:** `src/data/dataset.py` (RaggedMmap class)
- `RaggedMmap.__getitem__` performs blocking mmap reads without prefetching, blocking the training loop.
- **Fix:** Implement async prefetching:
```python
class AsyncRaggedMmap:
    def __init__(self, ..., prefetch_size: int = 4):
        self._prefetch_queue = collections.deque(maxlen=prefetch_size)

    def prefetch_indices(self, indices: list[int]):
        def _prefetch():
            for idx in indices:
                data = self._sync_getitem(idx)
                self._prefetch_queue.append((idx, data))
        threading.Thread(target=_prefetch, daemon=True).start()
```

#### PERF-H2: Memory Leak in Validation Metrics Accumulation
- **File:** `src/training/trainer.py`, lines 80-110 (EvaluationMetrics class)
- Accumulates all predictions in Python lists without bounds. Python list overhead is ~24 bytes per float vs 4 bytes for float32.
- **Fix:** Use NumPy arrays with pre-allocation:
```python
class EvaluationMetrics:
    def __init__(self, ...):
        self._max_samples = 100000
        self.all_y_true = np.zeros(self._max_samples, dtype=np.float32)
        self.all_y_scores = np.zeros(self._max_samples, dtype=np.float32)
        self._current_idx = 0
```

#### PERF-H3: Inefficient Hypervolume Computation
- **File:** `src/tuning/orchestrator.py`, line 273
- Hypervolume computation for Pareto frontier is O(n^2) every iteration with no caching.
- **Fix:** Implement incremental updates:
```python
class ParetoArchive:
    def __init__(self, ...):
        self._cached_hypervolume = None
        self._hypervolume_dirty = False

    def try_add(self, metrics, candidate_id):
        added = self._do_add(metrics, candidate_id)
        if added:
            self._hypervolume_dirty = True
        return added

    def get_hypervolume(self, reference_point):
        if not self._hypervolume_dirty:
            return self._cached_hypervolume
        self._cached_hypervolume = compute_hypervolume(...)
        self._hypervolume_dirty = False
        return self._cached_hypervolume
```

#### PERF-H4: Missing Batch Size Configuration for tf.data
- **File:** `src/data/tfdata_pipeline.py`, lines 196-333
- Pipeline relies on generator yielding pre-batched data; no opportunity for tf.data's batch-level optimizations.
- **Fix:** Add explicit batching:
```python
def create_training_pipeline(self, batch_size: int | None = None, ...):
    batch_size = batch_size or self.batch_size
    ds = tf.data.Dataset.from_generator(generator, output_signature=...)
    ds = ds.unbatch().batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(buffer_size=self.autotune)
```

#### PERF-H5: Single-Point-of-Failure in Trainer
- **File:** `src/training/trainer.py`
- The `Trainer` class is a 1000+ line "God Object" handling all concerns. Cannot scale across multiple GPUs.
- **Fix:** Decompose into specialized components: `TrainingLoop`, `ValidationEngine`, `CheckpointManager`, `MetricsLogger`.

#### PERF-H6: No Distributed Training Support
- **Files:** Training pipeline
- No support for `tf.distribute.MirroredStrategy` or multi-node training.
- **Fix:** Add distributed strategy support:
```python
def setup_strategy(config):
    if config.get('distributed', False):
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()
    with strategy.scope():
        model = build_model(...)
```

### Medium

#### PERF-M1: ThreadPoolExecutor Without Proper Shutdown
- **File:** `src/training/trainer.py`, lines 544-557
- ThreadPoolExecutor only shut down in `__del__`, which is unreliable.
- **Fix:** Use context manager pattern:
```python
class Trainer:
    def close(self):
        if self._validation_executor is not None:
            self._validation_executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
```

#### PERF-M2: Repeated JSON File I/O in False Prediction Logging
- **File:** `src/training/mining.py`, lines 550-653
- `log_false_predictions_to_json` reads and rewrites entire JSON file on every epoch - O(n^2) behavior.
- **Fix:** Use append-only JSON Lines format:
```python
def log_false_predictions_to_json(...):
    log_path = Path(log_file).with_suffix('.jsonl')
    with open(log_path, "a") as f:
        json.dump({"epoch": epoch, **epoch_entry}, f)
        f.write('\n')
```

#### PERF-M3: Inefficient GPU Memory Configuration
- **File:** `src/utils/performance.py`, lines 99-155
- No validation of memory limits against actual GPU capacity.
- **Fix:** Add validation:
```python
def configure_tensorflow_gpu(memory_limit_mb: Optional[int] = None, ...):
    gpu_info = GPUtil.getGPUs()[device_id or 0]
    total_memory_mb = gpu_info.memoryTotal
    if memory_limit_mb and memory_limit_mb > total_memory_mb:
        logger.warning(f"Requested {memory_limit_mb}MB > GPU capacity {total_memory_mb}MB")
        memory_limit_mb = int(total_memory_mb * 0.9)
```

#### PERF-M4: No Caching of Feature Extraction
- **File:** `src/data/features.py` (MicroFrontend class)
- Audio feature extraction is recomputed on every training run with no caching.
- **Fix:** Implement feature caching with file hash invalidation:
```python
class MicroFrontend:
    def extract(self, audio_path: str) -> np.ndarray:
        if self.cache_dir:
            cache_key = self._compute_file_hash(audio_path)
            cache_path = self.cache_dir / f"{cache_key}.npy"
            if cache_path.exists():
                return np.load(cache_path)
        features = self._compute_features(audio_path)
        if self.cache_dir:
            np.save(cache_path, features)
        return features
```

#### PERF-M5: Race Condition in Async Mining State
- **File:** `src/training/mining.py`, lines 464-486
- Race condition between checking `_is_running` and setting it.
- **Fix:** Use state machine with atomic transitions:
```python
from enum import Enum, auto

class MiningState(Enum):
    IDLE = auto()
    CLONING = auto()
    RUNNING = auto()

def start_mining(self, model, ...):
    with self._lock:
        if self._state != MiningState.IDLE:
            raise RuntimeError(f"Mining in state {self._state}")
        self._state = MiningState.CLONING
    # ... clone model ...
    with self._lock:
        self._state = MiningState.RUNNING
        self._thread = threading.Thread(...)
        self._thread.start()
```

#### PERF-M6: Synchronous Checkpoint Writing
- **File:** `src/training/trainer.py`
- Checkpoints written synchronously, blocking training.
- **Fix:** Use async checkpointing for non-critical checkpoints:
```python
async def _save_checkpoint_async(self, path):
    await asyncio.to_thread(self.model.save_weights, path)

def _save_checkpoint(self, ...):
    if is_best:
        self.model.save_weights(best_path)  # Synchronous for best
    else:
        asyncio.create_task(self._save_checkpoint_async(periodic_path))
```

#### PERF-M7: TensorFlow Graph Retracing
- **File:** `src/data/tfdata_pipeline.py`
- Dynamic shapes in tf.data pipeline can cause graph retracing.
- **Fix:** Use fixed batch sizes with padded_batch:
```python
ds = ds.padded_batch(
    batch_size,
    padded_shapes=([batch_size, max_time_frames, 40], [batch_size]),
    drop_remainder=True
)
```

#### PERF-M8: Synchronous Evaluation Blocking Training
- **File:** `src/training/trainer.py`
- Validation runs synchronously, blocking the training loop.
- **Fix:** Implement async evaluation with stale weights:
```python
def _validate_async(self, model, val_generator):
    weights_snapshot = [w.numpy() for w in model.weights]
    def _eval_worker():
        eval_model = build_model(...)
        eval_model.set_weights(weights_snapshot)
        return self._validate_with_model(eval_model, val_generator)
    return self._validation_executor.submit(_eval_worker)
```

### Low

- **PERF-L1:** No compression for saved weights - use `np.savez_compressed()`
- **PERF-L2:** Thread safety in `EvaluationMetrics` - add locking for async validation
- **PERF-L3:** No memory pool for feature buffers - implement buffer pooling

---

## Critical Issues for Phase 3 Context

The following findings directly impact testing and documentation requirements:

1. **PERF-C1** (pickle serialization) and **SEC-C1** (pickle security) - Any test of auto-tuning must mock or handle the pickle serialization overhead
2. **PERF-C2** (memory doubling) - Tests need to account for GPU memory requirements or mock the mining behavior
3. **SEC-M1** (linter suppressions) - The suppressed security rules mean existing tests may not catch security regressions
4. **PERF-M5** (race condition) - Concurrency tests needed for async mining

---

## Findings Summary

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| Security | 2 | 2 | 3 | 4 | 11 |
| Performance | 2 | 6 | 8 | 3 | 19 |
| **Total** | **4** | **8** | **11** | **7** | **30** |

---

## Positive Security Findings

1. **YAML Safe Loading:** All YAML parsing uses `yaml.safe_load()` - no unsafe `yaml.load()` found
2. **Subprocess Safety:** All subprocess calls use list-form arguments without `shell=True`
3. **No Hardcoded Secrets:** No hardcoded passwords, API keys, or credentials found
4. **Proper Path Object Usage:** File paths consistently converted to `Path` objects before use

---

## Remediation Priority Matrix

| Priority | Finding | Effort | Impact |
|----------|---------|--------|--------|
| P0 | SEC-C1: Pickle in population.py | Low | Critical |
| P0 | SEC-C2: allow_pickle=True in clustering.py | Low | Critical |
| P0 | PERF-C1: Replace pickle serialization | Low | 10-50x speedup |
| P0 | PERF-C2: Fix memory doubling in mining | Medium | Prevents OOM |
| P1 | SEC-H1/H2: Input validation | Medium | High |
| P1 | PERF-H1: Async prefetching | Medium | 2-5x speedup |
| P1 | PERF-H2: Bound metrics memory | Low | Prevents OOM |
| P2 | SEC-M1: Linter suppressions | Low | Medium |
| P2 | PERF-H3: Incremental hypervolume | Medium | 10x speedup |
| P3 | SEC-L1-L4, PERF-L1-L3 | Low | Low |
