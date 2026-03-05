# Integrated Project Profile: "Hey Katya" Wakeword Training

## 1. Environment & Operational Profile

This section details the physical and acoustic environment where the model will operate.

### Voice Characteristics

* **Primary Male Speaker:** Deep bass voice with a monotonous speaking style.
* **Primary Female Speaker:** Alto voice with a semi-monotonous to semi-fluctuating style.
* **Wakeword:** "Hey Katya" (Phonetic structure should be prioritized during training).

### Home Acoustic Background

The following sounds are expected to be present during real-world operation:

* **Media:** TV audio (English and Turkish YouTube content) and high-quality home cinema audio.
* **Domestic Sounds:** Cat meows, air conditioner humming.
* **Hardware Interaction:** Keyboard and mouse button clicking noises.
* **Other:** Ambient voices of both speakers; very little to no mobile phone ringtones.

---

## 2. Dataset Composition & Analysis

Detailed breakdown of the training data used for the microwakeword model.

### Deep Composition

* **Real Live Recordings:** 1% (recorded in multiple rooms, including bathrooms/kitchens, using high-quality microphones).
* **Synthetic Clones:** 50% (cloned versions of the primary users' voices).
* **General Synthetic:** 49% (from various TTS engines).
* **Gender Balance:** Approximately 50% male and 50% female distribution.
* **Hard Negatives:** 50+ specific words selected via LLM analysis to be phonetically similar to "Hey Katya".

### Quantitative Analysis

| Category | File Count | Total Duration (Min) | Avg Duration (Sec) |
| --- | --- | --- | --- |
| **TOTAL** | 153,654 | 3,006.44 | 1.17 |
| **Positive** | 17,326 | 244.32 | 0.85 |
| **Negative** | 42,813 | 904.09 | 1.27 |
| **Hard Negative** | 43,461 | 724.29 | 1.00 |
| **Background** | 29,999 | 965.96 | 1.93 |
| **RIRs** | 20,055 | 167.78 | 0.50 |

*(Note: All files are in `.wav` format at a 16000Hz sample rate with 0 corrupted files reported.)*

---

## 3. Training & Hardware Infrastructure

The technical environment and target deployment devices.

### Training Hardware (Workstation)

* **CPU:** Ryzen 9 7950X (16-Core / 32-Thread)
* **GPU:** RTX 3060 Ti (8 GB VRAM)
* **RAM:** 64 GB System RAM
* **Storage:** Samsung 990 Pro 1 TB NVMe
* **Network:** 1 Gbit/s Bandwidth

### Target Deployment Devices (MCU/SBC)

The trained model must be 100% compatible with microwakeword (OHF) standards for:

* 2 x M5Stack Atom Echo
* 1 x ESP32-S3-BOX3
* Multiple ESP32 + INMP441 Microphone setups
* 2 x Raspberry Pi Zero 2W (Wyoming Satellite)

### Audio Frontend Constants

| Constant | Value | Why It Cannot Change |
|---|---|---|
| `sample_rate_hz` | **16 000 Hz** | ESPHome ADC hardware clock |
| `mel_bins` | **40** | Feature tensor width; changes model input shape |
| `window_size_ms` | **30 ms** | 480 samples per FFT window; baked into pymicro-features |
| `window_step_ms` | **10 ms** | 160 samples per hop; determines temporal resolution |
| `upper_band_limit_hz` | **7 500 Hz** | Nyquist constraint for 16 kHz + margin |
| `lower_band_limit_hz` | **125 Hz** | DC rejection floor |
| `enable_pcan` | **True** | Per-Channel Amplitude Normalization; deactivating changes the feature distribution entirely |

### Audio Frontend Processing Parameters

| Parameter | Value | Category |
|---|---|---|
| `pcan_strength` | **0.95** | PCAN Gain Control |
| `pcan_offset` | **80.0** | PCAN Gain Control |
| `pcan_gain_bits` | **21** | PCAN Gain Control |
| `noise_even_smoothing` | **0.025** | Noise Reduction |
| `noise_odd_smoothing` | **0.06** | Noise Reduction |
| `noise_min_signal_remaining` | **0.05** | Noise Reduction |
| `log_scale_shift` | **6** | Log Scale |

---

## Model I/O Contract

### Input Tensor

| Property | Value |
|---|---|
| Shape | `[1, stride, 40]` = **`[1, 3, 40]`** |
| Dtype | **`int8`** |
| Quantization scale | `0.101961` |
| Quantization zero_point | `-128` |

### Output Tensor

| Property | Value |
|---|---|
| Shape | **`[1, 1]`** |
| Dtype | **`uint8`** ← **NOT int8. NOT float32. UINT8.** |
| Quantization scale | `0.003906` |
| Quantization zero_point | `0` |