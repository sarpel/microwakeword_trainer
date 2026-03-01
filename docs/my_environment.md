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
| **TOTAL** | 220,138 | 9,235.97 | 2.51 |
| **Positive** | 19,030 | 275.81 | 0.87 |
| **Negative** | 116,516 | 6,482.84 | 3.34 |
| **Hard Negative** | 34,187 | 586.19 | 1.03 |
| **Background** | 30,350 | 1,723.35 | 3.41 |
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

---


