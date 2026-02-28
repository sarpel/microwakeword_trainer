# MWW Eğitim Log ve Profil Yorumlama Rehberi

Bu rehber, microwakeword_trainer ile eğitim yaparken oluşan log ve profil dosyalarını nasıl yorumlayacağınızı açıklar.

---

## 📊 1. PROFİL DOSYALARI (`.prof`)

**Konum:** `./profiles/` dizini

### Profil Nedir?

cProfile ile oluşturulmuş Python performans analiz dosyalarıdır. Hangi fonksiyonların ne kadar zaman aldığını gösterir.

### İnceleme Yöntemleri

```bash
# 1. Python ile okuma (terminalde görüntüleme)
python -c "
import pstats
p = pstats.Stats('profiles/data_loading_123456.prof')
p.sort_stats('cumulative')  # Toplam süreye göre sırala
p.print_stats(20)  # İlk 20 fonksiyonu göster
"

# 2. Kod içinde kullanma
from src.training.profiler import TrainingProfiler

# Mevcut bir profili analiz et
summary = TrainingProfiler.get_summary("./profiles/training_step_123456.prof", top_n=30)
print(summary)
```

### Profil Çıktısı Nasıl Okunur?

```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   5000    2.345    0.000   15.678    0.003 spectrogram.py:45(compute_mel)
    200    0.123    0.001   12.456    0.062 model.py:89(call)
```

| Sütun | Anlamı | Yorumu |
|-------|--------|--------|
| **ncalls** | Çağrı sayısı | Çok fazla çağrı = optimizasyon adayı |
| **tottime** | Fonksiyon içinde geçen süre | Saf hesaplama zamanı |
| **percall** | Çağrı başına süre | Tek çağrı maliyeti |
| **cumtime** | Toplam birikimli süre | Alt fonksiyonlar dahil |
| **cumtime/percall** | Çağrı başına toplam | En önemli metrik! |

### 🔴 Bottleneck (Tıkanıklık) Tespiti

| Durum | Anlamı | Çözüm |
|-------|--------|-------|
| **cumtime yüksek, tottime düşük** | Fonksiyon başka yavaş fonksiyonları çağırıyor | Alt fonksiyonları optimize et |
| **tottime yüksek** | Fonksiyonun kendisi yavaş | Fonksiyonu optimize et veya vektörize et |
| **ncalls çok yüksek** | Gereksiz döngü içinde çağrı | Vektörizasyon yap, döngüden çıkar |

---

## 📋 2. TERMINAL LOG DOSYALARI (`terminal_*.log`)

**Konum:** `./logs/terminal_YYYYMMDD_HHMMSS.log`

### Log Dosyalarını Listeleme

```bash
# Log dosyalarını listele
ls -la ./logs/terminal_*.log

# En son logu izle
tail -f ./logs/terminal_$(date +%Y%m%d)*.log
```

### Log Yapısı ve Yorumlama

#### Eğitim Başlangıcı
```
Training Log Started: 2025-02-27T10:27:17
================================================================================

[TerminalLogger] Capturing output to: ./logs/terminal_20250227_102717.log

🎯 Wake Word Training
┌─────────────────┬────────────────────────────────┐
│ Phase 1         │ 20,000 steps @ LR 0.001000     │
│ Phase 2         │ 10,000 steps @ LR 0.000100     │
│ Class Weights   │ pos=[1.0, 1.0]  neg=[20.0...   │
│ Batch Size      │ 128                            │
└─────────────────┴────────────────────────────────┘
```

#### Eğitim İlerlemesi
```
Phase 1 • 500/30000 • 1.7% • 0:02:14 • 2:10:45 • loss=0.2341 acc=0.8912 lr=0.001000
```

| Alan | Anlamı |
|------|--------|
| `Phase 1` | Mevcut eğitim fazı |
| `500/30000` | Mevcut step / Toplam step |
| `1.7%` | Tamamlanma yüzdesi |
| `0:02:14` | Geçen süre |
| `2:10:45` | Tahmini kalan süre (ETA) |
| `loss=0.2341` | Kayıp değeri |
| `acc=0.8912` | Doğruluk |
| `lr=0.001000` | Öğrenme oranı |

---

## 🎯 Önemli Metrikler ve Anlamları

### 1. Loss (Kayıp)

```
loss=0.2341
```

| Değer Aralığı | Durum | Yorum |
|---------------|-------|-------|
| **0.1 - 0.3** | 🟢 İyi | Öğrenme devam ediyor |
| **0.3 - 0.5** | 🟡 Normal | Normal seyir |
| **> 0.5** | 🔴 Kötü | Düşük öğrenme oranı veya veri sorunu |
| **< 0.01** | 🟠 Uyarı | Aşırı öğrenme (overfitting) riski |

### 2. Accuracy, Precision, Recall, F1

```
acc=0.8912  prec=0.8234  recall=0.7567  f1=0.7889
```

| Metrik | Hedef | Düşükse Ne Yapılmalı? |
|--------|-------|----------------------|
| **Accuracy** | > 0.95 | Daha fazla veri, augmentation artır |
| **Precision** | > 0.90 | False Positive çok → negatif örnekleri artır |
| **Recall** | > 0.90 | False Negative çok → pozitif örnekleri artır |
| **F1** | > 0.90 | Dengesiz sınıflar → class weight ayarla |

### 3. Ambient FA/Hour (False Activation/Hour)

```
Ambient FA/Hour: 3.45  [🟡 Sarı]
```

**Bu, wake word için EN KRİTİK metriktir!** Saatte kaç yanlış alarm verdiğini gösterir.

| Değer | Renk | Durum | Anlamı |
|-------|------|-------|--------|
| **< 0.5** | 🟢 Yeşil | Mükemmel | Kabul edilebilir yanlış alarm |
| **0.5 - 2.0** | 🟡 Sarı | Kabul edilebilir | Sınırda, iyileştirilebilir |
| **> 2.0** | 🔴 Kırmızı | Kötü | Çok fazla yanlış uyandırma |

### 4. Checkpoint Mesajları

```
✅ BEST MODEL FAH improved: 3.45 → 2.12
   → checkpoints/best_fah_step_500.ckpt

💾 Checkpoint: step_1000.ckpt
```

| İkon | Anlamı |
|------|--------|
| **✅ BEST MODEL** | En iyi performans kaydedildi (daha iyi FAH) |
| **💾 Checkpoint** | Düzenli ara kayıt (her N adımda) |

---

## 📊 Validation (Doğrulama) Sonuçları

```
📊 Validation Results — Step 500/30000
┌──────────────────────┬────────┐
│ Accuracy             │ 0.8912 │
│ Precision            │ 0.8234 │
│ Recall               │ 0.7567 │
│ F1 Score             │ 0.7889 │  <- Hedef: >0.90
│ AUC-ROC              │ 0.9234 │
│ AUC-PR               │ 0.8567 │
│ Ambient FA/Hour      │ 3.45   │  <- 🟡 Sarı (hedef: <0.5)
│ Recall @ No FAPH     │ 0.6789 │
│ Threshold for No FAPH│ 0.8234 │
└──────────────────────┴────────┘
```

### Confusion Matrix

```
Confusion Matrix (threshold=0.5)
┌─────────────────┬──────────────────┬──────────────────┐
│                 │ Predicted Pos    │ Predicted Neg    │
├─────────────────┼──────────────────┼──────────────────┤
│ Actual Positive │ [green]850[/]     │ [red]150[/]       │
│ Actual Negative │ [red]200[/]       │ [green]7650[/]    │
├─────────────────┼──────────────────┼──────────────────┤
│ Total           │                  │ [bold]8850[/]     │
└─────────────────┴──────────────────┴──────────────────┘
```

- **TP (True Positive):** 850 - Doğru pozitif tahmin
- **FP (False Positive):** 200 - Yanlış pozitif (sesli komut olmadan tetikleme)
- **TN (True Negative):** 7650 - Doğru negatif tahmin
- **FN (False Negative):** 150 - Kaçırılan wake word

---

## 📈 3. TENSORBOARD LOG'LARI

**Konum:** `./logs/` dizini (TensorBoard event dosyaları)

### TensorBoard Başlatma

```bash
source ~/venvs/mww-tf/bin/activate
tensorboard --logdir ./logs

# Tarayıcıda aç: http://localhost:6006
```

### TensorBoard Sekmeleri

#### SCALARS (Metrikler)

| Metrik | Açıklama | İyi Seyir |
|--------|----------|-----------|
| `epoch_loss` | Her epoch sonundaki kayıp | ↓ Düşmeli |
| `epoch_accuracy` | Doğruluk grafiği | ↑ Artmalı |
| `val_loss` | Validasyon kaybı | ↓ Düşmeli (train_loss'a yakın) |
| `val_accuracy` | Validasyon doğruluğu | ↑ Artmalı |
| `learning_rate` | Öğrenme oranı değişimi | Fazlara göre adım adım düşer |

**Ne Aranır:**
- ✅ **loss ↓ düşüyor** → Model öğreniyor
- ❌ **val_loss ↑ artıyor** → Overfitting başladı
- ⚠️ **Loss dalgalanıyor** → Learning rate çok yüksek

#### GRAPHS (Model Grafiği)

Modelin katman yapısını görsel olarak gösterir:
- Op'lar arası bağlantılar
- Tensor boyutları
- Hesaplama grafiği

#### HISTOGRAMS (Ağırlık Dağılımları)

```
Layer weights   → Ağırlıkların dağılımı
Layer biases    → Bias değerleri
Gradients       → Gradyan büyüklükleri
```

**Yorumlama:**
- Ağırlıklar çok küçük → Vanishing gradient
- Ağırlıklar çok büyük → Exploding gradient
- Tüm ağırlıklar aynı → Başlatma sorunu

---

## 🔍 4. SIK KARŞILAŞILAN SORUNLAR

### Sorun: Loss Stagnant (Sabit Kalıyor)

```
Loss: 0.45 → 0.44 → 0.43 → 0.44 → 0.43 (1000 step sonra hâlâ)
```

**Çözüm:**
1. Learning rate çok düşük → `0.0001` → `0.001` yap
2. Veri yetersiz → Daha fazla örnek ekle
3. Augmentation az → `augmentation.yaml` ayarlarını artır

### Sorun: Validation İyi ama FA/Hour Kötü

```
val_accuracy: 0.98  (çok iyi!)
FA/Hour: 15.3      (çok kötü!)
```

**Çözüm:**
- Background audio ekle (ambient gürültü)
- Hard negative örnekleri artır
- Model threshold'u yükselt

### Sorun: Training Çok Yavaş

```
Step 100/30000 ETA: 48 hours
```

**Kontrol Adımları:**
```bash
# Profil dosyası var mı?
ls ./profiles/

# En yavaş fonksiyonu bul
python -c "
import pstats
p = pstats.Stats('profiles/training_step_xxx.prof')
p.sort_stats('cumulative').print_stats(5)
"
```

**Muhtemel Nedenler:**
- Data loading yavaş → `num_workers` artır
- GPU kullanılmıyor → `nvidia-smi` kontrol et
- CuPy kurulu değil → `uv pip install cupy-cuda12x`

---

## 🛠️ 5. PRATİK KOMUTLAR

```bash
# Son 100 satırı izle
tail -n 100 ./logs/terminal_20250227_*.log

# Tüm logları birleştir
cat ./logs/terminal_*.log > all_logs.txt

# ERROR/WARNING içeren satırları bul
grep -i "error\|warning\|exception" ./logs/terminal_*.log

# En son checkpoint'i bul
ls -lt ./checkpoints/*.ckpt | head -5

# En iyi modelin FAH değerini göster
grep "BEST MODEL" ./logs/terminal_*.log | tail -5

# Eğitim süresini hesapla
grep "Training Log Started\|Training Log Ended" ./logs/terminal_*.log
```

---

## 📋 6. HIZLI REFERANS TABLOSU

| Ne Arıyorsun? | Nereye Bak? | İyi Değer |
|--------------|-------------|-----------|
| Genel performans | Terminal log | F1 > 0.90 |
| Yanlış alarm | FA/Hour | < 0.5 |
| Yavaş fonksiyon | .prof dosyası | cumtime az |
| Model öğreniyor mu? | TensorBoard loss ↓ | Düşüyor |
| Overfitting | val_loss vs train_loss | Fark < 0.1 |
| Eğitim süresi | Log başlangıç/bitiş | Ne kadar azsa o kadar iyi |

---

## 🎯 Eğitim Başarı Kriterleri

Bir wake word modelinin başarılı sayılması için:

1. ✅ **F1 Score > 0.90**
2. ✅ **FA/Hour < 0.5** (en önemlisi!)
3. ✅ **Recall > 0.90** (kaçırmaması lazım)
4. ✅ **Precision > 0.90** (yanlış tetiklememesi lazım)
5. ✅ **Validation loss stabil** (overfitting yok)

---

*Bu rehber microwakeword_trainer v2.0.0 için hazırlanmıştır.*
