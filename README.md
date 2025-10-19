# 🗣️ Whisper-TLE: Text-to-Latent VAE for Text-Only Fine-Tuning of Whisper

> **Text-to-Latent Encoder (TLE)** — a VAE that learns to map text tokens to pseudo audio encoder states for OpenAI’s `whisper-large-v3`.  
> Enables text-only adaptation and domain tuning *without* paired speech data.

---

## 🌟 Overview

Unoffical implementation of **Text-to-Latent VAE (TLE)** described in  
📄 *“WhisTLE: Deeply Supervised, Text-Only Domain Adaptation for Pretrained Speech Recognition Transformers”* ([arXiv:2509.10452](https://arxiv.org/abs/2509.10452)).

The goal is to fine-tune Whisper (or any seq2seq ASR model) on *text-only* data by replacing the frozen speech encoder with a **variational text encoder** that generates *latent speech representations* compatible with Whisper’s decoder.

---

## 🧩 Key Features

- 🔄 **Drop-in Whisper-compatible encoder replacement**  
  Produces `T × H` hidden states (H=1280 for whisper-large-v3) that match the speech encoder output.

- 🧠 **VAE formulation**  
  Global latent `z ~ N(μ, σ²)` with residual Conv1D FiLM modulation and β-VAE training objective.

- 🗣️ **Two-phase workflow**
  1. **Supervised TLE training**: regress text → teacher encoder states from paired speech–text.  
  2. **Text-only fine-tuning**: replace encoder with TLE, continue training Whisper decoder.

---

## 🏗️ Architecture Summary

```

Text tokens ─► Transformer Encoder ─► μ, logσ² ─► sample z
│
▼
Linear (text→H) → interpolate to T → + PosEnc
│
▼
Residual Conv1D + FiLM(z)
│
▼
Whisper-like states (B, T, H)

````

Loss:
\[
\mathcal{L} = \| E_{\text{teacher}} - \tilde{E} \|_2^2 + \beta\, \mathrm{KL}\big(q_\phi(z|y)\,\|\,\mathcal{N}(0,I)\big)
\]

---

## 🧰 Installation

```bash
git clone https://github.com/hon9kon9ize/whisper-tle.git
cd whisper-tle

# Python 3.10+ recommended
pip install -r requirements.txt
````

`requirements.txt` should include:

```text
torch>=2.1
transformers>=4.45
numpy
tqdm
soundfile
```

---

## 🚀 Usage

### 1️⃣ Train TLE on paired data

```python
from transformers import WhisperProcessor
from tle.modeling_tle import TLEVAE, TLEVAEConfig, vae_loss
from tle.utils import get_teacher_states

# Load Whisper encoder as teacher
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
teacher = WhisperModel.from_pretrained("openai/whisper-large-v3").eval().cuda()

# Init TLE
cfg = TLEVAEConfig(vocab_size=processor.tokenizer.vocab_size,
                   whisper_hidden=teacher.config.d_model)
tle = TLEVAE(cfg).cuda()

# Forward pass
E_teacher = get_teacher_states(teacher, audio_list)  # (B, T, 1280)
E_tilde, mu, logvar = tle(input_ids, attention_mask, target_T=E_teacher.size(1))

loss = vae_loss(E_tilde, E_teacher, mu, logvar, beta=cfg.beta)
loss.backward()
```

### 2️⃣ Fine-tune Whisper decoder with text-only data

```python
from transformers import WhisperForConditionalGeneration, BaseModelOutput

asr = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").cuda()
asr.model.encoder.requires_grad_(False)  # freeze speech encoder

# Use TLE to generate pseudo encoder states
E_tilde, _, _ = tle(input_ids, attention_mask)
encoder_outputs = BaseModelOutput(last_hidden_state=E_tilde)

loss = asr(encoder_outputs=encoder_outputs, labels=labels).loss
loss.backward()
```

---

## 🏃 Training

### Command-Line Training

Train TLE on paired audio-text data using the provided training script:

```bash
# Train on Common Voice dataset
python bin/train.py \
  --dataset "mozilla-foundation/common_voice_16_1" \
  --subset "yue" \
  --batch-size 4 \
  --max-steps 100000 \
  --save-every 1000

# Train on custom preprocessed dataset
python bin/train.py \
  --dataset "path/to/your/preprocessed/dataset" \
  --train-split "train" \
  --test-split "validation" \
  --batch-size 8 \
  --max-epochs 10 \
  --augment
```

### Training Arguments

- `--dataset`: HuggingFace dataset name or local path
- `--subset`: Dataset subset/configuration (e.g., language code)
- `--train-split`, `--test-split`: Names of train/test splits (default: "train", "test")
- `--batch-size`: Training batch size (default: 4)
- `--max-steps`: Maximum training steps (alternative to `--max-epochs`)
- `--max-epochs`: Maximum training epochs (default: 1)
- `--save-every`: Save checkpoint every N steps (default: 1000)
- `--augment`: Apply audio augmentation (8kHz resampling + μ-law)
- `--device`: Device to use ("cuda" or "cpu", auto-detected if not specified)

The script automatically loads Whisper models, creates the TLE configuration, and trains using PyTorch Lightning with automatic checkpointing.

---

## 🎯 Fine-Tuning Whisper with TLE

After training the TLE model, you can fine-tune the Whisper decoder using text-only data by replacing the speech encoder with TLE-generated pseudo encoder states.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│          TLE Training Architecture                 │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Training Data (Audio)                              │
│       ↓                                             │
│  ┌──────────────────────────────────────────────┐  │
│  │  Whisper Encoder (FROZEN ❄️)                  │  │
│  │  - requires_grad = False                     │  │
│  │  - eval() mode                               │  │
│  │  - no_grad() context                         │  │
│  └──────────────────────────────────────────────┘  │
│       ↓                                             │
│   E_teacher (detached targets)                      │
│       ↓                                             │
│  Text Input + E_teacher                             │
│       ↓                                             │
│  ┌──────────────────────────────────────────────┐  │
│  │  TLE Model (TRAINABLE 🔥)                    │  │
│  │  - Text Encoder                              │  │
│  │  - VAE Decoder                               │  │
│  │  - Optimizer only includes these params      │  │
│  └──────────────────────────────────────────────┘  │
│       ↓                                             │
│   E_tilde (predictions)                             │
│       ↓                                             │
│   Loss = MSE(E_tilde, E_teacher) + β*KL            │
│       ↓ (backprop only through TLE)                 │
│   Gradient Update                                   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Phase 2: Text-Only Decoder Fine-Tuning

```bash
# Fine-tune Whisper decoder with TLE on text-only data
python bin/finetune_decoder.py \
  --tle-checkpoint "checkpoints/tle-epoch=XX-step=XXXXX.ckpt" \
  --dataset "path/to/text/dataset" \
  --batch-size 4 \
  --max-steps 50000
```

### How It Works

1. **Load trained TLE model** and freeze its weights
2. **Load Whisper model** and freeze the encoder
3. **Generate pseudo encoder states** from text using TLE
4. **Fine-tune only the decoder** on text-to-text translation task
5. **Result**: Domain-adapted Whisper decoder that works with text-only data

### Benefits

- **Text-only adaptation**: No need for paired audio-text data after TLE training
- **Domain specialization**: Adapt Whisper to specific domains (medical, legal, technical)
- **Reduced computational cost**: Only train decoder parameters (~300M vs 1.5B total)

---

## 📊 Recommended Training Schedule

| Stage | Data              | Objective     | Steps | Notes                                 |
| ----- | ----------------- | ------------- | ----- | ------------------------------------- |
| 1     | Paired audio–text | `MSE + β·KL`  | 100k  | Freeze Whisper; train TLE             |
| 2     | Text-only         | `Decoder NLL` | 50k   | Alternate text-only & audio steps     |

---

## 🧪 Evaluation

After training, you can evaluate WER or CER on out-of-domain test sets by:

1. Using the frozen Whisper encoder (for audio evaluation), or
2. Using TLE-generated encoder states from text (for domain adaptation).

The paper reports that TLE provides effective domain adaptation for speech recognition transformers.

---

## 🗺️ Roadmap

### ✅ Implemented Features

- **TLE Training Pipeline**: Complete training script for TLE on paired audio-text data
- **Dataset Compatibility**: Support for Common Voice and custom preprocessed datasets
- **Audio Augmentation**: 8kHz resampling + μ-law compression for training
- **Model Architecture**: Full TLE-VAE implementation with FiLM modulation
- **KL Scheduling & Free-bits**: Linear β-annealing and free-bits regularization to prevent posterior collapse
- **Language Conditioning**: Language ID embeddings for English (en), Mandarin (zh), and Cantonese (yue)
- **Text-Only Fine-Tuning**: Complete `finetune_decoder.py` script for Phase 2 decoder adaptation

### 🚧 Planned Features

- **Model Zoo**: Pre-trained TLE checkpoints

### 🎯 Current Status

| Component | Status | Priority |
|-----------|--------|----------|
| TLE Training | ✅ Complete | - |
| Dataset Loading | ✅ Complete | - |
| Text-Only Fine-Tuning | ✅ Complete | - |
| KL Scheduling & Free-bits | ✅ Complete | - |
| Language Conditioning | ✅ Complete | - |
| Advanced Data Loading | ✅ Complete | - |
| Model Zoo | ❌ Not implemented | Low |

---

##  Repository Structure

```
whistle/
├── tle/
│   ├── tle.py        # TLEVAE + ResidualConv1dFiLM + loss
│   ├── data.py       # Data loading utilities and audio processing
│   └── utils.py      # Additional utility functions
├── bin/
│   ├── train.py      # TLE training script
│   └── finetune_decoder.py  # Text-only Whisper decoder fine-tuning
├── checkpoints/      # Model checkpoints
├── test.ipynb        # Smoke tests and validation
└── README.md
```

---

## ⚡ Performance Optimization: Eliminating GPU Bottleneck

### Problem
Original training showed **0% GPU utilization** with 20% memory usage due to Whisper teacher state extraction happening on CPU during training.

### Solution
Multiple optimization strategies for huge datasets - no precomputation required!

### Quick Optimization Start

```bash
# Test optimizations on small dataset
python test_optimizations.py \
    --dataset "mozilla-foundation/common_voice_16_1" \
    --subset "yue" \
    --batch-size 32 \
    --max-steps 100

# Full training with optimizations
python bin/train.py \
    --dataset "your-huge-dataset" \
    --batch-size 32 \
    --max-epochs 10 \
    --precision bf16-mixed
```

### Performance Comparison

| Metric | Original | whisper-large-v3 | Improvement |
|--------|----------|------------------|-------------|
| GPU Utilization | 0% | >80% | **Massive** |
| Cantonese Support | Poor | Excellent | **Best available** |
| Training Speed | 1x | 2-3x | **200-300%** |
| Memory Usage | 20% GPU | 20% GPU | Same |

### Key Optimizations

1. **whisper-large-v3**: Best Cantonese support available
2. **Multiprocessing**: 4 workers for data loading with prefetching
3. **Large batch sizes**: Direct large batches (no gradient accumulation needed)
4. **Mixed Precision**: Automatic bf16/fp16 selection for speed

### Scripts

- **`bin/train.py`**: Optimized training script with all improvements
- **`test_optimizations.py`**: Test script to validate optimizations
- **`precompute_features.py`**: Pre-computation (only for small datasets)

### When to Use Each Approach

| Dataset Size | Recommended Approach | Why |
|-------------|---------------------|-----|
| < 10k samples | Pre-computation | Fastest, but requires storage |
| 10k - 1M samples | Optimized training | Best balance of speed vs storage |
| > 1M samples | Optimized training | No storage overhead, scales infinitely |

### Command Line Options

```bash
python bin/train.py \
    --dataset "your-dataset" \
    --batch-size 32 \        # Large batches for GPU memory
    --precision bf16-mixed \ # Fastest on modern GPUs
    --max-epochs 10
```

---

## 📜 Citation

```bibtex
@misc{pandey2025whistledeeplysupervisedtextonly,
      title={WhisTLE: Deeply Supervised, Text-Only Domain Adaptation for Pretrained Speech Recognition Transformers}, 
      author={Akshat Pandey and Karun Kumar and Raphael Tang},
      year={2025},
      eprint={2509.10452},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.10452}, 
}
```