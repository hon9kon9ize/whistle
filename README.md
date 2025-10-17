# 🗣️ Whisper-TLE: Text-to-Latent VAE for Text-Only Fine-Tuning of Whisper

> **Text-to-Latent Encoder (TLE)** — a multilingual VAE that learns to map text tokens to pseudo audio encoder states for OpenAI’s `whisper-large-v3`.  
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

- 🌍 **Multilingual support**  
  Optional language embeddings and temperature-balanced sampling for multilingual fine-tuning.

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

## 🌍 Multilingual Training

To build a multilingual TLE:

* Add a `lang_embed = nn.Embedding(num_langs, text_hidden)` and sum it into the token embeddings.
* Use **temperature-based sampling** to balance language proportions:
  [
  p_l \propto n_l^\alpha, \ \alpha < 1
  ]
* Include typologically diverse languages (different families & scripts).
* Optionally train with TTS-generated audio for low-resource languages.

---

## 📊 Recommended Training Schedule

| Stage | Data              | Objective     | Steps | Notes                                 |
| ----- | ----------------- | ------------- | ----- | ------------------------------------- |
| 1     | Paired audio–text | `MSE + β·KL`  | 100k  | Freeze Whisper; train TLE             |
| 2     | Text-only         | `Decoder NLL` | 50k   | Alternate text-only & audio steps     |
| 3     | Optional          | Joint TLE+TTS | 30k   | Combine both latent & TTS supervision |

---

## 🧪 Evaluation

After training, you can evaluate WER or CER on out-of-domain test sets by:

1. Using the frozen Whisper encoder (for audio evaluation), or
2. Using TLE-generated encoder states from text (for domain adaptation).

The paper reports that **TLE+TTS** outperforms either approach alone across multiple languages.

---

## 📁 Repository Structure

```
whistle/
├── tle/
│   ├── tle.py        # TLEVAE + ResidualConv1dFiLM + loss
├── teacher.py        # feature extraction, teacher state cache
├── train.py          # supervised training loop
└── README.md
```

**Planned additions:**
- `tle/utils.py` - data loading utilities and helper functions
- `tle/data.py` - dataset classes and multilingual sampling  
- `finetune_decoder.py` - text-only Whisper decoder fine-tuning
- `requirements.txt` - project dependencies

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