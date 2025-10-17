# TODO List

## Completed Tasks

- [x] **Test the changes**
  - Test the updated training pipeline with preprocessed data

- [x] **Implement KL scheduling & free-bits**
  - Add linear/cyclical β-annealing plus free-bits (e.g., 0.5–1.0 nat per dim) to prevent posterior collapse and stabilize small-data languages in β-VAE MSE+KL setup

- [x] **Implement language conditioning**
  - Add a tiny language ID embedding into the text path (and optionally the prior) for explicit language conditioning

- [x] **Data Loader Simplification**
  - Simplify data loader to expect standardized 'text', 'audio', and 'language' fields

- [x] **Teacher State Augmentation**
  - Add small Gaussian noise and random time jitter to encoder targets E during TLE training to regularize duration/prosody mapping

- [x] **Text-only Fine-tuning**
  - Create finetune_decoder.py script for Phase 2 Whisper decoder adaptation using TLE-generated pseudo encoder states

- [x] **Pipeline Validation**
  - Test complete TLE pipeline with all implemented features for multilingual training
  - Successfully implemented and validated text-only fine-tuning with proper teacher forcing and tensor dimensions

## Future Improvements

- [x] **Advanced Data Loading Enhancements**
  - Enhanced tle/utils.py and tle/data.py with better error handling, logging, safe audio processing functions, and additional data augmentation options (noise, pitch_shift, time_stretch)

- [x] **Performance Optimization**
  - Implement mixed precision training and gradient checkpointing for larger batch sizes