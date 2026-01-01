# StyleTTS2 Hyperparameter Explanation
This document explains the hyperparameters found in `Configs/config_ft.yml`.

## General Training Settings
- **`log_dir`**: Directory where model checkpoints and tensorboard logs will be saved.
- **`save_freq`**: How often (in epochs) to save a model checkpoint.
- **`log_interval`**: How often (in steps) to log training metrics to the console/tensorboard.
- **`device`**: computation device (`cuda` or `cpu`).
- **`epochs`**: Total number of training epochs.
- **`batch_size`**: Number of samples processed in one training step.
- **`max_len`**: Maximum length of audio frames to use during training (for memory management).
- **`pretrained_model`**: Path to the base model checkpoint to start fine-tuning from.
- **`second_stage_load_pretrained`**: If `true`, loads the pretrained weights for the second stage (decoder/diffusion) as well.
- **`load_only_params`**: If `true`, resets the optimizer state and epoch counter, effectively starting a "fresh" training from the pretrained weights.

## External Models (frozen)
These paths point to auxiliary models used for specific losses or feature extraction:
- **`F0_path`**: Path to the JDC network (Pitch extractor).
- **`ASR_config` / `ASR_path`**: Path to the Automatic Speech Recognition model (used for linguistic alignment/loss).
- **`PLBERT_dir`**: Directory for the PL-BERT model (used to encode phonemes with context).

## Data Parameters (`data_params`)
- **`train_data` / `val_data`**: specific paths to text files listing the training and validation samples (format: `filename|text|speaker`).
- **`root_path`**: The root directory where the actual `.wav` files are stored.
- **`OOD_data`**: "Out Of Distribution" texts used to test style generalization during training logging.
- **`min_length`**: Minimum text length for OOD samples.

## Preprocess Parameters (`preprocess_params`)
- **`sr`**: Sample rate of the audio (e.g., 24000 Hz).
- **`spect_params`**: Settings for creating Mel Spectrograms (FFT size, window length, hop length). These MUST match what the pretrained model expects.

## Model Architecture (`model_params`)
- **`multispeaker`**: Boolean. If true, adds speaker embedding layers.
- **`dim_in`, `hidden_dim`, `n_layer`, etc.**: Dimensions of the internal Transformer/Conformer layers.
- **`style_dim`**: Size of the style vector (captures emotion/prosody).
- **`decoder`**: Configuration for the audio waveform generator (HiFi-GAN).
    - `upsample_rates`/`kernel_sizes`: Define how the spectrogram is upsampled to raw audio.
- **`slm`**: "Speech Language Model" (WavLM) configuration. Used for the adversarial loss to improve naturalness.
- **`diffusion`**: Settings for the style diffusion module (predicts style from text).

## Loss Weights (`loss_params`)
These control the balance of different objectives during training:
- **`lambda_mel`**: Importance of reconstructing the Mel-spectrogram accurately.
- **`lambda_F0`, `lambda_dur`, `lambda_sty`**: Importance of matching Pitch, Duration, and Style respectively.
- **`lambda_diff`**: Loss weight for the diffusion / score matching model.
- **`lambda_slm`**: Weight for the WavLM-based adversarial loss (critical for high fidelity).
- **`diff_epoch`**: Epoch number to START training the diffusion model. (Usually delayed to let other parts stabilize).
- **`joint_epoch`**: Epoch to START acting as a joint optimization (SLM adversarial training starts here).

## Optimizer (`optimizer_params`)
- **`lr`**: Global learning rate.
- **`bert_lr`**: Learning rate specifically for the PL-BERT encoder (usually lower to prevent catastrophic forgetting).
- **`ft_lr`**: Learning rate for fine-tuning specific acoustic modules.

## SLM Adversarial Params (`slmadv_params`)
Controls the WavLM-based discriminator:
- **`batch_percentage`**: Reduces batch size for this heavy loss to save VRAM.
- **`iter`**: Run this optimization only every N steps (saves time/compute).
- **`thresh` / `scale`**: Gradient clipping and scaling to prevent instability when utilizing the large WavLM model.
