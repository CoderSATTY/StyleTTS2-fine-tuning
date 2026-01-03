import os
import torch
import torchaudio
from tqdm import tqdm
from chatterbox.tts_turbo import ChatterboxTurboTTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = "/home/cloud/StyleTTS2-fine-tuning"
OUTPUT_DIR = os.path.join(BASE_DIR, "Data")

REFERENCE_AUDIO_PATH = os.path.join(OUTPUT_DIR, "reference_wavs/british_accent_audio.wav")
INPUT_TEXT_FILE = os.path.join(OUTPUT_DIR, "source_text_final.txt")
TRAIN_LIST_PATH = os.path.join(OUTPUT_DIR, "train_list_new.txt")
WAVS_DIR = os.path.join(OUTPUT_DIR, "wavs")
TARGET_SAMPLE_RATE = 24000

os.makedirs(WAVS_DIR, exist_ok=True)

model = ChatterboxTurboTTS.from_pretrained(device=DEVICE)

def get_sentences(text_path):
    if not os.path.exists(text_path):
        return []
    
    with open(text_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    valid_sentences = []
    for line in lines:
        cleaned = line.strip() 
        if cleaned:  
            valid_sentences.append(cleaned)
    
    return valid_sentences

def get_completed_indices():
    if not os.path.exists(TRAIN_LIST_PATH):
        return set()
    
    completed = set()
    with open(TRAIN_LIST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if parts and len(parts) >= 1:
                filename = parts[0]
                try:
                    number_part = filename.split("_")[1].split(".")[0]
                    completed.add(int(number_part))
                except:
                    continue
    return completed

def generate_dataset():
    sentences = get_sentences(INPUT_TEXT_FILE)
    completed_indices = get_completed_indices()
    
    print(f"Total sentences: {len(sentences)}")
    print(f"Already done:    {len(completed_indices)}")
    
    resampler = None
    if model.sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=model.sr, new_freq=TARGET_SAMPLE_RATE).to(DEVICE)

    for i, sentence in enumerate(tqdm(sentences)):
        if (i + 1) in completed_indices:
            continue

        filename = f"file_{i+1:04d}.wav"
        filepath = os.path.join(WAVS_DIR, filename)
        print(f"Generating {filename}...")
        try:
            with torch.inference_mode():
                wav_tensor = model.generate(
                    sentence,
                    audio_prompt_path=REFERENCE_AUDIO_PATH
                )

            if wav_tensor.dim() == 1:
                wav_tensor = wav_tensor.unsqueeze(0)

            if wav_tensor.shape[0] > 1:
                wav_tensor = wav_tensor.mean(dim=0, keepdim=True)

            wav_tensor = wav_tensor.cpu()

            if resampler:
                wav_tensor = resampler(wav_tensor)

            torchaudio.save(filepath, wav_tensor, TARGET_SAMPLE_RATE)

            with open(TRAIN_LIST_PATH, "a", encoding="utf-8") as f:
                f.write(f"{filename}|{sentence}|0\n")

            del wav_tensor
            torch.cuda.empty_cache()

            if (i + 1) % 50 == 0:
                torch.cuda.synchronize()

        except Exception as e:
            print(f"Error at sample {i+1}: {e}")
            continue


if __name__ == "__main__":
    generate_dataset()
