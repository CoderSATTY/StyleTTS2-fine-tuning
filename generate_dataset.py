import os
import torch
import torchaudio
from tqdm import tqdm
from chatterbox.tts_turbo import ChatterboxTurboTTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REFERENCE_AUDIO_PATH = "/home/cloud/STT-Livekit-RTC/test_audio_2.wav"
INPUT_TEXT_FILE = "source_text.txt"
OUTPUT_DIR = "Data"
TARGET_SAMPLE_RATE = 24000
TRAIN_LIST_PATH = os.path.join(OUTPUT_DIR, "train_list.txt")

WAVS_DIR = os.path.join(OUTPUT_DIR, "wavs")
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
        if 10 < len(cleaned) < 200:
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
                # Extract number from "file_0001.wav"
                try:
                    # filename is file_XXXX.wav. Split by _ and then .
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
        # Check if this index (1-based) is already done
        if (i + 1) in completed_indices:
            continue

        filename = f"file_{i+1:04d}.wav"
        filepath = os.path.join(WAVS_DIR, filename)
        
        try:
            wav_tensor = model.generate(sentence, audio_prompt_path=REFERENCE_AUDIO_PATH)
            
            if wav_tensor.dim() == 1:
                wav_tensor = wav_tensor.unsqueeze(0)
            
            if wav_tensor.shape[0] > 1:
                wav_tensor = torch.mean(wav_tensor, dim=0, keepdim=True)

            if resampler:
                wav_tensor = resampler(wav_tensor)
            
            num_samples = wav_tensor.shape[-1]
            duration = num_samples / TARGET_SAMPLE_RATE
            
            if duration < 1.0 or duration > 12.0:
                continue
                
            torchaudio.save(filepath, wav_tensor.cpu(), TARGET_SAMPLE_RATE)
            
            line_content = f"{filename}|{sentence}|0"
            with open(TRAIN_LIST_PATH, "a", encoding="utf-8") as f:
                f.write(line_content + "\n")
            
        except Exception:
            continue

if __name__ == "__main__":
    generate_dataset()