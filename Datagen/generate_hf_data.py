import os
from datasets import load_dataset

BASE_DIR = "/home/cloud/StyleTTS2-fine-tuning"
OUTPUT_DIR = os.path.join(BASE_DIR, "Data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "source_text_hf.txt")
HF_DATASET_ID = "agentlans/high-quality-english-sentences"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_and_save_text():
    print(f"Downloading first 10,000 lines from {HF_DATASET_ID}...")
    ds = load_dataset(HF_DATASET_ID, split="train[:8000]")

    text_column = "text"
    if "sentence" in ds.column_names:
        text_column = "sentence"
    elif "content" in ds.column_names:
        text_column = "content"
    
    print(f"Extracting to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in ds:
            line = row[text_column]
            if line and isinstance(line, str):
                clean_line = line.strip()
                if clean_line:
                    f.write(clean_line + "\n")

    print(f"Saved {len(ds)} lines to {OUTPUT_FILE}")

if __name__ == "__main__":
    download_and_save_text()