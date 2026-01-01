import os
import random
import pdfplumber
from datasets import load_dataset

BASE_DIR = "/home/cloud/StyleTTS2-fine-tuning"
DATA_DIR = os.path.join(BASE_DIR, "Data")
INPUT_PDF = os.path.join(DATA_DIR, "English_CORE2000.pdf")
FINAL_OUTPUT_FILE = os.path.join(DATA_DIR, "source_text_final.txt")
HF_DATASET_ID = "agentlans/high-quality-english-sentences"
HF_DOWNLOAD_COUNT = 8000

os.makedirs(DATA_DIR, exist_ok=True)

def extract_pdf_sentences():
    pdf_sentences = []
    
    if not os.path.exists(INPUT_PDF):
        print(f"Warning: PDF file not found at {INPUT_PDF}. Skipping.")
        return []

    with pdfplumber.open(INPUT_PDF) as pdf:
        for page in pdf.pages:
            words = page.extract_words()
            header_x = None
            header_bottom = None
            
            for j, word in enumerate(words):
                if word['text'] == 'Sample' and j+1 < len(words):
                    next_word = words[j+1]
                    if next_word['text'] == 'Sentence':
                        header_x = word['x0']
                        header_bottom = word['bottom']
                        break
            
            if header_x is not None:
                crop_box = (header_x - 5, header_bottom + 5, page.width, page.height)
                try:
                    cropped_page = page.crop(crop_box)
                    text_block = cropped_page.extract_text()
                    if text_block:
                        lines = text_block.split('\n')
                        for line in lines:
                            clean_line = line.strip()
                            if len(clean_line) > 10:
                                pdf_sentences.append(clean_line)
                except ValueError:
                    pass
    
    print(f"Extracted {len(pdf_sentences)} raw lines from PDF.")
    return pdf_sentences

def clean_sentences(sentences):
    print("Cleaning")
    cleaned_list = []
    removed_count = 0
    
    for line in sentences:
        cleaned = line.replace('EnglishClass101.com', '')
        stripped = cleaned.strip()
        
        if not stripped:
            continue
            
        if len(stripped.split()) < 4:
            removed_count += 1
            continue
            
        cleaned_list.append(stripped)
        
    print(f"Removed {removed_count} short sentences. Kept {len(cleaned_list)}.")
    return cleaned_list

def get_hf_sentences(count):
    print(f"Downloading {count} lines from Hugging Face")
    hf_sentences = []
    
    try:
        ds = load_dataset(HF_DATASET_ID, split=f"train[:{count}]")
        text_column = "text"
        if "sentence" in ds.column_names:
            text_column = "sentence"
        elif "content" in ds.column_names:
            text_column = "content"

        for row in ds:
            line = row[text_column]
            if line and isinstance(line, str):
                clean_line = line.strip()
                if clean_line:
                    hf_sentences.append(clean_line)
                    
        print(f"Downloaded {len(hf_sentences)} lines from HF.")
    except Exception as e:
        print(f"Error downloading HF dataset: {e}")
        
    return hf_sentences

def merge_and_save():
    raw_pdf_lines = extract_pdf_sentences()
    clean_pdf_lines = clean_sentences(raw_pdf_lines)
    hf_lines = get_hf_sentences(HF_DOWNLOAD_COUNT)
    
    print("Merging and Shuffling ")
    combined_data = clean_pdf_lines + hf_lines
    random.shuffle(combined_data)
    
    print(f"Saving {len(combined_data)} total lines to {FINAL_OUTPUT_FILE}...")
    with open(FINAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(combined_data))
        
    print("Dataset generation complete.")

if __name__ == "__main__":
    merge_and_save()