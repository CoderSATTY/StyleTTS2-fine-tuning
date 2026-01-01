import pdfplumber
import os

INPUT_PDF = r"Data\English_CORE2000.pdf"
OUTPUT_TXT = r"Data\source_text.txt"

def extract_sentences_visually():
    all_sentences = []
    print(f"Opening {INPUT_PDF}...")
    
    with pdfplumber.open(INPUT_PDF) as pdf:
        for i, page in enumerate(pdf.pages):
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
                crop_box = (
                    header_x - 5,
                    header_bottom + 5,
                    page.width,
                    page.height
                )
                
                try:
                    cropped_page = page.crop(crop_box)
                    text_block = cropped_page.extract_text()
                    
                    if text_block:
                        lines = text_block.split('\n')
                        for line in lines:
                            clean_line = line.strip()
                            if len(clean_line) > 10:
                                all_sentences.append(clean_line)
                except ValueError:
                    pass

    if not all_sentences:
        print("ERROR: Still found nothing.")
    else:
        with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
            f.write("\n".join(all_sentences))
        print(f"Success! Extracted {len(all_sentences)} sentences to {OUTPUT_TXT}")

def clean_source_text(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    filtered_lines = []
    removed_count = 0
    
    for line in lines:
        cleaned = line.replace('EnglishClass101.com', '')
        stripped = cleaned.strip() 
        
        if not stripped:
            continue

        word_count = len(stripped.split())
        
        if word_count < 4:
            removed_count += 1
            continue
            
        filtered_lines.append(stripped)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(filtered_lines))
    
    print("-" * 30)
    print(f"CLEANING COMPLETE")
    print(f"Original count: {len(lines)}")
    print(f"Removed:        {removed_count} sentences (2-3 words)")
    print(f"Remaining:      {len(filtered_lines)}")
    print("-" * 30)

def analyze_lengths(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    counts = {
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        "over_5": 0
    }
    
    total_sentences = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            word_count = len(line.split())
            total_sentences += 1
            
            if word_count == 2:
                counts[2] += 1
            elif word_count == 3:
                counts[3] += 1
            elif word_count == 4:
                counts[4] += 1
            elif word_count == 5:
                counts[5] += 1
            elif word_count > 5:
                counts["over_5"] += 1

    print(f"ANALYSIS RESULT")
    print(f"Total Sentences: {total_sentences}")
    print(f"2 words:   {counts[2]}")
    print(f"3 words:   {counts[3]}")
    print(f"4 words:   {counts[4]}")
    print(f"5 words:   {counts[5]}")
    print(f"> 5 words: {counts['over_5']}")
    print("-" * 30)

if __name__ == "__main__":
    extract_sentences_visually()
    clean_source_text(OUTPUT_TXT)
    analyze_lengths(OUTPUT_TXT)