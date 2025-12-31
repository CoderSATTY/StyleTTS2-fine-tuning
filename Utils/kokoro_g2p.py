import json
import os
from misaki import en
import torch

class KokoroPhonemizer:
    def __init__(self, config_path="Configs/config.json", style="us"):
        if not os.path.exists(config_path):
             pass
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
            self.vocab = self.config['vocab']
            
        self.g2p = en.G2P(trf=False, british=(style=='gb'), fallback=None)
        
    def __call__(self, text):
        try:
            result = self.g2p(text)
            if isinstance(result, tuple):
                phonemes = result[0]
            else:
                phonemes = result
        except Exception as e:
            print(f"Error phonemizing text '{text}': {e}")
            return []

        ids = []
        if isinstance(phonemes, str):
            for char in phonemes:
                if char in self.vocab:
                    ids.append(self.vocab[char])
        elif isinstance(phonemes, list):
             for item in phonemes:
                 if isinstance(item, str):
                      for char in item:
                           if char in self.vocab:
                               ids.append(self.vocab[char])
                 elif isinstance(item, int): # Direct ID pass-through? Likely not but safe to ignore
                      pass
                      
        return ids

if __name__ == "__main__":
    kp = KokoroPhonemizer()
    text = "Hello World."
    result = kp.g2p(text)
    
    phonemes = result[0]

    ids = kp(text)
    print(f"Text: {text}")
    print(f"Phonemes: {phonemes}")
    print(f"IDs: {ids}")
    print(f"Count: {len(ids)}")
