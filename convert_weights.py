import torch
import yaml
from munch import Munch
import click
import os
import sys

# Add current dir to path to import models
sys.path.append(os.getcwd())

from models import build_model
from utils import recursive_munch

@click.command()
@click.option('--config_path', default='Configs/config_kokoro.yml', type=str)
@click.option('--kokoro_path', default='models/kokoro-v0_19.pth', type=str)
@click.option('--output_path', default='models/epoch_0_kokoro.pth', type=str)
def convert(config_path, kokoro_path, output_path):
    print(f"Loading config from {config_path}")
    config = yaml.safe_load(open(config_path))
    
    # Mock ASR/JDC/PLBERT loading to avoid errors during build_model
    # We only need the model structure to verify keys
    text_aligner = None 
    pitch_extractor = None 
    plbert = None
    
    # ... actually build_model requires these. 
    # We can try to load them if paths exist, or mock them.
    # For conversion, we might not need to instantiate the whole model if we just map keys.
    # But checking against the actual model class is safer.
    
    # Let's try to just load the checkpoint and inspect it.
    print(f"Loading Kokoro weights from {kokoro_path}")
    if not os.path.exists(kokoro_path):
        print("Kokoro weight file not found. Please download it first.")
        return

    kokoro_ckpt = torch.load(kokoro_path, map_location='cpu')
    
    # Handle if it's wrapped in 'net' or not
    if 'net' in kokoro_ckpt:
        kokoro_sd = kokoro_ckpt['net']
    else:
        kokoro_sd = kokoro_ckpt
        
    print(f"Kokoro keys found: {len(kokoro_sd)}")
    
    # Create a new state dict
    new_sd = {}
    
    # Mappings from Kokoro (StyleTTS2-82M) to standard StyleTTS2
    # Standard usually has:
    # 'bert_encoder', 'text_encoder', 'predictor', 'style_encoder', 'decoder', 'bert', 'text_aligner', 'pitch_extractor'
    
    # Inspect some keys
    keys = list(kokoro_sd.keys())
    print("Example keys:", keys[:5])
    
    # Heuristics for mapping
    # 1. Simple copy
    # 2. Rename 'dec.' -> 'decoder.'
    
    for k, v in kokoro_sd.items():
        if k.startswith('dec.'):
            new_k = k.replace('dec.', 'decoder.')
        else:
            new_k = k
            
        new_sd[new_k] = v
        
    # Wrap in expected format for train_finetune_accelerate.py
    # it expects: 'net', 'optimizer', 'epoch', etc for resuming.
    # But usually creating a pretrained model means just saving the dict.
    # However, create inputs for load_checkpoint
    
    final_dict = {
        'net': new_sd,
        'optimizer': None,
        'epoch': 0,
        'iters': 0
    }
    
    print(f"Saving converted model to {output_path}")
    torch.save(final_dict, output_path)
    print("Done.")

if __name__ == "__main__":
    convert()
