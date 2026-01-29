import torch
import os

file_path = 'my_voice_fixed.pt' 

try:
    data = torch.load(file_path)
    # print(data)
    if isinstance(data, torch.Tensor):
        print(f"Successfully loaded {file_path} tensor.")
        print(data)
        print(f"Shape of the tensor: {data.shape}")
        print(f"Weights (values) of the tensor:\n{data}")
    else:
        print(f"The loaded file does not contain a single tensor. It is a {type(data)}.")
except Exception as e:
        print(e)
