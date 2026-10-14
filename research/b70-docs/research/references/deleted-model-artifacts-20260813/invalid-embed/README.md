---
library_name: mlx
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
pipeline_tag: text-generation
language:
- en
- es
- fr
- de
- it
- ja
tags:
- nvidia
- pytorch
- nemotron-3.5
- mlx
datasets:
- nvidia/nemotron-post-training-v3
- nvidia/nemotron-pre-training-datasets
track_downloads: true
base_model: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
---

# mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit

This model [mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit](https://huggingface.co/mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit) was
converted to MLX format from
[nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
using mlx-lm version **0.31.3** (4-bit, group size 64).

NVIDIA-Nemotron-3.5-Lightning-30B-A3B is a hybrid Mamba-2 / attention
Mixture-of-Experts model (~31B total parameters, ~3B active per token).

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit")

prompt = "Hello, who are you?"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```
