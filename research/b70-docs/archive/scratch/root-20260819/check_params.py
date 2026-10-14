import torch
from vllm.config import VllmConfig, ModelConfig, CacheConfig, ParallelConfig
from vllm.model_executor.models.nemotron_h_mtp import NemotronHMTP
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="/mnt/<models-disk>/nemotron-lightning-gptq-sym64-mtp-20260812")
args = parser.parse_args()

vllm_config = VllmConfig(model_config=ModelConfig(args.model, args.model, tokenizer_mode="auto", trust_remote_code=False, dtype="float16", seed=0), cache_config=CacheConfig(block_size=16, gpu_memory_utilization=0.9, swap_space=0, cache_dtype="auto"), parallel_config=ParallelConfig(1, 1, False))
vllm_config.model_config.hf_config.num_hidden_layers = 1 # speedup
model = NemotronHMTP(vllm_config=vllm_config)
print([name for name, _ in model.named_parameters() if "embed" in name])
