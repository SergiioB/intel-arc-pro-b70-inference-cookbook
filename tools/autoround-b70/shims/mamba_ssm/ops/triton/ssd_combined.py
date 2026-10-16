def _missing(*a, **k):
    raise NotImplementedError("mamba_ssm CUDA kernel not available on this host")

mamba_chunk_scan_combined = _missing
mamba_split_conv1d_scan_combined = _missing
