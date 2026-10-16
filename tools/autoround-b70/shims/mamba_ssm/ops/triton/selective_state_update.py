def _missing(*a, **k):
    raise NotImplementedError("mamba_ssm CUDA kernel not available on this host")

selective_state_update = _missing
