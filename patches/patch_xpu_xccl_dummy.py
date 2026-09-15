import torch.distributed as dist
import builtins
orig_all_reduce = dist.all_reduce
def dummy_all_reduce(tensor, *args, **kwargs):
    if tensor.numel() == 1 and tensor.device.type == "xpu" and (tensor == 0).all():
        return None
    return orig_all_reduce(tensor, *args, **kwargs)
dist.all_reduce = dummy_all_reduce
