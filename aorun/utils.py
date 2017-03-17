import torch
import numpy as np
from functools import wraps


def numpyio(method):
    @wraps(method)
    def decorator(self, *args, **kwargs):
        has_np = False
        args = list(args)
        for i, arg in enumerate(args):
            if type(arg) is np.ndarray:
                args[i] = torch.from_numpy(arg)
                has_np = True
        for key, arg in kwargs.items():
            if type(arg) is np.ndarray:
                kwargs[key] = torch.from_numpy(arg)
                has_np = True

        out = method(self, *args, **kwargs)

        if has_np and type(out) is torch.Tensor:
            out = out.numpy()
        elif has_np and type(out) is torch.autograd.Variable:
            out = out.data.numpy()

        return out

    return decorator
