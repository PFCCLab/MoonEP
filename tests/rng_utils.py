"""Random helpers shared by MoonEP tests."""

from contextlib import contextmanager

import torch


@contextmanager
def seeded_cuda_rng(seed, device):
    """Use a deterministic CUDA RNG stream without changing outer state."""
    state = torch.cuda.get_rng_state(device)
    previous_device = torch.cuda.current_device()
    torch.cuda.set_device(device)
    try:
        torch.cuda.manual_seed(seed)
        yield
    finally:
        try:
            torch.cuda.set_rng_state(state, device)
        finally:
            torch.cuda.set_device(previous_device)
