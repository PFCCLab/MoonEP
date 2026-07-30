"""TopK routing generation shared by communication benchmarks and tests."""

import torch

from tests.rng_utils import seeded_cuda_rng


def generate_topk_routing(S, K, E, R, bias_ratio, dev, seed, rank=0):
    """Generate topk_experts and tokens_per_expert with benchmark semantics.

    bias_ratio is the sigma of the underlying lognormal expert-logit
    distribution:
      0   -> round-robin balanced baseline
      0.5 -> mild skew
      1   -> typical dropless-MoE skew
      2   -> heavy skew
      5   -> near-degenerate routing

    `seed` seeds rank-shared state (the expert-logit distribution and the
    round-robin expert permutation); `rank` seeds the per-token draws. This
    is bit-for-bit identical to the biased-routing generator used in
    training when run on the same CUDA device: the shared RNG scope uses
    the base seed and the local scope uses the EP rank, so hot experts line
    up across ranks while per-token draws stay independent.
    """
    if bias_ratio == 0.0:
        epn = E // R
        toks = torch.arange(S, device=dev)
        ks = torch.arange(K, device=dev)
        target_rank = (toks[:, None] + ks[None, :]) % R
        target_local = ((toks[:, None] // R) + ks[None, :]) % epn
        with seeded_cuda_rng(rank, dev):
            perm = torch.randperm(epn, device=dev)
        topk = (target_rank * epn + perm[target_local]).to(torch.int32)
    else:
        with seeded_cuda_rng(seed, dev):
            logits = torch.exp(torch.randn(E, device=dev) * bias_ratio)
        probs = logits[None, :].expand(S, E)
        with seeded_cuda_rng(rank, dev):
            topk = torch.multinomial(probs, K, replacement=False).to(torch.int32)

    tpe = torch.bincount(topk.flatten(), minlength=E).to(torch.int32)
    return topk, tpe
