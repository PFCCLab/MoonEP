import os

import paddle

paddle.enable_compat(scope={"moonep", "tests"})

import pytest

import moonep  # noqa: F401
import paddle.distributed as dist


def _local_rank():
    for name in (
        "LOCAL_RANK",
        "PADDLE_LOCAL_RANK",
        "PADDLE_RANK_IN_NODE",
        "RANK",
        "PADDLE_TRAINER_ID",
    ):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    raise RuntimeError("distributed launcher did not set a rank environment variable")


@pytest.fixture(autouse=True)
def cleanup_moonep_buffers():
    yield
    from tests.kernel_test_utils import destroy_active_buffers

    destroy_active_buffers()


@pytest.fixture(scope="session")
def dist_env():
    if "RANK" not in os.environ and "PADDLE_TRAINER_ID" not in os.environ:
        pytest.skip(
            "distributed kernel tests must be launched with torchrun "
            "or paddle.distributed.launch"
        )

    local_rank = _local_rank()
    paddle.cuda.set_device(local_rank)
    if not dist.is_initialized():
        # TODO(Paddle compat): use init_process_group once the
        # torch.distributed proxy provides it.
        dist.init_parallel_env()
    rank = dist.get_rank()

    yield rank, dist.get_world_size()

    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
