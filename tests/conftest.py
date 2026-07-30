import os

import paddle

paddle.enable_compat(scope={"moonep", "tests"})

import pytest

import moonep  # noqa: F401
import paddle.distributed as dist


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

    from tests.kernel_test_utils import local_device_index

    device = local_device_index()
    paddle.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()

    yield rank, dist.get_world_size()

    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
