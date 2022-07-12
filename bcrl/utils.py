import time
import random

import torch
import numpy as np

from collections import defaultdict
import logging

log = logging.getLogger(__name__)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_gradient_norm(model):
    params = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    if len(params) == 0:
        return 0.0
    device = params[0].device
    norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), p=2).to(device) for p in params]), p=2
    )
    return norm.item()


def get_test_traj(path, device, seq_len=50):
    test_fn_dir = list(path.glob("*.npz"))
    test_fn = np.random.choice(test_fn_dir)
    traj = np.load(test_fn)
    start_idx = np.random.randint(0, 501 - seq_len)
    obs = (
        torch.from_numpy(traj["observation"][start_idx : start_idx + seq_len])
        .unsqueeze(1)
        .to(device)
    )
    act = (
        torch.from_numpy(traj["action"][start_idx : start_idx + seq_len])
        .unsqueeze(1)
        .to(device)
    )

    # grab actual
    seq = traj["observation"][start_idx : start_idx + seq_len]  # (seq_len, 9, 64, 64)
    actual = [np.moveaxis(x[:3, :, :].astype(np.uint8), 0, -1) for x in seq]
    return (obs, act), actual


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._last_time


class Aggregator:
    def __init__(self):
        self.info = defaultdict(list)
        self.initialized_keys = False

    def append(self, new_info):
        for k, v in new_info.items():
            if self.initialized_keys:
                if k not in self.info:
                    log.info(f"{k} not found in {list(self.info.keys())}")
                assert k in self.info, f"{k} not found in {list(self.info.keys())}"

            if isinstance(v, torch.Tensor):
                v = v.cpu().detach().item()
            elif isinstance(v, np.ndarray):
                v = v.item()

            self.info[k].append(v)

        self.initialized_keys = True

    def aggregate(self):
        assert self.initialized_keys
        return {k: np.mean(v) for k, v in self.info.items()}
