import io
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


def load_episode(fn, relevant_keys):
    # Loads episode and only grabs relevant
    with fn.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in relevant_keys}
        return episode


# NOTE: techincally since we aren't adding in more data we do not need iterable dataset.....
class ReplayBuffer(IterableDataset):
    def __init__(self, offline_dir, num_workers, num_episodes=None, init_states=False):
        self.fns = list(offline_dir.glob("*.npz"))
        if num_episodes:
            self.fns = self.fns[:num_episodes]
        self.relevant_keys = ["observation", "action", "reward"]
        self.init_states = init_states
        self.num_workers = num_workers
        self.episodes = dict()
        self.size = 0
        self._prefetch()

    def _prefetch(self):
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except Exception:
            worker_id = 0
        for fn in self.fns:
            eps_idx, eps_len = [int(x) for x in fn.stem.split("_")[1:]]
            if eps_idx % self.num_workers != worker_id:
                continue
            if fn in self.episodes.keys():
                break
            # Load episode
            try:
                traj = load_episode(fn, self.relevant_keys)
            except Exception:
                raise RuntimeError(f"Could not load episode {eps_idx}")
            self.size += eps_len
            self.episodes[fn] = traj

    def _sample(self):
        eps_fn = random.choice(list(self.episodes.keys()))
        traj = self.episodes[eps_fn]
        # If we only want initial states then pick beginning of traj
        if self.init_states:
            idx = 0
        else:
            idx = np.random.choice(np.arange(0, traj["observation"].shape[0] - 1))

        obs = torch.from_numpy(traj["observation"][idx]).float()
        act = torch.from_numpy(traj["action"][idx + 1]).float()
        next_obs = torch.from_numpy(traj["observation"][idx + 1]).float()
        reward = torch.from_numpy(traj["reward"][idx + 1]).float()

        return obs, act, next_obs, reward

    def __iter__(self):
        while True:
            yield self._sample()


class ReplayMap(Dataset):
    def __init__(self, offline_dir, mixture_dir=None, num_episodes=None):
        # For now, no functionality for both mixture and num_episodes
        self.fns = list(offline_dir.glob("*.npz"))
        if num_episodes:
            self.fns = self.fns[:num_episodes]

        if mixture_dir:
            mix_fns = list(mixture_dir.glob("*.npz"))
            self.fns = [*self.fns, *mix_fns]

        self.relevant_keys = ["observation", "action", "reward"]
        self.data = defaultdict(list)
        self.size = 0
        self._prefetch()

    def _prefetch(self):
        for fn in self.fns:
            eps_idx, eps_len = [int(x) for x in fn.stem.split("_")[1:]]
            self.size += eps_len
            try:
                traj = load_episode(fn, self.relevant_keys)
            except Exception:
                raise RuntimeError(f"Could not load episode {eps_idx}")
            for k, v in traj.items():
                if k == "observation":
                    self.data[k].append(v[:-1])
                    self.data["next_observation"].append(v[1:])  # NOTE: double save
                else:
                    self.data[k].append(v[1:])

        for k, v in self.data.items():
            self.data[k] = np.concatenate(v, axis=0)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        obs = torch.from_numpy(self.data["observation"][idx]).float()
        act = torch.from_numpy(self.data["action"][idx]).float()
        next_obs = torch.from_numpy(self.data["next_observation"][idx]).float()
        reward = torch.from_numpy(self.data["reward"][idx]).float()

        # (S, A, S', R)
        return obs, act, next_obs, reward


class SamplesDataset(Dataset):
    def __init__(self, offline_dir, num_episodes=None):
        self.fns = list(offline_dir.glob("*.npz"))
        if num_episodes:
            self.fns = self.fns[:num_episodes]
        self.relevant_keys = ["observation", "action"]
        self.data = defaultdict(list)
        self.size = 0
        self._prefetch()

    def _prefetch(self):
        for fn in self.fns:
            eps_idx, eps_len = [int(x) for x in fn.stem.split("_")[1:]]
            self.size += eps_len
            try:
                traj = load_episode(fn, self.relevant_keys)
            except Exception:
                raise RuntimeError(f"Could not load episode {eps_idx}")
            for k, v in traj.items():
                if k == "observation":
                    self.data[k].append(v[:-1])
                else:
                    self.data[k].append(v[1:])

        for k, v in self.data.items():
            self.data[k] = np.concatenate(v, axis=0)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        obs = torch.from_numpy(self.data["observation"][idx]).float()
        act = torch.from_numpy(self.data["action"][idx]).float()

        return obs, act


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def create_buffer(data_path, cfg, init_states=False):
    buffer = ReplayBuffer(data_path, cfg.num_workers, init_states=init_states)
    loader = DataLoader(
        buffer,
        batch_size=cfg.agent_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    return loader


def create_epoch_loader(data_path, cfg, mixture_path=None):
    db = ReplayMap(data_path, mixture_dir=mixture_path)
    loader = DataLoader(
        db,
        batch_size=cfg.ope_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    return loader


def create_bc_loader(data_path, cfg):
    db = SamplesDataset(data_path)
    loader = DataLoader(
        db,
        batch_size=cfg.bc_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return loader
