from pathlib import Path

import torch
import numpy as np
import hydra
from hydra.utils import to_absolute_path


# For now just import everything
from bcrl.agent import BCRL
from bcrl.utils import set_seed_everywhere, Until, Every
from bcrl.logger import Logger
from bcrl.replay_buffer import create_epoch_loader
import bcrl.dmc as dmc
import logging

import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

log = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True


def to_torch(x):
    return torch.from_numpy(x).float()


def load_eval(path):
    with path.open("rb") as f:
        db = np.load(f)
        db = {k: to_torch(db[k]) for k in ["observation", "action", "return"]}
        return db


class Workspace:
    def __init__(self, cfg):

        self.work_dir = Path.cwd()
        log.info(f"Current Workspace: {self.work_dir}")

        self.cfg = cfg

        self.setup()
        log.info("Setup Complete")

        self.agent = BCRL(cfg, self.target_checkpoint, self.device)
        self.agent_copy = BCRL(cfg, self.target_checkpoint, torch.device("cpu"))

        self.global_step = 0

    def setup(self):
        # Set Seeds
        set_seed_everywhere(self.cfg.seed)

        # Logger
        self.logger = Logger(self.work_dir)

        # Device
        device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # does not support multigpu as is
        self.device = torch.device(device)

        # Create Env
        self.env = dmc.make(
            self.cfg.task,
            self.cfg.frame_stack,
            self.cfg.action_repeat,
            self.cfg.seed,
            self.cfg.img_size,
        )
        self.cfg.obs_shape = self.env.observation_spec().shape
        self.cfg.action_shape = self.env.action_spec().shape
        self.cfg.n_actions = self.cfg.action_shape[0]

        # Create Data Paths
        data_root = Path(to_absolute_path("data")) / self.cfg.task
        self.offline_dir = data_root / self.cfg.offline_db
        self.target_dir = (
            data_root / self.cfg.target_policy if self.cfg.mix_data else None
        )
        self.target_checkpoint = (
            data_root / "checkpoints" / f"{self.cfg.target_policy}.pt"
        )

        # Evaluation
        eval_init_path = data_root / f"eval_init_{self.cfg.target_policy}.npz"
        eval_every_path = data_root / f"eval_every_{self.cfg.target_policy}.npz"
        self.init_db = load_eval(eval_init_path)
        self.every_db = load_eval(eval_every_path)

        # Create Replay Buffer
        self.replay_buffer = create_epoch_loader(
            self.offline_dir, self.cfg, mixture_path=self.target_dir
        )

        # For Mixture data no design regularization
        if self.cfg.mix_data:
            self.cfg.design_weight = 0.0

    # ======== Training =======
    def train(self):

        # Init Loops
        train_until = Until(self.cfg.train_steps)
        eval_every = Every(self.cfg.eval_every_steps)

        log.info("Training Starting")
        while train_until(self.global_step):
            log.info(f"Starting global_step={self.global_step}")

            info = {"global_step": self.global_step}

            # Evaluate
            if eval_every(self.global_step):
                # LSPE
                lspe_info = self.agent.lspe(self.replay_buffer)
                info = info | lspe_info

                eval_info = self.eval_agent()
                info = info | eval_info

            # Update
            agent_info = self.agent.update(self.replay_buffer, self.global_step)
            info = info | agent_info

            self.global_step += 1

            # Save Model
            if self.cfg.save_checkpoint:
                self.save_checkpoint()

    # ======== Saving/Loading Checkpoint ========

    def save_checkpoint(self):
        snapshot = self.work_dir / "checkpoint.pt"
        keys_to_save = ["global_step"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(self.agent.get_checkpoint())
        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def load_checkpoint(self, ckpt):
        with Path(ckpt).open("rb") as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v

    # ======== Evaluation ========

    def eval_agent(self):
        def rmse(db, agent):
            pred = agent.Q(db["observation"], db["action"])
            return torch.sqrt(torch.pow(pred - db["return"], 2).mean())

        self.agent_copy.load_checkpoint(
            self.agent.get_checkpoint(), torch.device("cpu")
        )
        init_rmse = rmse(self.init_db, self.agent_copy)
        every_rmse = rmse(self.every_db, self.agent_copy)
        log.info(f"RMSE INIT | EVERY: {init_rmse} | {every_rmse}")
        return {
            "Init RMSE": init_rmse,
            "Every RMSE": every_rmse,
        }


@hydra.main(config_path="cfgs", config_name="config")
def main(cfg):
    workspace = Workspace(cfg)
    workspace.train()


if __name__ == "__main__":
    main()
