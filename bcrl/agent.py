import torch
import torch.nn.functional as F
from torch.optim import Adam

from bcrl.networks import Actor, Phi, M
from bcrl.utils import Aggregator
import logging

log = logging.getLogger(__name__)


class BCRL:
    def __init__(self, cfg, target_chkpt, device):
        super().__init__()
        self.cfg = cfg
        self.gamma = cfg.gamma
        self.device = device
        self.feature_dim = cfg.hidden_dim

        # Actor
        self.actor = Actor(
            cfg.obs_shape, cfg.action_shape, cfg.feature_dim, cfg.actor_hidden_dim
        )
        # Load target Actor
        chkpt = torch.load(target_chkpt, map_location=torch.device("cpu"))
        self.actor.encoder.load_state_dict(chkpt["encoder"])
        self.actor.load_state_dict(chkpt["actor"], strict=False)

        self.phi = Phi(
            cfg.obs_shape,
            cfg.action_shape,
            cfg.feature_dim,
            cfg.hidden_dim,
            double=False,
        )
        self.M = M(cfg.hidden_dim, double=False)
        self.theta = torch.zeros(cfg.hidden_dim)

        # Optimizers
        self.phi_opt = Adam(self.phi.parameters(), lr=cfg.phi_lr, eps=cfg.phi_eps)
        self.M_opt = Adam(self.M.parameters(), lr=cfg.m_lr, eps=cfg.m_eps)

        self.to(device)

    def act(self, obs, eval_mode):
        next_act = self.actor(torch.from_numpy(obs).unsqueeze(0))
        return next_act.squeeze(0).numpy()

    def log_det(self, A):
        assert A.dim() in [2, 3]
        # regularize when computing log-det
        A = A + self.cfg.design_cov_reg * torch.eye(A.shape[1], device=A.device)
        return 2 * torch.linalg.cholesky(A).diagonal(dim1=-2, dim2=-1).log().sum(-1)

    def update_phi(self, obs, act, reward, next_obs, next_act):
        phi_t = self.phi(obs, act)
        phi_tp1 = self.phi(next_obs, next_act)

        pred_tp1, pred_r = self.M(phi_t, None)

        # Design Loss
        cov = torch.bmm(phi_t.unsqueeze(-1), phi_t.unsqueeze(-2)).mean(dim=0)
        design_loss = -self.log_det(cov)

        # Reward Loss
        reward_loss = F.mse_loss(pred_r, reward)

        # Transition Loss
        BC_loss = (torch.norm(pred_tp1 - self.gamma * phi_tp1, dim=1) ** 2).mean()

        # Combined Loss
        phi_loss = (
            self.cfg.reward_weight * reward_loss
            + BC_loss
            + self.cfg.design_weight * design_loss
        )

        self.phi_opt.zero_grad(set_to_none=True)
        phi_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.phi.parameters(), self.cfg.phi_grad_norm_clip
        )
        self.phi_opt.step()

        return {
            "phi/design_loss": design_loss,
            "phi/reward_loss": reward_loss,
            "phi/BC_loss": BC_loss,
            "phi/combined_loss": phi_loss,
        }

    def update_M(self, obs, act, reward, next_obs, next_act):
        with torch.no_grad():
            phi_t = self.phi(obs, act)
            phi_tp1 = self.phi(next_obs, next_act)

        pred_tp1, pred_r = self.M(phi_t, None)
        reward_loss = F.mse_loss(pred_r, reward)

        BC_loss = (torch.norm(pred_tp1 - self.gamma * phi_tp1, dim=1) ** 2).mean()
        M_update_loss = self.cfg.reward_weight * reward_loss + BC_loss

        self.M_opt.zero_grad(set_to_none=True)
        M_update_loss.backward()
        self.M_opt.step()

        return {
            "M/BC_loss": BC_loss,
            "M/reward_loss": reward_loss,
            "M/combined_loss": M_update_loss,
        }

    @torch.no_grad()
    def lspe(self, loader):
        cov_sum = None
        n = 0
        phi_t, phi_tp1, reward = [], [], []
        for batch in loader:
            obs, act, next_obs, batch_rew = [x.to(self.device) for x in batch]
            with torch.no_grad():
                next_act = self.actor(next_obs)

            batch_phi_t = self.phi(obs, act)
            batch_phi_tp1 = self.phi(next_obs, next_act)
            if cov_sum is None:
                cov_sum = torch.zeros(batch_phi_t.size(-1), batch_phi_t.size(-1)).to(
                    self.device
                )
            cov_sum += torch.bmm(
                batch_phi_t.unsqueeze(-1), batch_phi_t.unsqueeze(-2)
            ).sum(dim=0)

            # Store
            phi_t.append(batch_phi_t)
            phi_tp1.append(batch_phi_tp1)
            reward.append(batch_rew)
            n += obs.shape[0]

        cov = cov_sum / n
        inv_cov = torch.linalg.pinv(cov)
        phi_t = torch.cat(phi_t, dim=0)
        phi_tp1 = torch.cat(phi_tp1, dim=0)
        reward = torch.cat(reward, dim=0)
        assert reward.dim() == 2 and reward.shape[-1] == 1

        # initialize at 0
        self.theta.zero_()
        for _ in range(self.cfg.lspe_iter):
            """
            Shapes
            prev_q: [N,1]
            y: [N,1]

            We want to solve \min_\theta \frac1N \sum_i (\theta^T \phi_i - y_i)^2
            optimality condition: \frac1N \sum_i \phi_i \phi_i^T \theta = \frac1N \sum_i \phi_i y_i
            So, \theta = inv_cov @ (\frac1N \sum_i \phi_i y_i).
            """

            prev_q = (phi_tp1 @ self.theta).unsqueeze(-1)
            y = reward + self.gamma * prev_q
            pred = torch.mean(phi_t * y, dim=0)
            self.theta = inv_cov @ pred

        """
        Residual is the actual objective function.
        """
        residual = torch.mean((phi_t @ self.theta - y.squeeze(1)) ** 2)
        theta_norm = torch.norm(self.theta)
        phi_norm = torch.norm(phi_t, dim=1).mean()
        return {
            "lspe/residual": residual,
            "lspe/theta_norm": theta_norm,
            "lspe/phi_norm": phi_norm,
            "lspe/neg_logdet": -self.log_det(cov),
        }

    @torch.no_grad()
    def Q(self, obs, act):
        phi = self.phi(obs, act)
        Q = phi @ self.theta
        return Q.unsqueeze(-1)

    def update(self, loader, step):
        combined_info = {}
        # M Update
        M_info = Aggregator()
        for batch in loader:
            obs, act, next_obs, rew = [x.to(self.device) for x in batch]
            with torch.no_grad():
                next_act = self.actor(next_obs)
            info = self.update_M(obs, act, rew, next_obs, next_act)
            M_info.append(info)
        M_info = M_info.aggregate()
        combined_info = combined_info | M_info

        # Phi Update
        phi_info = Aggregator()
        for batch in loader:
            obs, act, next_obs, rew = [x.to(self.device) for x in batch]
            with torch.no_grad():
                next_act = self.actor(next_obs)
            info = self.update_phi(obs, act, rew, next_obs, next_act)
            phi_info.append(info)
        phi_info = phi_info.aggregate()

        combined_info = combined_info | phi_info
        return combined_info

    def get_checkpoint(self):
        return {
            "actor_state_dict": self.actor.state_dict(),
            "phi_state_dict": self.phi.state_dict(),
            "M_state_dict": self.M.state_dict(),
            "theta": self.theta,
            "phi_opt_state_dict": self.phi_opt.state_dict(),
            "M_opt_state_dict": self.M_opt.state_dict(),
        }

    def load_checkpoint(self, chkpt, device):
        self.actor.load_state_dict(chkpt["actor_state_dict"])
        self.phi.load_state_dict(chkpt["phi_state_dict"])
        self.M.load_state_dict(chkpt["M_state_dict"])
        self.theta = chkpt["theta"]
        self.phi_opt.load_state_dict(chkpt["phi_opt_state_dict"])
        self.M_opt.load_state_dict(chkpt["M_opt_state_dict"])

        self.to(device)

    def to(self, device):
        self.actor.to(device)
        self.phi.to(device)
        self.M.to(device)
        self.theta = self.theta.to(device)
