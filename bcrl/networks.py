import torch
import torch.nn as nn


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35
        # self.repr_dim = 20000

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, obs_shape, action_shape, feature_dim, hidden_dim):
        super().__init__()
        self.encoder = Encoder(obs_shape)
        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0]),
        )

        self.apply(weight_init)

    def forward(self, obs):
        encoding = self.encoder(obs)
        h = self.trunk(encoding)
        mu = self.policy(h)
        return torch.tanh(mu)


class Phi(nn.Module):
    def __init__(self, obs_shape, action_shape, feature_dim, hidden_dim, double=True):
        super().__init__()
        self.double = double
        self.encoder = Encoder(obs_shape)
        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )

        # NOTE: perhaps we should do ELU instead of RELU?
        self.phi1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        if double:
            self.phi2 = nn.Sequential(
                nn.Linear(feature_dim + action_shape[0], hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
            )

        self.apply(weight_init)

    def forward(self, obs, action):
        encoding = self.encoder(obs)
        h = self.trunk(encoding)
        h_action = torch.cat([h, action], dim=-1)
        phi1 = self.phi1(h_action)
        if self.double:
            phi2 = self.phi2(h_action)
            return phi1, phi2
        return phi1


class M(nn.Module):
    def __init__(self, hidden_dim, double=True):
        super().__init__()
        self.double = double
        self.hidden_dim = hidden_dim
        self.M_phi1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.M_reward1 = nn.Linear(hidden_dim, 1, bias=False)
        if double:
            self.M_phi2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.M_reward2 = nn.Linear(hidden_dim, 1, bias=False)

        self.apply(weight_init)

    def forward(self, phi1, phi2):
        phi_sp1 = self.M_phi1(phi1)
        reward1 = self.M_reward1(phi1)
        if self.double:
            phi_sp2 = self.M_phi2(phi2)
            reward2 = self.M_reward2(phi2)

            return phi_sp1, reward1, phi_sp2, reward2
        return phi_sp1, reward1
