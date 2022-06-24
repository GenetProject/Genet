import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from pensieve.a3c.network import ActorNetwork, CriticNetwork

RAND_RANGE = 1000


def entropy_weight_decay_func(epoch):
    # linear decay
    # return np.maximum((1-0.1)/(10**5) * epoch + 1, 0.1)
    return np.maximum(-0.05/(10**4) * epoch + 0.5, 0.1)


class A3C(object):
    def __init__(self, is_central, s_dim, action_dim,
                 actor_lr=1e-4, critic_lr=1e-3):
        self.s_dim = s_dim
        self.a_dim = action_dim
        self.discount = 0.99
        self.entropy_weight = 0.5
        self.entropy_eps = 1e-6

        self.is_central = is_central
        # self.device=torch.device("cuda:0" if torch.cuda.is_available() else
        # "cpu")
        self.device = torch.device(
            "cpu" if torch.cuda.is_available() else "cpu")

        self.actor_network = ActorNetwork(
            self.s_dim, self.a_dim).to(self.device)
        if self.is_central:
            # unify default parameters for tensorflow and pytorch
            self.actor_optim = torch.optim.RMSprop(
                self.actor_network.parameters(), lr=actor_lr, alpha=0.9,
                eps=1e-10)
            self.actor_optim.zero_grad()

            self.critic_network = CriticNetwork(
                self.s_dim, self.a_dim).to(self.device)
            self.critic_optim = torch.optim.RMSprop(
                self.critic_network.parameters(), lr=critic_lr, alpha=0.9,
                eps=1e-10)
            self.critic_optim.zero_grad()
        else:
            self.actor_optim = None
            self.critic_network = None
            self.critic_optim = None
            self.actor_network.eval()

        self.loss_function = nn.MSELoss()

    def get_network_gradient(self, s_batch, a_batch, r_batch, terminal, epoch):
        s_batch = torch.from_numpy(s_batch).type('torch.FloatTensor')
        a_batch = torch.LongTensor(a_batch).to(self.device)
        r_batch = torch.tensor(r_batch).to(self.device)
        R_batch = torch.zeros(r_batch.shape).to(self.device)

        R_batch[-1] = r_batch[-1]
        for t in reversed(range(r_batch.shape[0]-1)):
            R_batch[t] = r_batch[t] + self.discount*R_batch[t+1]

        with torch.no_grad():
            v_batch = self.critic_network.forward(
                s_batch).squeeze().to(self.device)
        td_batch = R_batch-v_batch
        # else:
        #     td_batch = R_batch

        probability = self.actor_network.forward(s_batch)
        m_probs = Categorical(probability)
        log_probs = m_probs.log_prob(a_batch)
        actor_loss = torch.sum(log_probs*(-td_batch))
        # entropy_loss=-self.entropy_weight*torch.sum(m_probs.entropy())
        entropy_loss = - \
            entropy_weight_decay_func(epoch)*torch.sum(m_probs.entropy())
        actor_loss = actor_loss+entropy_loss
        actor_loss.backward()

        critic_loss = self.loss_function(
            R_batch, self.critic_network.forward(s_batch).squeeze())

        critic_loss.backward()

        # use the feature of accumulating gradient in pytorch

    def select_action(self, stateInputs):
        # if not self.is_central:
        if isinstance(stateInputs, np.ndarray):
            stateInputs = torch.from_numpy(stateInputs).type('torch.FloatTensor')
        with torch.no_grad():
            stateInputs_gpu = stateInputs.to(self.device)
            probability = self.actor_network.forward(stateInputs_gpu)
            m = Categorical(probability)
            action = m.sample().detach().cpu().numpy()
            return action, probability.detach().cpu().numpy()

    def hard_update_actor_network(self, actor_net_params):
        for target_param, source_param in zip(self.actor_network.parameters(),
                                              actor_net_params):
            if isinstance(source_param, np.ndarray):
                target_param.data.copy_(
                    torch.from_numpy(source_param).to(self.device))
            elif torch.is_tensor(source_param):
                target_param.data.copy_(source_param.data)

    def update_network(self):
        # use the feature of accumulating gradient in pytorch
        if self.is_central:
            self.actor_optim.step()
            self.actor_optim.zero_grad()
            self.critic_optim.step()
            self.critic_optim.zero_grad()

    def get_actor_param(self):
        return list(self.actor_network.parameters())

    def get_critic_param(self):
        if self.is_central:
            return list(self.critic_network.parameters())

    def load_actor_model(self, actor_model_path):
        self.actor_network.load_state_dict(torch.load(actor_model_path))

    def load_critic_model(self, critic_model_path):
        if self.is_central:
            self.critic_network.load_state_dict(torch.load(critic_model_path))

    def save_actor_model(self, model_save_path):
        torch.save(self.actor_network.state_dict(), model_save_path)

    def save_critic_model(self, model_save_path):
        if self.is_central:
            torch.save(self.critic_network.state_dict(), model_save_path)


def compute_entropy(x):
    """Given vector x, computes the entropy H(x) = - sum( p * log(p))."""
    return -1 * np.nansum(x * np.log(x), axis=1)
    # H = 0.0
    # for i in range(len(x)):
    #     if 0 < x[i] < 1:
    #         H -= x[i] * np.log(x[i])
    # return H
