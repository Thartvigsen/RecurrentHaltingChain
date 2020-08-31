from torch import nn
import torch
from torch.distributions import Bernoulli

class BaselineNetwork(nn.Module):
    """
    A network which predicts the average reward observed
    during a markov decision-making process.
    Weights are updated w.r.t. the mean squared error between
    its prediction and the observed reward.
    """

    def __init__(self, input_size, output_size):
        super(BaselineNetwork, self).__init__()

        # --- Mappings ---
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, h_t):
        b_t = self.fc(h_t.detach())
        return b_t

class Controller(nn.Module):
    """
    A network that chooses whether or not enough information
    has been seen to predict a label of a time series.
    """
    def __init__(self, ninp, nout):
        super(Controller, self).__init__()

        # --- Mappings ---
        self.fc = nn.Linear(ninp, nout)  # Optimized w.r.t. reward

    def forward(self, h, eps=0.):
        """Read in hidden state, predict one halting probability per class"""
        # Predict one probability per class
        probs = torch.sigmoid(self.fc(x))

        # Balance between explore/exploit by randomly picking some actions
        probs = (1-self._epsilon)*probs + self._epsilon*torch.FloatTensor([0.05])  # Explore/exploit (can't be 0)

        # Parameterize bernoulli distribution with prediced probabilities
        m = Bernoulli(probs=probs)

        # Sample an action and compute the log probability of that action being
        # picked (for use during optimization)
        action = m.sample() # sample an action
        log_pi = m.log_prob(action) # compute log probability of sampled action

        # We also return the negative log probability of the probabilities themselves
        # if we minimize this, it will maximize the likelihood of halting!
        return action, log_pi, -torch.log(probs)
