"""
Deep Markov Model for sequence modelling.
Implements the Deep Kalman Smoother (DKL) as described in [1].

FUTURE WORK:
It is possible to use the approach described in [3] and model the latent variabels
with a normalizing flow, as shown by [2].

[1] Structured Inference Networks for Nonlinear State Space Models https://arxiv.org/abs/1609.09869
[2] Improving Variational Inference with Inverse Autoregressive Flow https://arxiv.org/abs/1606.04934v2
[3] Variational Inference with Normalizing Flows https://arxiv.org/abs/1505.05770
"""
from pathlib import Path
from typing import Optional, Tuple, List, Sequence
from os.path import exists

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence

import pyro
from pyro import poutine
from pyro.distributions import Normal
from pyro.optim import PyroOptim
from pyro.poutine import trace


class Emit(nn.Module):
    """Parameterizes the observation likelihood p(x_t | z_t)"""

    def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Assume observation model is in location-scale family
        self.loc = nn.Linear(hidden_dim, obs_dim)
        self.scale = nn.Sequential(nn.Linear(hidden_dim, obs_dim), nn.Softplus())
        # self.scale = lambda x: x.new_ones(x.size(0), obs_dim) * 0.1

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.hidden(z)
        loc = self.loc(h)
        scale = self.scale(h)
        return loc, scale


class Transition(nn.Module):
    """Parameterizes the latent transition probability p(z_t | z_{t-1})"""

    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()

        # Initialize linear dynamics as the identity function
        self.linear_loc = nn.Linear(latent_dim, latent_dim)
        self.linear_loc.weight.data = torch.eye(latent_dim)
        self.linear_loc.bias.data = torch.zeros(latent_dim)

        self.nonlinear_loc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.gate = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid(),
        )

        self.scale = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Softplus(),
        )

    def forward(self, z):
        a = self.gate(z)
        loc = (1 - a) * self.linear_loc(z) + a * self.nonlinear_loc(z)
        scale = self.scale(z)
        return loc, scale


class Combine(nn.Module):
    """
    Models the parameters for q(z_t | z_{t-1}, h_t) = q(z_t | z_{t-1}, x_{t:T}).
    The dependence on x_{t:T} is through the hidden state h_t of the encoder.
    We assume that x_{t:T} and z_t are conditionally independent given h_t.
    """

    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.propagate_z = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.Tanh())

        # Assume latent variable is in location-scale family
        self.loc = nn.Linear(hidden_dim, latent_dim)
        self.scale = nn.Sequential(nn.Linear(hidden_dim, latent_dim), nn.Softplus())
        # self.scale = lambda x: x.new_ones(x.size(0), latent_dim) * 0.1

    def forward(self, h: Tensor, z_prev: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Average previous latent z_{t-1} and current hidden state h_t
        and compute parameters for q(z_t | z_{t-1}, h_t).
        """
        h_combined = 0.5 * (self.propagate_z(z_prev) + h)
        loc = self.loc(h_combined)
        scale = self.scale(h_combined)
        return loc, scale


class DMM(nn.Module):
    """Implementation of a Deep Markov Model as described in [1].

    [1] Structured Inference Networks for Nonlinear State Space Models, https://arxiv.org/abs/1609.09869
    """

    def __init__(
        self,
        data_dim: int,
        latent_dim: int,
        encode_hidden_dim: int,
        encode_num_layers: int,
        encode_dropout_rate: float,
        transition_hidden_dim: int,
        emit_hidden_dim: int,
    ):
        super().__init__()
        self.encode = nn.GRU(
            input_size=data_dim,
            hidden_size=encode_hidden_dim,
            # nonlinearity="relu",
            batch_first=True,
            bidirectional=False,
            num_layers=encode_num_layers,
            dropout=0.0 if encode_num_layers == 1 else encode_dropout_rate,
        )
        self.combine = Combine(latent_dim=latent_dim, hidden_dim=encode_hidden_dim)
        self.transition = Transition(latent_dim, transition_hidden_dim)
        self.emit = Emit(data_dim, latent_dim, emit_hidden_dim)
        self.pz0 = nn.Parameter(torch.zeros(latent_dim))
        self.qz0 = nn.Parameter(torch.zeros(latent_dim))
        self.h0 = nn.Parameter(torch.zeros(encode_num_layers, 1, encode_hidden_dim))
        # self.pz0 = nn.Parameter(torch.randn(latent_dim) * 0.1)
        # self.qz0 = nn.Parameter(torch.randn(latent_dim) * 0.1)
        # self.h0 = nn.Parameter(
        #     torch.randn(encode_num_layers, 1, encode_hidden_dim) * 0.1
        # )

    def model(
        self,
        x: torch.Tensor,
        x_packed_reversed: nn.utils.rnn.PackedSequence,
        seq_mask: torch.Tensor,
        seq_lengths: torch.Tensor,
        annealing: float = 1.0,
    ) -> None:
        """
        :x: Batch of observation of dimensions BxTxD (batch, temporal and data
        :annealing: Value to scale the KL-loss for z. [1] show good results
        when annealing 0->1 over 5000 update steps when modeling 88-dimensional time series.
        """
        batch_dim, time_steps, _ = x.shape
        return self._model(
            z0=self.pz0,
            batch_dim=batch_dim,
            time_steps=time_steps,
            x=x,
            seq_mask=seq_mask,
            annealing=annealing,
        )

    def _model(
        self,
        z0: Tensor,
        batch_dim: int,
        time_steps: int,
        x: Optional[Tensor] = None,
        seq_mask: Optional[Tensor] = None,
        annealing: float = 1.0,
    ) -> None:
        pyro.module("dmm", self)
        seq_mask = seq_mask if seq_mask is not None else torch.ones(z0.size(0),  time_steps)

        z = z0.expand(batch_dim, z0.size(-1))
        with pyro.plate("data", batch_dim):
            for t in pyro.markov(range(time_steps)):
                m = seq_mask[:, t : t + 1]
                z_loc, z_scale = self.transition(z)
                with poutine.scale(None, annealing):
                    z = pyro.sample(f"z_{t+1}", Normal(z_loc, z_scale).mask(m).to_event(1))
                
                x_loc, x_scale = self.emit(z)
                pyro.sample(
                    f"x_{t+1}",
                    Normal(x_loc, x_scale).mask(m).to_event(1),
                    obs=x[:, t, :] if x is not None else None,
                )

    def reverse_sequences(self, mini_batch, seq_lengths):
        reversed_mini_batch = torch.zeros_like(mini_batch)
        for b in range(mini_batch.size(0)):
            T = seq_lengths[b]
            time_slice = torch.arange(T - 1, -1, -1, device=mini_batch.device)
            reversed_sequence = torch.index_select(mini_batch[b, :, :], 0, time_slice)
            reversed_mini_batch[b, 0:T, :] = reversed_sequence
        return reversed_mini_batch

    def guide(
        self,
        x: torch.Tensor,
        x_packed_reversed: nn.utils.rnn.PackedSequence,
        seq_mask: torch.Tensor,
        seq_lengths: torch.Tensor,
        annealing=1.0,
    ) -> Tensor:

        pyro.module("dmm", self)
        batch_dim, time_steps, _ = x.shape
        h0 = self.h0.expand(self.h0.size(0), batch_dim, self.h0.size(-1)).contiguous()
        h_packed_reversed = self.encode(x_packed_reversed, h0)[0]
        h_reversed, _ = pad_packed_sequence(h_packed_reversed, batch_first=True)
        h = self.reverse_sequences(h_reversed, seq_lengths)
        z = self.qz0.expand(batch_dim, self.qz0.size(-1))
        with pyro.plate("data", batch_dim):
            for t in range(time_steps):
                z_params = self.combine(h[:, t, :], z)
                with poutine.scale(None, annealing):
                    z = pyro.sample(
                        f"z_{t+1}",
                        Normal(*z_params).mask(seq_mask[:, t : t + 1]).to_event(1),
                    )
        return z

    @torch.no_grad()
    def sample(self, num_steps: int, num_samples: Optional[int] = None, x: Optional[Tensor] = None) -> Tensor:
        num_samples = x.size(0) if x is not None else (num_samples or 1)
        z = self.guide(*pack_sequences(list(x))) if x is not None else self.qz0 
        nodes = trace(self._model).get_trace(z, num_samples, num_steps).nodes
        x_pred = torch.stack([nodes[f"x_{i+1}"]["value"] for i in range(num_steps)], 1)
        return x_pred

    @torch.no_grad()
    def latents(self, x: Tensor) -> Tensor:
        nodes = trace(self.guide).get_trace(*pack_sequences(list(x))).nodes
        means = [nodes[f"z_{i+1}"]['fn'].mean for i in range(x.size(1))]
        return torch.stack(means, dim=1)
        #nodes = trace(self._model).get_trace(z, num_samples, num_steps).nodes
        #x_pred = torch.stack([nodes[f"x_{i+1}"]["value"] for i in range(num_steps)], 1)
        #return x_pred

def save_checkpoint(path: Path, dmm: DMM, optimizer: PyroOptim) -> None:
    assert exists(path), "Invalid save path"
    torch.save(dmm.state_dict(), path / "state_dict.pt")
    optimizer.save(path / "opt.pt")


def load_checkpoint(path: Path, dmm: DMM, optimizer: PyroOptim) -> None:
    model_path = path / "state_dict.pt"
    opt_path = path / "opt.pt"
    assert exists(model_path) and exists(opt_path), "Invalid load paths"
    dmm.load_state_dict(torch.load(model_path))
    optimizer.load(opt_path)


def pack_sequences(batch: List[Tensor]) -> Tuple[Tensor, PackedSequence, Tensor, Tensor]:
    """Function to collate a batch into all inputs required for model and guide.
    :batch: List of B D-dimensional sequences of varying length to be collated into a BxTxD batch.
    T will be the length of the longest sequence in the batch. The padded observations will be dealt with using
    a masking tensor. Since the batch will be fed to an RNN, and consumed right-to-left, a reversed and packed copy is also returned. 
    """
    
    def reverse_seqs(seqs: Tensor, seq_lengths: Sequence[int]) -> Tensor:
        reversed_mini_batch = torch.zeros_like(seqs)
        for b in range(seqs.size(0)):
            T = seq_lengths[b]
            time_slice = torch.arange(T - 1, -1, -1, device=seqs.device)
            reversed_sequence = torch.index_select(seqs[b, :, :], 0, time_slice)
            reversed_mini_batch[b, 0:T, :] = reversed_sequence
        return reversed_mini_batch


    def create_seq_mask(seq: Tensor, seq_lengths: Sequence[int]) -> Tensor:
        mask = torch.zeros(seq.shape[0:2])
        for b in range(seq.shape[0]):
            mask[b, 0 : seq_lengths[b]] = torch.ones(seq_lengths[b])
        return mask



    seq_lens = [x.shape[0] for x in batch]
    seq_lens, seq_order = torch.sort(torch.tensor(seq_lens), descending=True)
    seqs = torch.stack(batch, 0)[seq_order][:, : seq_lens[0], :]
    mask = create_seq_mask(seqs, seq_lens.tolist())
    reversed_packed_seqs = pack_padded_sequence(
        reverse_seqs(seqs, seq_lens.tolist()), seq_lens, batch_first=True
    )
    return seqs, reversed_packed_seqs, mask, seq_lens
