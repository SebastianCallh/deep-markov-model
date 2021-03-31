from argparse import ArgumentParser, Namespace
from itertools import starmap
from typing import Callable, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pyro.optim import ClippedAdam
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
import pyro

from dmm.models.dmm import DMM, pack_sequences


def parse_args() -> Namespace:
    parser = ArgumentParser(description="DMM arguments")
    parser.add_argument("--latent-dim", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=100)
    parser.add_argument("--emit-hidden-dim", type=int, default=100)
    parser.add_argument("--transition-hidden-dim", type=int, default=100)
    parser.add_argument("--encode-hidden-dim", type=int, default=100)
    parser.add_argument("--encode-num-layers", type=int, default=1)
    parser.add_argument("--encode-dropout-rate", type=float, default=0.1)

    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.0008)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.9999)
    parser.add_argument("--clip-norm", type=float, default=10.0)
    parser.add_argument("--learning-rate-decay", type=float, default=0.9995)
    parser.add_argument("--weight-decay", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--anneal-steps", type=int, default=50)
    parser.add_argument("--anneal-start-val", type=float, default=0.2)
    parser.add_argument("--anneal-end-val", type=float, default=1.0)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--jit", action="store_true")

    return parser.parse_args()


def linear_annealing(
    start_val: float, end_val: float, steps: int
) -> Callable[[int], float]:
    """Returns a function which starts at x_start,
    increasing linearly to x_end over steps.

    :start_val: Function start value.
    :end_val: Function end value.
    :steps: The number of time steps to linearly interpolate between xmin and xmas.

    :return: Callable function which implements the annealing strategy.
    """
    return lambda i: start_val + (end_val - start_val) * (i / steps)


def train(
    dmm: nn.Module,
    svi: SVI,
    data_loader: DataLoader,
    annealing: Callable[[int], float],
    device: str,
) -> float:
    dmm.train()

    def loss(i: int, batch) -> float:
        return svi.step(*batch, annealing=annealing(i))

    return sum(starmap(loss, enumerate(data_loader))) / len(data_loader.dataset)


@torch.no_grad()
def evaluate(dmm: nn.Module, svi: SVI, data_loader: DataLoader) -> float:
    dmm.eval()
    return sum(svi.evaluate_loss(x) for x in data_loader) / len(data_loader.dataset)


@torch.no_grad()
def rmse(dmm: DMM, data_loader: DataLoader, num_seen: Optional[int] = None) -> float:
    def batch_rmse(x) -> torch.Tensor:
        B, T, _ = x.shape
        x_seen = x[:, :num_seen, :]
        x_unseen = x[:, num_seen:, :]
        x_pred = dmm.forecast(x=x_seen, num_steps=T - num_seen)
        return F.mse_loss(x_pred, x_unseen)

    return torch.sqrt(sum(map(batch_rmse, data_loader))).item() / len(data_loader)


def toy_data(batch_size):
    B, T, D = 128, 25, 1
    t = torch.linspace(0, 1, T)
    x = 2.5 * t.expand(B, T) + 0.1 * torch.randn(B, T).to(device) + 1

    x = 5 - x ** 2
    x = (x.T - x.T.mean(0)) / x.T.std(0)
    x = x.T.unsqueeze(-1)

    return DataLoader(
        x,
        batch_size=batch_size,
        collate_fn=pack_sequences,
        shuffle=True,
    )


if __name__ == "__main__":
    pyro.enable_validation(True)
    pyro.clear_param_store()
    pyro.set_rng_seed(12348)
    args = parse_args()
    device = "cuda" if args.cuda and torch.has_cuda else "cpu"
    data_loader = toy_data(args.batch_size)

    dmm = DMM(
        data_dim=1,
        latent_dim=args.latent_dim,
        encode_hidden_dim=args.encode_hidden_dim,
        encode_num_layers=args.encode_num_layers,
        encode_dropout_rate=args.encode_dropout_rate,
        transition_hidden_dim=args.transition_hidden_dim,
        emit_hidden_dim=args.emit_hidden_dim,
    ).to(device)

    if args.jit:
        dmm = torch.jit.script(dmm)

    optimizer = ClippedAdam(
        {
            "lr": args.learning_rate,
            "betas": (args.beta1, args.beta2),
            "clip_norm": args.clip_norm,
            "lrd": args.learning_rate_decay,
            "weight_decay": args.weight_decay,
        }
    )

    svi = SVI(
        dmm.model, dmm.guide, optimizer, JitTrace_ELBO() if args.jit else Trace_ELBO()
    )

    annealing = linear_annealing(
        start_val=args.anneal_start_val,
        end_val=args.anneal_end_val,
        steps=args.anneal_steps,
    )

    seen_test_points = 4
    losses = []
    iterator = tqdm(range(args.num_epochs))
    for epoch in iterator:
        losses.append(train(dmm, svi, data_loader, annealing, device))
        iterator.set_postfix_str(f"Train loss {losses[-1]:.3f}")

    x = data_loader.dataset
    B, T, D = x.shape
    x_seen = x[:, :seen_test_points, :] if seen_test_points > 0 else None
    unseen_test_points = T - seen_test_points
    t = torch.linspace(0, 1, T)
    x_hat = dmm.sample(num_steps=unseen_test_points, x=x_seen)
    t_hat = t[seen_test_points:]

    # boi = dmm.latents(x_seen)
    # b = boi["z_1"]['fn'].mean
    # print(b)
    def plot_loss(ax, losses):
        ax.plot(losses)
        ax.set_title("Training loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("-ELBO")

    def plot_pred(ax, t, x, t_pred, x_pred):
        def plot_data(t, x, ax, **kwargs):
            for i in range(x.size(0)):
                ax.plot(t.numpy(), x[i, :].numpy(), **kwargs)

        x_mean = x[:, :, 0].mean(0)
        x_std = x[:, :, 0].std(0)
        ax.plot(t, x_mean, label="Empirical dist mean")
        ax.fill_between(
            t,
            x_mean + 2 * x_std,
            x_mean - 2 * x_std,
            label="Empirical dist 2std",
            alpha=0.6,
            color="blue",
        )

        x_pred_mean = x_pred[:, :, 0].mean(0)
        x_pred_std = x_pred[:, :, 0].std(0)
        ax.plot(t_pred, x_pred_mean, label="Predictive dist mean")
        ax.fill_between(
            t_pred,
            x_pred_mean + 2 * x_pred_std,
            x_pred_mean - 2 * x_pred_std,
            label="Predictive dist 2std",
            alpha=0.6,
            color="green",
        )
        plot_data(t, x[:, :50, 0], ax=ax)
        plot_data(
            t_pred,
            x_pred[:, :50, 0],
            ax=ax,
            linestyle="--",
            color="green",
        )
        ax.legend()

    fig, (pred_ax, loss_ax) = plt.subplots(2, 1, figsize=(8, 12))
    plot_pred(pred_ax, t, x, t_hat, x_hat)
    plot_loss(loss_ax, losses)
    fig.savefig("result.png")
