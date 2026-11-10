# -*- coding: utf-8 -*-
"""Plot the retained total-loss history of a parametric ensemble checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rbf_fd_full_order.visualization import apply_journal_style, save_figure, style_axis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results") / "parametric_many_query_hollow_5000_best" / "hollow" / "parametric_operator.pt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results") / "parametric_many_query_hollow_5000_best" / "hollow" / "training_loss_curve.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    histories = payload["members"]
    if not histories or not all(member.get("history") for member in histories):
        raise RuntimeError("The checkpoint does not contain retained loss histories.")

    epochs = np.asarray([item["epoch"] for item in histories[0]["history"]], dtype=float)
    losses = np.asarray(
        [[item["total_loss"] for item in member["history"]] for member in histories],
        dtype=float,
    )
    if losses.shape[1] != epochs.size:
        raise RuntimeError("Ensemble members have incompatible loss-history lengths.")

    apply_journal_style()
    figure, axis = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
    for index, member_loss in enumerate(losses):
        axis.semilogy(
            epochs,
            member_loss,
            color=colors[index % len(colors)],
            linewidth=0.9,
            alpha=0.78,
            label=f"member {index + 1}",
        )
        best_epoch = float(histories[index].get("best_epoch", epochs[np.argmin(member_loss)]))
        best_loss = float(histories[index].get("best_total_loss", np.min(member_loss)))
        axis.plot(
            best_epoch,
            best_loss,
            marker="o",
            markersize=3.2,
            markerfacecolor="white",
            markeredgecolor=colors[index % len(colors)],
            markeredgewidth=0.8,
        )

    axis.semilogy(
        epochs,
        np.mean(losses, axis=0),
        color="#000000",
        linewidth=1.5,
        label="ensemble mean",
    )
    axis.set_xlabel("epoch")
    axis.set_ylabel("total loss")
    axis.set_xlim(float(epochs[0]), float(epochs[-1]))
    axis.grid(True, which="major", linestyle=":", linewidth=0.45, alpha=0.55)
    style_axis(axis)
    axis.legend(loc="best", ncol=2, frameon=False)
    save_figure(figure, args.out)
    plt.close(figure)
    print(f"saved: {args.out.with_suffix('.png')}")
    print(f"saved: {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
