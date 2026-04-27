"""Plotting helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt


def save_current_figure(path, dpi: int = 300) -> None:
    """Save and close the current matplotlib figure."""

    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

