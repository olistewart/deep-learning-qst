"""
Small shared matplotlib helpers so every experiment script produces
consistently styled figures without repeating boilerplate.
"""

from pathlib import Path

import matplotlib.pyplot as plt


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.autolayout": True,
        }
    )


def strip_spines(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def savefig(fig, path) -> None:
    """Save a figure, creating the parent directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
