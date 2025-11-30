# TODO: Add the plot_phase functionality

import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
     
from matplotlib.colors import to_rgba
from pyparsing import col
from ekf_vindy.plotting import latex_available
from typing import List
from sympy import Symbol

"""
This code is quite ugly, and it was mostly made with AI. 
Since it is purely plotting code, I won't bother making it cleaner.
"""

LABEL_SIZE_MAP = {
    "small": 12,
    "medium": 16,
    "large": 22,
}

def sympy_to_latex_label(sym):
    s = str(sym)

    # Replace powers: z_0**2 → z_0^2
    s = re.sub(r"\*\*(\d+)", r"^\1", s)

    # Replace function names with LaTeX
    s = s.replace("sin", r"\sin")
    s = s.replace("cos", r"\cos")
    s = s.replace("exp", r"\exp")
    s = s.replace("log", r"\log")

    # Replace variables starting with b -> \beta
    s = re.sub(r"\bb(_\w+)?\b", r"\\beta", s)

    # Wrap in $...$ for math mode
    s = f"${s}$"

    # Optionally wrap non-math parts in \textrm{} (similar to your format_label)
    if latex_available:
        return s
    else:
        return s.strip("$")

def sympy_list_to_latex(symbols):
    return [sympy_to_latex_label(s) for s in symbols]

def format_label(label):
    """
    Wraps text parts in \textrm{} and preserves math parts in $...$.
    """
    if not latex_available:
        return label

    # Split by math blocks ($...$)
    parts = re.split(r"(\$.*?\$)", label)

    formatted_parts = []
    for part in parts:
        if part.startswith("$") and part.endswith("$"):
            # Math part, leave as-is
            formatted_parts.append(part)
        else:
            # Text part, wrap in \textrm{} if not empty
            if part.strip():
                formatted_parts.append(f"\\textrm{{{part}}}")
    return "".join(formatted_parts)
    
def _generic_labels(components: int):
    """
    Returns the name of the i-th component of the state.
    """
    return [r"$x_{" + f"{i}" + r"}(t)$" for i in range(components)]

def plot_trajectory(x: np.ndarray, time_instants: np.ndarray, sdevs: np.ndarray | None = None,
                    state_names: List[str] | None = None, legend_fontsize: int = 16, title: str = "",
                    x_tick_skip: int = None, ylim: tuple | None = None, 
                    reference: np.ndarray | None = None, palette="muted",
                    xlabel=r"$t$", ylabel=r"$y(t)$"):
    """
    We assume that x is of shape (T, n), where T are the time instances, and n is the number of dimensions.
    """
    # format title depending on LaTeX availability
    title_str = title if not latex_available else format_label(title)

    state_dimension = x.shape[1] 

    # format labels based on LaTeX availability
    if state_names:
        labels = [format_label(label) for label in state_names] if latex_available else state_names
    else:
        labels = _generic_labels(state_dimension)
    
    xlabel = xlabel if not latex_available else format_label(xlabel)
    ylabel = ylabel if not latex_available else format_label(ylabel)

    # set seaborn style
    sns.set_theme(style="whitegrid", palette="muted")

    colors = sns.color_palette(palette, n_colors=state_dimension) 
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # plot each trajectory
    for i in range(state_dimension):
        ax.plot(time_instants, x[:, i], label=labels[i], lw=2, color=colors[i])
        
        # add confidence intervals if provided
        if sdevs is not None:
            upper = x[:, i] + 1.96 * sdevs[:, i]
            lower = x[:, i] - 1.96 * sdevs[:, i]
            fill_color = to_rgba(colors[i], alpha=0.2)
            ax.fill_between(time_instants, lower, upper, color=fill_color)

    # overlay reference trajectory if provided
    if reference is not None:
        ax.plot(
            time_instants, reference,
            color="red", linestyle="--", lw=2.5,
            label="$\mu (t)$"
        )
    
    # titles and labels
    ax.set_title(title_str, fontsize=18, color='black', pad=25)
    ax.set_xlabel(xlabel, fontsize=18, weight='bold', color='black', labelpad=10)
    ax.set_ylabel(ylabel, fontsize=18, weight='bold', color='black', labelpad=10)

    # ticks
    if not x_tick_skip:
        adaptive_skip = np.floor(np.abs(time_instants[-1] - time_instants[0]) / 6)
        ax.set_xticks(np.arange(0, time_instants[-1] + 1, adaptive_skip))
    else: 
        ax.set_xticks(np.arange(0, time_instants[-1] + 1, x_tick_skip))
    ax.tick_params(axis='both', labelsize=16, color='black')
    
    # spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')    
    ax.spines['bottom'].set_color('black')

    if ylim: 
        ax.set_ylim(ylim)

    # grid and background
    ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
    ax.set_axisbelow(True)
    fig.patch.set_facecolor('white')

    # legend (skip if too many states)
    if state_dimension <= 6:
        ax.legend(frameon=True, fontsize=legend_fontsize, framealpha=1.0, 
                  edgecolor='black', fancybox=False)
        
    return fig, ax

def plot_pdf(
    x: np.ndarray,
    pdf_values: np.ndarray,
    batch_labels: list | tuple | None = None,
    dim_labels: list | tuple | None = None,
    ylim: tuple | None = (0, 10),
    palette="muted",
    label_size="medium",    # NEW
):
    """
    Plot PDFs in a grid:
    - Rows = batch labels (variables)
    - Columns = dim labels (library terms)
    """

    fontsize = LABEL_SIZE_MAP.get(label_size, LABEL_SIZE_MAP["medium"])
    tick_fontsize = max(fontsize - 4, 8)

    # --- Ensure (B, D, N) shape ---
    if x.ndim == 1:
        x = x[None, None, :]
        pdf_values = pdf_values[None, None, :]
    elif x.ndim == 2:
        x = x[None, :, :]
        pdf_values = pdf_values[None, :, :]

    B, D, N = pdf_values.shape

    # --- Format labels ---
    if batch_labels is not None:
        batch_labels_fmt = [
            sympy_to_latex_label(b) if isinstance(b, Symbol) else format_label(str(b))
            for b in batch_labels
        ]
    else:
        batch_labels_fmt = [format_label(f"batch {b}") for b in range(B)]

    if dim_labels is not None:
        dim_labels_fmt = [
            sympy_to_latex_label(d) if isinstance(d, Symbol) else format_label(str(d))
            for d in dim_labels
        ]
    else:
        dim_labels_fmt = [format_label(f"dim {d}") for d in range(D)]

    # --- Plot theme ---
    sns.set_theme(style="whitegrid", palette=palette)
    colors = sns.color_palette(palette, n_colors=D)

    # --- Create subplot grid ---
    fig, axes = plt.subplots(
        B, D,
        figsize=(4 * D, 3 * B),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for b in range(B):
        for d in range(D):

            ax = axes[b, d]
            color = colors[d % len(colors)]

            ax.plot(x[b, d, :], pdf_values[b, d, :], lw=2, color=color)

            ax.fill_between(
                x[b, d, :], 0, pdf_values[b, d, :],
                color=to_rgba(color, alpha=0.25)
            )

            # column titles
            ax.set_title(dim_labels_fmt[d], fontsize=fontsize, pad=6)

            # row labels (left column only)
            if d == 0:
                ax.set_ylabel(
                    batch_labels_fmt[b],
                    fontsize=fontsize,
                    rotation=0,
                    labelpad=35,
                    va="center",
                )

            ax.set_ylim(ylim)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("black")
            ax.spines["left"].set_color("black")

            ax.tick_params(axis="both", labelsize=tick_fontsize)
            ax.grid(True, linestyle="-", linewidth=0.5, color="gray", alpha=0.5)
            ax.set_axisbelow(True)

    fig.tight_layout()
    return fig, axes