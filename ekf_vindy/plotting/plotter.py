# TODO: Add the plot_phase functionality

import re
from matplotlib.colors import to_rgba
from ekf_vindy.plotting import latex_available
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches

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