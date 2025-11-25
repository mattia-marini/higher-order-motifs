import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
import config as cfg
import hypergraphx as hx


def plot_dist_hyperedges(edges, title):
    os.makedirs(cfg.PLOT_OUT_DIR, exist_ok=True)

    x = []
    for i in edges:
        if len(i) > 1 and len(i) < 20:
            x.append(len(i))

    M = max(x)

    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.title(title)
    plt.xlabel("Size S")
    plt.ylabel("P(S)")

    ax.hist(
        x,
        bins=np.arange(M + 1) + 0.5,
        density=True,
        alpha=0.5,
        histtype="bar",
        ec="black",
    )
    plt.savefig("{}/{}.pdf".format(cfg.PLOT_OUT_DIR, title))


def plot_dist_hyperedges_weights(edges, title, min_size=2, max_size=20):
    """
    edges: dict[tuple[int], int] -- key is hyperedge (tuple of vertices), value is its weight
    """
    # 1. Group weights by edge size
    size2weights = {}
    for edge, weight in edges.items():
        size = len(edge)
        if min_size <= size <= max_size:
            size2weights.setdefault(size, []).append(weight)

    # 2. Plot distributions
    os.makedirs(cfg.PLOT_OUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(
        2, 1, figsize=(5, 10), gridspec_kw={"height_ratios": [1, 1], "hspace": 0.6}
    )
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    ax[0].set_xlabel("w")
    ax[0].set_ylabel("count(w)")

    ax[0].set_title(f"{title}\n")
    ax[0].text(
        0.5,
        1.01,
        "The number of weights by hyperedge size",
        ha="center",
        va="bottom",
        transform=ax[0].transAxes,
        fontsize=8,
        fontstyle="italic",
    )
    ax[0].text(
        0.5,
        -0.2,
        "The graph purposely keeps only the percentile range containing 80% of the weights,\n centered around the expected value, to exclude outliers.",
        ha="center",
        va="center",
        transform=ax[0].transAxes,
        fontsize=8,
    )

    # Calculate the percentile range containing 80% of the weights.
    total_weight_count = np.zeros(max(edges.values()) + 1)
    for w in edges.values():
        total_weight_count[w] += 1

    weight_dist = total_weight_count / len(edges)

    # Exclude outliers by focusing on central 80% around expected value
    expected_value = 0
    for w, p in enumerate(weight_dist):
        expected_value += w * p

    target_count = round(0.8 * len(edges))  # I want to include at least 80% of edges
    current_count = total_weight_count[
        round(expected_value)
    ]  # Start from expected value
    l = round(expected_value) - 1
    r = round(expected_value) + 1

    while current_count < target_count:
        if l >= 0:
            current_count += total_weight_count[l]
            l -= 1
        if r < len(total_weight_count):
            current_count += total_weight_count[r]
            r += 1
    l = max(l, 0)

    max_hyperedge_size = max(size2weights.keys()) + 1
    discarded_values = [list() for _ in range(max_hyperedge_size)]

    for idx, (size, weights) in enumerate(sorted(size2weights.items())):
        if len(weights) == 0:
            continue

        curr_weight_count = np.zeros(r + 1)
        for w in weights:
            try:
                curr_weight_count[w] += 1
            except:
                discarded_values[size].append(w)

        # print(f"{size} - |e|:{len(weights)} e: {expected_value} l:{l} r:{r}")
        ax[0].plot(
            range(l, r),
            curr_weight_count[l:r],
            label=f"size={size}",
            alpha=0.8,
        )

    ax[0].legend(title="Hyperedge size")

    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    ax[1].set_xlabel("hyperedge size S")
    ax[1].set_ylabel("count(outliers)")

    ax[1].set_title(f"Outliers discarded from {title}\n")
    ax[1].text(
        0.5,
        1.01,
        "The number of edges discarded as outliers by hyperedge size",
        ha="center",
        va="bottom",
        transform=ax[1].transAxes,
        fontsize=8,
        fontstyle="italic",
    )

    discarded_values_count = [len(dv) for dv in discarded_values]

    ax[1].bar(
        range(2, max_hyperedge_size),
        discarded_values_count[2:],
        alpha=0.5,
    )

    fig.savefig(f"{cfg.PLOT_OUT_DIR}/{title}_weights_by_size.pdf", bbox_inches="tight")
    plt.close()


def plot_dist_motifs(motifs, title, graphs_per_row=None):
    os.makedirs(cfg.PLOT_OUT_DIR, exist_ok=True)

    if graphs_per_row is None:
        graphs_per_row = len(motifs)

    counts = [c for (_, c) in motifs]
    motifs_reprentations = [m for (m, _) in motifs]

    fig, main_axes = get_bisected_motifs_layout(motifs_reprentations, graphs_per_row)
    main_axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    main_axes.set_title(title)
    main_axes.set_xlabel("Motif id")
    main_axes.set_ylabel("Count")

    main_axes.bar(range(len(counts)), counts, align="center", alpha=0.7)

    fig.tight_layout()
    fig.savefig("{}/{}.pdf".format(cfg.PLOT_OUT_DIR, title))


def plot_leading_motifs(motifs, title, graphs_per_row=None, percentile=1, limit=None):
    """
    Plots the more dominant motifs included in the given percentile

    Args:
        motifs (list[tuple[tuple[int]]]): List of motifs to be plotted
        title (str): Title of the plot
        graphs_per_row (int, optional): Number of motifs per row.
            Defaults to None, meaning all motifs in one row.
        percentile (float, optional): Percentile of motifs to be plotted.
        limit (int, optional): Maximum number of motifs to be plotted.
    """
    os.makedirs(cfg.PLOT_OUT_DIR, exist_ok=True)

    if graphs_per_row is None:
        graphs_per_row = len(motifs)

    sorted_motifs = sorted(motifs, key=lambda x: x[1], reverse=True)
    total_weight = sum(x[1] for x in motifs)
    target_weight = total_weight * percentile

    idx = 0
    curr_sum = 0
    while curr_sum < target_weight and idx < len(sorted_motifs):
        curr_sum += sorted_motifs[idx][1]
        idx += 1
    if limit is not None:
        idx = min(idx, limit)

    motifs_to_plot = [x[0] for x in sorted_motifs[:idx]]
    motifs_counts = [x[1] for x in sorted_motifs[:idx]]
    # print(sorted_motifs[:idx])
    fig, main_axes = get_bisected_motifs_layout(motifs_to_plot, graphs_per_row)
    main_axes.set_title(title)
    main_axes.set_xlabel("Motif id")
    main_axes.set_ylabel("Motif count")
    main_axes.bar(range(idx), motifs_counts, align="center", alpha=0.7)

    fig.tight_layout()
    fig.savefig("{}/{}.pdf".format(cfg.PLOT_OUT_DIR, title))


def get_bisected_motifs_layout(motifs, graphs_per_row):
    """
    Returns a matplotlib figure to achieve a 2 row layout. The first row is
    left empty and returned whereas the second is filled with plots of the
    given motifs

    Args:
        motifs (list[tuple[tuple[int]]]): List of motifs to be plotted

    Returns:
        fig (matplotlib.figure.Figure): The matplotlib figure containing the
            layout
        main_axes (matplotlib.axes.Axes): The empty axes corresponding to the
            first row of the layout
    """
    from hypergraphx.viz import draw_hypergraph

    n = len(motifs)
    m = math.ceil(n / graphs_per_row)
    fig = plt.figure(figsize=(5, 5 + 5 / graphs_per_row * m))
    gs = gridspec.GridSpec(
        1 + m, graphs_per_row, height_ratios=[graphs_per_row] + [1] * m, wspace=0
    )

    main_axes = fig.add_subplot(gs[0, :])
    for i, motif in enumerate(motifs):
        nodes = set()
        for e in motif:
            for v in e:
                nodes.add(v)

        node_size = 50
        hyperedge_alpha = 0.8
        pos = {1: (-1, 0), 2: (1, 0), 3: (0, 1.5)}

        if len(nodes) == 4:
            node_size = 50
            hyperedge_alpha = 0.3
            pos = {1: (-1, -1), 2: (1, -1), 3: (1, 1), 4: (-1, 1)}

        # print(f"Plotting motif {i}")
        ax = fig.add_subplot(gs[1 + math.floor(i / graphs_per_row), i % graphs_per_row])
        ax.set_frame_on(False)
        ax.set_xlabel(str(i), fontsize=8)

        hypergraph = hx.Hypergraph()
        hypergraph.add_edges(list(motif))
        draw_hypergraph(
            hypergraph,
            ax=ax,
            edge_color="black",
            pos=pos,
            node_size=node_size,
            hyperedge_alpha=hyperedge_alpha,
        )

    return fig, main_axes
