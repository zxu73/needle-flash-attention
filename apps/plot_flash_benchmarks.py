"""Figures for the FlashAttention benchmarks.

Run from the project root, on the files written by the two benchmark scripts:
    python apps/plot_flash_benchmarks.py --attention results/attention_cuda.json \
        --transformer results/transformer_cuda.json --out-dir results/figures

Writes attention_time.png and attention_memory.png (operator benchmark) and
transformer_step.png and transformer_loss.png (model benchmark).  The numbers behind every
figure are the tables the benchmark scripts print.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullLocator

# chart chrome
SURFACE, INK, INK_SECONDARY, GRID, AXIS = "#ffffff", "#191919", "#666666", "#e7e7e7", "#c2c2c2"
# one color per implementation, the same in every figure (checked for colour-blind separation)
COLORS = {"flash": "#257de0", "naive": "#f47230", "torch": "#00b894"}
SHORT_LABELS = {"flash": "flash", "naive": "standard", "torch": "PyTorch"}
ORDER = ["flash", "naive", "torch"]
ZORDER = {"naive": 3, "torch": 4, "flash": 5}  # ours is drawn last, on top


def marker_style(impl):
    """Filled dots, except rings for PyTorch: where another series lies exactly on top of it
    (flash_attention's memory does) the dot shows inside the ring instead of hiding it."""
    if impl == "torch":
        return dict(marker="o", markersize=9.5, markerfacecolor=SURFACE,
                    markeredgecolor=COLORS[impl], markeredgewidth=2)
    return dict(marker="o", markersize=6.5, markerfacecolor=COLORS[impl],
                markeredgecolor=SURFACE, markeredgewidth=1.5)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def load(path):
    with open(path) as f:
        return json.load(f)


def plain_number(value, _=None):
    """Tick / value text without scientific notation: 0.05, 1.5, 12, 1,024."""
    if value >= 100:
        return f"{value:,.0f}"
    return f"{value:.3g}"


def style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3, width=0.8, color=AXIS)


def log_axes(ax, x_values):
    """Log-log axes with a tick at every measured sequence length, written out in full."""
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(x_values))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(plain_number))
    ax.yaxis.set_minor_locator(NullLocator())


def refine_log_ticks(ax):
    """Call after plotting: a y range of under two decades gets 1-2-5 ticks, not one label."""
    low, high = ax.get_ylim()
    if high / low < 100:
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))


def draw_series(ax, xs, ys, impl):
    ax.plot(xs, ys, color=COLORS[impl], linewidth=2, solid_capstyle="round",
            solid_joinstyle="round", clip_on=False, zorder=ZORDER[impl], **marker_style(impl))


def label_line_ends(ax, ends, unit):
    """Name and final value next to the last point of each line: [(x, y, impl), ...].

    Labels that would overlap are stacked, centred on the points they belong to; a label
    that ends up far from its point is tied back to it by a leader line.
    """
    fig = ax.figure
    fig.canvas.draw()  # settle the limits so data -> pixel positions are final
    to_points = 72.0 / fig.dpi
    min_gap = 13.0 / to_points  # pixels between label centres

    points = sorted(((*ax.transData.transform((x, y)), x, y, impl) for x, y, impl in ends),
                    key=lambda p: p[1])
    clusters = []  # runs of labels that start at about the same x and are too close in y
    for point in points:
        last = clusters[-1][-1] if clusters else None
        if last and abs(last[0] - point[0]) < 110 and point[1] - last[1] < min_gap:
            clusters[-1].append(point)
        else:
            clusters.append([point])

    for cluster in clusters:
        centre = sum(p[1] for p in cluster) / len(cluster)
        for rank, (px, py, x, y, impl) in enumerate(cluster):
            label_y = centre + (rank - (len(cluster) - 1) / 2) * min_gap
            far = abs(label_y - py) > min_gap
            ax.annotate(
                f"{SHORT_LABELS[impl]}  {plain_number(y)} {unit}", xy=(x, y),
                xytext=(11 + (6 if far else 0), (label_y - py) * to_points),
                textcoords="offset points", va="center", ha="left", fontsize=9, color=INK,
                annotation_clip=False, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5),
                arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.8, shrinkA=0, shrinkB=4)
                if far else None)


def add_header(fig, title, subtitle, impls, labels, top=0.97):
    """Figure title, a one-line description, and the legend, stacked at the top left."""
    left = fig.subplotpars.left
    fig.text(left, top, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(left, top - 0.075, subtitle, fontsize=9.5, color=INK_SECONDARY, va="top")
    handles = [plt.Line2D([], [], color=COLORS[i], linewidth=2, **marker_style(i))
               for i in impls]
    fig.legend(handles, [labels[i] for i in impls], loc="upper left",
               bbox_to_anchor=(left - 0.012, top - 0.125), ncol=len(impls), frameon=False,
               fontsize=9.5, labelcolor=INK_SECONDARY, handlelength=2.2, columnspacing=2.0)


def stop_note(records, impls, labels, budget_gb, panel_key=None, panel_names=None):
    """Why a line ends before the others, if it does (per panel when panels differ)."""
    notes = []
    for impl in impls:
        failed = [r for r in records if r["impl"] == impl and r["status"] != "ok"]
        if not failed:
            continue
        panels = sorted({r[panel_key] for r in failed}) if panel_key else [None]
        stops = []
        for panel in panels:
            first = min((r for r in failed if panel is None or r[panel_key] == panel),
                        key=lambda r: r["N"])
            where = f" ({panel_names[panel].lower()})" if panel is not None else ""
            stops.append(f"N = {first['N']:,}{where}")
        skipped = all(r["status"] == "skipped" for r in failed)
        reason = (f"estimated peak memory above the {budget_gb:g} GB budget"
                  if skipped else "out of memory, or estimated peak memory above the budget")
        notes.append(f"{labels[impl]} is not run from {' and '.join(stops)}: {reason}.")
    return notes


def add_footnotes(fig, notes):
    if notes:
        fig.text(fig.subplotpars.left, 0.025, "\n".join(notes), fontsize=8.5,
                 color=INK_SECONDARY, va="bottom", linespacing=1.5)


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def attention_figure(data, key, unit, title, ylabel, out_dir, name):
    """One panel per mode of the operator benchmark; `key` is the plotted record field."""
    meta, records = data["meta"], data["records"]
    labels = meta["labels"]
    mode_names = {"forward": "Forward", "forward+backward": "Forward + backward"}
    modes = [m for m in mode_names if any(r["mode"] == m for r in records)]
    plotted = lambda r: r["status"] == "ok" and r.get(key) is not None
    impls = [i for i in ORDER if any(r["impl"] == i and plotted(r) for r in records)]
    lens = sorted({r["N"] for r in records})

    # the panels share the y axis, so the two modes can be compared directly
    fig, axes = plt.subplots(1, len(modes), figsize=(5.6 * len(modes), 5.1), squeeze=False,
                             sharey=True)
    fig.subplots_adjust(left=0.075, right=0.87, top=0.72, bottom=0.24, wspace=0.52)
    panel_ends = []
    for ax, mode in zip(axes[0], modes):
        style_axes(ax, mode_names[mode], "Sequence length N", ylabel)
        log_axes(ax, lens)
        ax.tick_params(labelleft=True)
        ends = []
        for impl in impls:
            points = sorted((r["N"], r[key]) for r in records
                            if r["impl"] == impl and r["mode"] == mode and plotted(r))
            if points:
                xs, ys = zip(*points)
                draw_series(ax, xs, ys, impl)
                ends.append((xs[-1], ys[-1], impl))
        panel_ends.append((ax, ends))
    for ax, ends in panel_ends:  # after all panels are drawn: the shared limits are final
        refine_log_ticks(ax)
        label_line_ends(ax, ends, unit)

    mask = "causal mask" if meta["causal"] else "no mask"
    runs = f"median of {meta['repeats']} runs, " if key == "time_ms" else ""
    subtitle = (f"{meta['hardware']} ({meta['device']})  ·  B = {meta['B']}, H = {meta['H']}, "
                f"D = {meta['D']}, {mask}  ·  {runs}log axes")
    add_header(fig, title, subtitle, impls, labels)
    notes = stop_note([r for r in records if r["impl"] in impls], impls, labels,
                      meta["budget_gb"], panel_key="mode", panel_names=mode_names)
    if key == "peak_mb" and "torch" in labels and "torch" not in impls:
        notes.append("PyTorch's peak memory is only measured on cuda.")
    add_footnotes(fig, notes)
    save(fig, out_dir, name)


def transformer_step_figure(data, out_dir):
    """Time and peak memory of one training step against the sequence length."""
    meta = data["meta"]
    records = [r for r in data["records"] if r["experiment"] == "step"]
    if not records:
        return
    labels = meta["labels"]
    impls = [i for i in ORDER if any(r["impl"] == i and r["status"] == "ok" for r in records)]
    lens = sorted({r["N"] for r in records})

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.1))
    fig.subplots_adjust(left=0.075, right=0.87, top=0.72, bottom=0.24, wspace=0.52)
    panels = [("time_ms", "ms", "Time per training step", "Time (ms)"),
              ("peak_mb", "MB", "Peak memory of a training step", "Peak memory (MB)")]
    for ax, (key, unit, title, ylabel) in zip(axes, panels):
        style_axes(ax, title, "Sequence length N", ylabel)
        log_axes(ax, lens)
        ends = []
        for impl in impls:
            points = sorted((r["N"], r[key]) for r in records
                            if r["impl"] == impl and r["status"] == "ok")
            xs, ys = zip(*points)
            draw_series(ax, xs, ys, impl)
            ends.append((xs[-1], ys[-1], impl))
        refine_log_ticks(ax)
        label_line_ends(ax, ends, unit)

    subtitle = (f"{meta['hardware']} ({meta['device']})  ·  {meta['layers']} layers, "
                f"{meta['num_head']} heads x {meta['dim_head']}, batch {meta['batch']}  ·  "
                f"forward, loss, backward and Adam update, log axes")
    add_header(fig, "Training a Transformer language model", subtitle, impls, labels)
    add_footnotes(fig, stop_note(records, impls, labels, meta["budget_gb"]))
    save(fig, out_dir, "transformer_step.png")


def transformer_loss_figure(data, out_dir):
    """Loss curves on Penn Treebank; the two variants are expected to coincide."""
    meta = data["meta"]
    runs = {r["impl"]: r for r in data["records"]
            if r["experiment"] == "training" and r["status"] == "ok"}
    if not runs:
        return
    labels = meta["labels"]
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    fig.subplots_adjust(left=0.1, right=0.95, top=0.66, bottom=0.13)
    style_axes(ax, "", "Optimizer step", "Training loss")

    # the curves lie on top of each other: a line for one, dots for the other
    if "naive" in runs:
        losses = runs["naive"]["losses"]
        ax.plot(range(len(losses)), losses, color=COLORS["naive"], linewidth=2,
                solid_capstyle="round", solid_joinstyle="round", zorder=2)
    if "flash" in runs:
        losses = runs["flash"]["losses"]
        steps = np.arange(len(losses))
        every = max(1, len(losses) // 25)
        ax.plot(steps[::every], np.array(losses)[::every], linestyle="none", marker="o",
                markersize=6.5, color=COLORS["flash"], markeredgecolor=SURFACE,
                markeredgewidth=1.5, zorder=3, clip_on=False)
    ax.set_xlim(left=0)

    any_run = next(iter(runs.values()))
    subtitle = (f"Penn Treebank, vocabulary {any_run['vocab']:,}, sequences of {any_run['N']} "
                f"tokens, batch {meta['batch']}  ·  {meta['hardware']} ({meta['device']})")
    top, left = 0.97, fig.subplotpars.left
    fig.text(left, top, "Same loss curve with and without flash attention", fontsize=13,
             fontweight="bold", color=INK, va="top")
    fig.text(left, top - 0.075, subtitle, fontsize=9.5, color=INK_SECONDARY, va="top")
    handles, names = [], []
    if "naive" in runs:
        handles.append(plt.Line2D([], [], color=COLORS["naive"], linewidth=2))
        names.append(f"{labels['naive']}  ({runs['naive']['time_ms']:.0f} ms / step)")
    if "flash" in runs:
        handles.append(plt.Line2D([], [], linestyle="none", marker="o", markersize=6.5,
                                  color=COLORS["flash"], markeredgecolor=SURFACE,
                                  markeredgewidth=1.5))
        names.append(f"{labels['flash']}  ({runs['flash']['time_ms']:.0f} ms / step)")
    fig.legend(handles, names, loc="upper left", bbox_to_anchor=(left - 0.015, top - 0.125),
               ncol=1, frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY, labelspacing=0.5)
    if len(runs) == 2:
        gap = np.abs(np.array(runs["flash"]["losses"]) - np.array(runs["naive"]["losses"])).max()
        ax.annotate(f"largest difference between the curves: {gap:.1e}", xy=(1, 1),
                    xycoords="axes fraction", ha="right", va="top", fontsize=9,
                    color=INK_SECONDARY)
    save(fig, out_dir, "transformer_loss.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--attention", help="JSON written by apps/bench_flash_attention.py")
    parser.add_argument("--transformer", help="JSON written by apps/bench_flash_transformer.py")
    parser.add_argument("--out-dir", default="results/figures")
    args = parser.parse_args()
    if not (args.attention or args.transformer):
        parser.error("give --attention and/or --transformer")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.attention:
        data = load(args.attention)
        attention_figure(data, "time_ms", "ms", "Attention: time per call", "Time (ms)",
                         args.out_dir, "attention_time.png")
        attention_figure(data, "peak_mb", "MB", "Attention: peak memory on top of the inputs",
                         "Peak memory (MB)", args.out_dir, "attention_memory.png")
    if args.transformer:
        data = load(args.transformer)
        transformer_step_figure(data, args.out_dir)
        transformer_loss_figure(data, args.out_dir)


if __name__ == "__main__":
    main()
