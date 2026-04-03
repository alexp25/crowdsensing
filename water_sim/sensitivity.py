"""
sensitivity.py
Sensitivity matrix: average network load per agent across adoption
rates and intervention types.

Scenario definition
-------------------
X-axis (columns) â€” Adoption rate: 5 levels
    20%, 40%, 60%, 80%, 100%
    Represents the fraction of agents in the mixed network that actually
    have the intervention installed / active. Models real-world rollout
    gradients where not all households participate at once.

Y-axis (rows) â€” Intervention type: 5 categories
    fixture      : Low-flow Toilet + Shower Reducer (combined)
    appliance    : Efficient Dishwasher
    digital      : Smart Meter + Leak Detection (combined)
    infrastructure: Rainwater Tank + Pressure Limiter (combined)
    behaviour    : Water Education Program

    Each row activates all interventions within that category together,
    giving a category-level view rather than per-item clutter.
    A baseline row (no interventions, adoption irrelevant) anchors the top.

Cell value â€” Average daily load per agent (L/day)
    Averaged over the mixed network (30 family + 40 working + 10 workplace)
    AND over N_SEEDS independent seeds to smooth stochastic adoption draws.

A secondary Î” annotation shows the reduction vs the baseline row.

Why this scenario?
    Policy decisions rarely select individual interventions â€” they select
    programmes (e.g. "a fixture upgrade scheme" or "a digital monitoring
    rollout"). Adoption rate is the key policy lever: you can mandate a
    technology but not guarantee 100% uptake. This matrix answers:
    "For each programme category, how much does partial uptake erode the
    benefit?" â€” directly informing the cost-benefit threshold where a
    rollout becomes worth funding.

Total model runs:
    (5 adoption rates Ã— 5 categories + 1 baseline) Ã— N_SEEDS seeds
    = 6 Ã— 5 Ã— N_SEEDS = 150 runs (at N_SEEDS=5).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from model import WaterNetworkModel
from agents import INTERVENTIONS

# ---------------------------------------------------------------------------
# Scenario constants
# ---------------------------------------------------------------------------

N_SEEDS = 5       # runs averaged per cell to smooth stochastic adoption

# Fixed mixed network â€” same for all cells
NETWORK = dict(n_families=30, n_working=40, n_workplaces=10)
N_AGENTS = sum(NETWORK.values())  # 80

ADOPTION_RATES  = [0.20, 0.40, 0.60, 0.80, 1.00]
ADOPTION_LABELS = ["20%", "40%", "60%", "80%", "100%"]

# Intervention categories â€” each row activates all IDs in its group.
# "baseline" is handled separately (no interventions); the remaining 4
# are passed as category rows in run_sensitivity().
CATEGORIES = {
    "fixtures":   ["efficient_toilet", "shower_reducer"],
    "appliances": ["dishwasher", "rainwater_tank"],
    "behaviour":  ["education_program", "smart_meter", "leak_detection"],
    "combined":   list(INTERVENTIONS.keys()),   # all 8
}
CATEGORY_LABELS = {
    "fixtures":   "Fixtures\n(toilet + shower)",
    "appliances": "Appliances\n(dishwasher + rainwater)",
    "behaviour":  "Behaviour\n(education + digital)",
    "combined":   "Combined\n(all interventions)",
}
CATEGORY_COLORS = {
    "fixtures":   "#38bdf8",
    "appliances": "#a78bfa",
    "behaviour":  "#fb923c",
    "combined":   "#4ade80",
}

FIG_BG = "#ffffff"
AX_BG  = "#ffffff"
TEXT   = "#111111"
GRID   = "#dddddd"

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def avg_daily_load(interventions, adoption_rate, seeds):
    """Mean daily load per agent over multiple seeds."""
    loads = []
    for seed in seeds:
        m = WaterNetworkModel(
            **NETWORK,
            interventions=interventions,
            adoption_rate=adoption_rate,
            seed=seed,
        )
        loads.append(m.total_daily_intervention() / N_AGENTS)
    return float(np.mean(loads))


# ---------------------------------------------------------------------------
# Run sensitivity
# ---------------------------------------------------------------------------

def run_sensitivity():
    seeds = list(range(N_SEEDS))
    cat_keys = list(CATEGORIES.keys())   # 5 categories

    n_rows = len(cat_keys) + 1           # +1 for baseline
    n_cols = len(ADOPTION_RATES)         # 5

    load_matrix  = np.zeros((n_rows, n_cols))
    delta_matrix = np.zeros((n_rows, n_cols))

    total_runs = n_rows * n_cols * N_SEEDS
    print(f"Running sensitivity matrix "
          f"({n_rows} rows Ã— {n_cols} adoption levels Ã— {N_SEEDS} seeds "
          f"= {total_runs} model runs)...\n")

    # --- Baseline row (row 0) â€” no interventions, adoption rate irrelevant ---
    baseline_load = avg_daily_load([], adoption_rate=0.0, seeds=seeds)
    for j in range(n_cols):
        load_matrix[0, j]  = baseline_load
        delta_matrix[0, j] = 0.0
    print(f"  [baseline]  load={baseline_load:.1f} L/agent/day\n")

    # --- Category rows ---
    for i, cat in enumerate(cat_keys, start=1):
        inv_ids = CATEGORIES[cat]
        for j, rate in enumerate(ADOPTION_RATES):
            load = avg_daily_load(inv_ids, adoption_rate=rate, seeds=seeds)
            load_matrix[i, j]  = load
            delta_matrix[i, j] = load - baseline_load
            print(f"  [{CATEGORY_LABELS[cat].split(chr(10))[0]:<22}]"
                  f"  adoption={ADOPTION_LABELS[j]}  "
                  f"load={load:6.1f} L/agent  "
                  f"Î”={delta_matrix[i,j]:+.1f}")
        print()

    return load_matrix, delta_matrix, cat_keys


# ---------------------------------------------------------------------------
# Plot: main heatmap + marginal line charts
# ---------------------------------------------------------------------------

def plot_sensitivity(load_matrix, delta_matrix, cat_keys, out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)

    row_labels = ["Baseline\n(no intervention)"] + [
        CATEGORY_LABELS[c] for c in cat_keys
    ]
    row_colors = ["#64748b"] + [CATEGORY_COLORS[c] for c in cat_keys]

    # Heatmap figure (standalone)
    fig = plt.figure(figsize=(12, 7))
    fig.patch.set_facecolor(FIG_BG)
    ax_heat = fig.add_subplot(111)

    # fig.suptitle(
    #     "Sensitivity Matrix - Intervention Category x Adoption Rate\n"
    #     "Cell: average daily load per agent (L/day) and (delta vs baseline)",
    #     color=TEXT, fontsize=13, fontweight="bold", y=1.02,
    # )
    
    fig.suptitle(
        "Network Load Sensitivity Matrix",
        color=TEXT, fontsize=13, fontweight="bold", y=1.02,
    )

    ax_heat.set_facecolor(AX_BG)

    cmap = plt.get_cmap("YlOrRd")
    vmin, vmax = load_matrix.min(), load_matrix.max()
    im = ax_heat.imshow(load_matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    for i in range(load_matrix.shape[0]):
        for j in range(load_matrix.shape[1]):
            val = load_matrix[i, j]
            dval = delta_matrix[i, j]
            norm = (val - vmin) / (vmax - vmin + 1e-9)
            tc = "#0f172a" if 0.25 < norm < 0.80 else TEXT

            ax_heat.text(j, i - 0.15, f"{val:.1f}",
                         ha="center", va="center",
                         fontsize=10, fontweight="bold", color=tc)
            if i > 0:
                dc = "#166534" if norm < 0.55 else "#0f172a"
                ax_heat.text(j, i + 0.22, f"({dval:+.1f})",
                             ha="center", va="center",
                             fontsize=8, color=dc)

    ax_heat.set_xticks(range(len(ADOPTION_LABELS)))
    ax_heat.set_xticklabels(ADOPTION_LABELS, color=TEXT, fontsize=11)
    ax_heat.set_xlabel("Adoption rate ->", color=TEXT, fontsize=11, labelpad=8)

    ax_heat.set_yticks(range(len(row_labels)))
    ax_heat.set_yticklabels(row_labels, color=TEXT, fontsize=9)
    for tick, color in zip(ax_heat.get_yticklabels(), row_colors):
        tick.set_color(color)

    ax_heat.set_title("Average daily load per agent (L/day)",
                      color=TEXT, fontsize=10, pad=10)
    ax_heat.tick_params(colors=TEXT, length=0)
    for spine in ax_heat.spines.values():
        spine.set_edgecolor(GRID)

    for x in np.arange(-0.5, len(ADOPTION_RATES), 1):
        ax_heat.axvline(x, color=GRID, linewidth=1.0)
    for y in np.arange(-0.5, load_matrix.shape[0], 1):
        ax_heat.axhline(y, color=GRID, linewidth=1.0)

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=TEXT, labelcolor=TEXT)
    cbar.set_label("L/day per agent", color=TEXT, fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "sensitivity_matrix.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=140, facecolor=FIG_BG)
    plt.close(fig)
    print(f"  Saved -> {out_path}")

    # Trend line figure (standalone)
    fig_line, ax_line = plt.subplots(figsize=(9, 6))
    fig_line.patch.set_facecolor(FIG_BG)
    ax_line.set_facecolor(AX_BG)
    ax_line.tick_params(colors=TEXT)
    for spine in ax_line.spines.values():
        spine.set_edgecolor(GRID)
    ax_line.grid(color=GRID, linestyle="-", linewidth=0.8)

    x = np.array(ADOPTION_RATES) * 100
    ax_line.axhline(load_matrix[0, 0], color="#64748b", linewidth=1.5,
                    linestyle="--", label="Baseline", zorder=1)

    for i, cat in enumerate(cat_keys, start=1):
        color = CATEGORY_COLORS[cat]
        label = CATEGORY_LABELS[cat].replace("\n", " ")
        ax_line.plot(x, load_matrix[i, :],
                     marker="o", markersize=6, linewidth=2,
                     color=color, label=label, zorder=2)
        ax_line.annotate(f"{load_matrix[i, -1]:.0f}",
                         xy=(100, load_matrix[i, -1]),
                         xytext=(4, 0), textcoords="offset points",
                         color=color, fontsize=8, va="center")

    ax_line.set_xlabel("Adoption rate (%)", color=TEXT, fontsize=10)
    ax_line.set_ylabel("Avg load (L/day per agent)", color=TEXT, fontsize=10)
    ax_line.set_title("Load vs Adoption Rate per Category",
                      color=TEXT, fontsize=10, pad=8)
    ax_line.set_xticks([20, 40, 60, 80, 100])
    ax_line.set_xticklabels(["20%", "40%", "60%", "80%", "100%"],
                            color=TEXT, fontsize=9)
    ax_line.legend(facecolor=AX_BG, edgecolor=GRID,
                   labelcolor=TEXT, fontsize=7.5, loc="upper right")

    plt.tight_layout()
    out_line_path = os.path.join(out_dir, "sensitivity_matrix_trend.png")
    fig_line.savefig(out_line_path, bbox_inches="tight", dpi=140, facecolor=FIG_BG)
    plt.close(fig_line)
    print(f"  Saved -> {out_line_path}")

def run_pressure_sensitivity():
    """
    Sensitivity matrix for mean pressure deviation [bar].

    Same axes as the load matrix:
      rows : baseline + 4 intervention categories
      cols : 5 adoption rates (20%..100%)

    Cell value: mean pressure deviation from nominal [bar], averaged
    over all demand nodes and all 24 hours, and over N_SEEDS seeds.

    Negative = pressure drop (demand-driven head loss increases when
    fewer agents reduce consumption; counter-intuitively, higher demand
    means MORE head loss, so LOWER pressure at leaf nodes).
    With interventions reducing demand, flows shrink, head losses shrink,
    and pressure RISES toward the source head â€” so deviation becomes
    positive (pressure relief).
    """
    seeds   = list(range(N_SEEDS))
    cat_keys = list(CATEGORIES.keys())
    n_rows   = len(cat_keys) + 1
    n_cols   = len(ADOPTION_RATES)

    dev_matrix = np.zeros((n_rows, n_cols))

    print("\nRunning PRESSURE sensitivity matrix "
          f"({n_rows} rows Ã— {n_cols} cols Ã— {N_SEEDS} seeds)...\n")

    # baseline row â€” adoption irrelevant (no interventions, scenario=baseline)
    baseline_devs = []
    for seed in seeds:
        m = WaterNetworkModel(**NETWORK, interventions=[], adoption_rate=0.0, seed=seed)
        # baseline deviation is 0 by definition (nominal = baseline average)
        baseline_devs.append(0.0)
    for j in range(n_cols):
        dev_matrix[0, j] = 0.0
    print(f"  [baseline]  deviation=0.000 bar (reference)\n")

    for i, cat in enumerate(cat_keys, start=1):
        inv_ids = CATEGORIES[cat]
        for j, rate in enumerate(ADOPTION_RATES):
            devs = []
            for seed in seeds:
                m = WaterNetworkModel(
                    **NETWORK,
                    interventions=inv_ids,
                    adoption_rate=rate,
                    seed=seed,
                )
                devs.append(m.mean_pressure_deviation("intervention"))
            dev = float(np.mean(devs))
            dev_matrix[i, j] = dev
            print(f"  [{CATEGORY_LABELS[cat].split(chr(10))[0]:<22}]"
                  f"  adoption={ADOPTION_LABELS[j]}  "
                  f"deviation={dev:+.4f} bar")
        print()

    return dev_matrix, cat_keys


def plot_pressure_sensitivity(dev_matrix, cat_keys, out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)

    row_labels = ["Baseline\n(reference, delta=0)"] + [
        CATEGORY_LABELS[c] for c in cat_keys
    ]
    row_colors = ["#64748b"] + [CATEGORY_COLORS[c] for c in cat_keys]

    # Heatmap figure (standalone)
    fig = plt.figure(figsize=(12, 7))
    fig.patch.set_facecolor(FIG_BG)
    ax_heat = fig.add_subplot(111)

    # fig.suptitle(
    #     "Pressure Sensitivity Matrix - Intervention Category x Adoption Rate\n"
    #     "Cell: mean pressure deviation from nominal [bar]  (+)=relief, (-)=drop",
    #     color=TEXT, fontsize=13, fontweight="bold", y=1.02,
    # )
    
    fig.suptitle(
        "Pressure Sensitivity Matrix",
        color=TEXT, fontsize=13, fontweight="bold", y=1.02,
    )

    ax_heat.set_facecolor(AX_BG)
    vmax = max(abs(dev_matrix).max(), 0.001)
    cmap = plt.get_cmap("RdYlGn")
    im = ax_heat.imshow(dev_matrix, aspect="auto", cmap=cmap,
                        vmin=-vmax, vmax=vmax)

    for i in range(dev_matrix.shape[0]):
        for j in range(dev_matrix.shape[1]):
            val = dev_matrix[i, j]
            norm = (val + vmax) / (2 * vmax + 1e-9)
            tc = "#0f172a" if 0.2 < norm < 0.8 else TEXT
            ax_heat.text(j, i, f"{val:+.4f}",
                         ha="center", va="center",
                         fontsize=9.5, fontweight="bold", color=tc)

    ax_heat.set_xticks(range(len(ADOPTION_LABELS)))
    ax_heat.set_xticklabels(ADOPTION_LABELS, color=TEXT, fontsize=11)
    ax_heat.set_xlabel("Adoption rate ->", color=TEXT, fontsize=11, labelpad=8)
    ax_heat.set_yticks(range(len(row_labels)))
    ax_heat.set_yticklabels(row_labels, color=TEXT, fontsize=9)
    for tick, color in zip(ax_heat.get_yticklabels(), row_colors):
        tick.set_color(color)

    ax_heat.set_title("Mean pressure deviation from nominal (bar)",
                      color=TEXT, fontsize=10, pad=10)
    ax_heat.tick_params(colors=TEXT, length=0)
    for spine in ax_heat.spines.values():
        spine.set_edgecolor(GRID)

    for x in np.arange(-0.5, len(ADOPTION_RATES), 1):
        ax_heat.axvline(x, color=GRID, linewidth=1.0)
    for y in np.arange(-0.5, dev_matrix.shape[0], 1):
        ax_heat.axhline(y, color=GRID, linewidth=1.0)

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=TEXT, labelcolor=TEXT)
    cbar.set_label("delta pressure (bar)", color=TEXT, fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "pressure_sensitivity_matrix.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=140, facecolor=FIG_BG)
    plt.close(fig)
    print(f"  Saved -> {out_path}")

    # Trend line figure (standalone)
    fig_line, ax_line = plt.subplots(figsize=(9, 6))
    fig_line.patch.set_facecolor(FIG_BG)
    ax_line.set_facecolor(AX_BG)
    ax_line.tick_params(colors=TEXT)
    for spine in ax_line.spines.values():
        spine.set_edgecolor(GRID)
    ax_line.grid(color=GRID, linestyle="-", linewidth=0.8)
    ax_line.axhline(0, color="#64748b", linewidth=1.5, linestyle="--",
                    label="Baseline (ref)", zorder=1)

    x = np.array(ADOPTION_RATES) * 100
    for i, cat in enumerate(cat_keys, start=1):
        color = CATEGORY_COLORS[cat]
        label = CATEGORY_LABELS[cat].replace("\n", " ")
        ax_line.plot(x, dev_matrix[i, :],
                     marker="o", markersize=6, linewidth=2,
                     color=color, label=label, zorder=2)
        ax_line.annotate(f"{dev_matrix[i, -1]:+.4f}",
                         xy=(100, dev_matrix[i, -1]),
                         xytext=(4, 0), textcoords="offset points",
                         color=color, fontsize=8, va="center")

    ax_line.set_xlabel("Adoption rate (%)", color=TEXT, fontsize=10)
    ax_line.set_ylabel("delta pressure (bar)", color=TEXT, fontsize=10)
    ax_line.set_title("Pressure deviation vs Adoption Rate",
                      color=TEXT, fontsize=10, pad=8)
    ax_line.set_xticks([20, 40, 60, 80, 100])
    ax_line.set_xticklabels(["20%", "40%", "60%", "80%", "100%"], color=TEXT, fontsize=9)
    ax_line.legend(facecolor=AX_BG, edgecolor=GRID,
                   labelcolor=TEXT, fontsize=7.5, loc="lower right")

    plt.tight_layout()
    out_line_path = os.path.join(out_dir, "pressure_sensitivity_matrix_trend.png")
    fig_line.savefig(out_line_path, bbox_inches="tight", dpi=140, facecolor=FIG_BG)
    plt.close(fig_line)
    print(f"  Saved -> {out_line_path}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out_dir = os.path.join("output", "sensitivity")

    load_matrix, delta_matrix, cat_keys = run_sensitivity()
    plot_sensitivity(load_matrix, delta_matrix, cat_keys, out_dir)

    dev_matrix, _ = run_pressure_sensitivity()
    plot_pressure_sensitivity(dev_matrix, cat_keys, out_dir)

    print("\nDone.")
