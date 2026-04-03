"""
visualization.py
Produces all charts for the water consumption simulator.
All figures are saved to the output/ directory.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --------------------------------------------------------------------------
# Style constants
# --------------------------------------------------------------------------
COLORS = {
    "family":    "#38bdf8",
    "working":   "#a78bfa",
    "workplace": "#34d399",
    "baseline":  "#94a3b8",
    "savings":   "#4ade80",
    "accent":    "#f59e0b",
}
PROFILE_COLORS = {
    "family":    "#1f77b4",
    "working":   "#ff7f0e",
    "workplace": "#2ca02c",
}
TYPE_LABELS = {
    "family":    "Family w/ Children",
    "working":   "Working Adults",
    "workplace": "Workplace / School",
}
CAT_COLORS = {
    "fixture":        "#38bdf8",
    "appliance":      "#a78bfa",
    "digital":        "#34d399",
    "infrastructure": "#f59e0b",
    "behaviour":      "#fb923c",
}

FIG_BG = "#ffffff"
AX_BG  = "#ffffff"
TEXT   = "#111111"
GRID   = "#dddddd"

def _dark_fig(figsize=(12, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(color=GRID, linestyle="-", linewidth=0.8)
    return fig, ax

def _dark_fig_multi(rows, cols, figsize):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.patch.set_facecolor(FIG_BG)
    for ax in np.array(axes).flat:
        ax.set_facecolor(AX_BG)
        ax.tick_params(colors=TEXT)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, linestyle="-", linewidth=0.8)
    return fig, axes

def _save(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")


# --------------------------------------------------------------------------
# Plot 1: Hourly profiles — baseline vs intervention, stacked by type
# --------------------------------------------------------------------------

def plot_hourly_profiles(model, out_dir):
    hours = list(range(24))
    hour_labels = [f"{h:02d}h" for h in hours]

    base = model.hourly_profile("baseline")
    intv = model.hourly_profile("intervention")

    fig, axes = _dark_fig_multi(1, 2, figsize=(16, 5))
    fig.suptitle("Hourly Water Consumption Profile — Baseline vs. Intervention",
                 color=TEXT, fontsize=13, fontweight="bold", y=1.02)

    for ax, scenario, profiles, title in [
        (axes[0], "baseline", base, "Baseline (no interventions)"),
        (axes[1], "intervention", intv, "With Interventions"),
    ]:
        bottom = np.zeros(24)
        for atype in ("family", "working", "workplace"):
            vals = np.array(profiles[atype])
            n    = len(model.agents_by_type(atype))
            network_vals = vals * n
            ax.bar(hours, network_vals, bottom=bottom,
                   color=PROFILE_COLORS[atype], label=TYPE_LABELS[atype], width=0.8)
            bottom += network_vals

        ax.set_title(title, color=TEXT, fontsize=11)
        ax.set_xlabel("Hour of day", color=TEXT)
        ax.set_ylabel("Network consumption (L/h)", color=TEXT)
        ax.set_xticks(hours[::2])
        ax.set_xticklabels(hour_labels[::2], rotation=45, ha="right", fontsize=8)
        leg = ax.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "01_hourly_profiles.png"))


# --------------------------------------------------------------------------
# Plot 2: Per-type daily consumption bar chart (baseline vs intervention)
# --------------------------------------------------------------------------

def plot_per_type_comparison(model, out_dir):
    summary = model.per_type_summary()
    types   = list(summary.keys())
    x       = np.arange(len(types))
    width   = 0.35

    fig, ax = _dark_fig(figsize=(10, 5))
    ax.set_title("Daily Consumption per Agent Type — Baseline vs. Intervention",
                 color=TEXT, fontsize=12, fontweight="bold")

    bars1 = ax.bar(x - width/2,
                   [summary[t]["avg_baseline_L"] for t in types],
                   width, color=COLORS["baseline"], label="Baseline")
    bars2 = ax.bar(x + width/2,
                   [summary[t]["avg_intervention_L"] for t in types],
                   width, color=COLORS["savings"], label="With Interventions")

    # Annotate savings %
    for i, t in enumerate(types):
        pct = summary[t]["avg_savings_pct"]
        ax.text(x[i] + width/2, summary[t]["avg_intervention_L"] + 2,
                f"−{pct:.1f}%", ha="center", va="bottom",
                color=COLORS["savings"], fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([TYPE_LABELS[t] for t in types], color=TEXT)
    ax.set_ylabel("Average daily consumption per agent (L/day)", color=TEXT)
    leg = ax.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "02_per_type_comparison.png"))


# --------------------------------------------------------------------------
# Plot 3: Network totals — baseline vs intervention (horizontal bar)
# --------------------------------------------------------------------------

def plot_network_totals(model, out_dir):
    base_total = model.total_daily_baseline()
    intv_total = model.total_daily_intervention()
    savings    = model.total_daily_savings()
    pct        = model.network_savings_pct()

    fig, ax = _dark_fig(figsize=(10, 4))
    ax.set_title("Network-Wide Daily Consumption: Baseline vs. Intervention",
                 color=TEXT, fontsize=12, fontweight="bold")

    labels = ["Baseline", "With Interventions"]
    values = [base_total, intv_total]
    colors = [COLORS["baseline"], COLORS["savings"]]

    bars = ax.barh(labels, values, color=colors, height=0.4)
    for bar, val in zip(bars, values):
        ax.text(val + 50, bar.get_y() + bar.get_height()/2,
                f"{val:,.0f} L/day", va="center", color=TEXT, fontsize=10)

    ax.text(intv_total / 2, 0, f"Savings: {savings:,.0f} L/day ({pct:.1f}%)",
            ha="center", va="center", color=TEXT, fontsize=11, fontweight="bold")

    ax.set_xlabel("Total network daily consumption (L/day)", color=TEXT)
    ax.set_xlim(0, base_total * 1.15)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "03_network_totals.png"))


# --------------------------------------------------------------------------
# Plot 4: Intervention savings breakdown (waterfall-style bar)
# --------------------------------------------------------------------------

def plot_intervention_breakdown(model, out_dir):
    breakdown = model.intervention_savings_breakdown()
    if not breakdown:
        return

    labels  = [v["label"] for v in breakdown.values()]
    savings = [v["savings_L"] for v in breakdown.values()]
    cats    = [v["category"] for v in breakdown.values()]
    pcts    = [v["savings_pct"] for v in breakdown.values()]
    colors  = [CAT_COLORS.get(c, "#94a3b8") for c in cats]

    # Sort by savings descending
    order   = sorted(range(len(savings)), key=lambda i: savings[i], reverse=True)
    labels  = [labels[i] for i in order]
    savings = [savings[i] for i in order]
    cats    = [cats[i] for i in order]
    pcts    = [pcts[i] for i in order]
    colors  = [colors[i] for i in order]

    fig, ax = _dark_fig(figsize=(12, 5))
    ax.set_title("Estimated Daily Savings per Intervention (network-wide)",
                 color=TEXT, fontsize=12, fontweight="bold")

    x = np.arange(len(labels))
    bars = ax.bar(x, savings, color=colors, width=0.6)

    for bar, val, pct in zip(bars, savings, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, val + 5,
                f"{val:,.0f} L\n({pct:.1f}%)", ha="center", va="bottom",
                color=TEXT, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", color=TEXT, fontsize=9)
    ax.set_ylabel("Estimated savings (L/day)", color=TEXT)

    # Category legend
    seen = {}
    for cat, col in zip(cats, colors):
        if cat not in seen:
            seen[cat] = col
    patches = [mpatches.Patch(facecolor=c, label=cat)
               for cat, c in seen.items()]
    ax.legend(handles=patches, facecolor=AX_BG, edgecolor=GRID,
              labelcolor=TEXT, fontsize=8)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "04_intervention_breakdown.png"))


# --------------------------------------------------------------------------
# Plot 5: Agent distribution + savings histogram
# --------------------------------------------------------------------------

def plot_agent_savings_distribution(model, out_dir):
    fig, axes = _dark_fig_multi(1, 2, figsize=(14, 5))
    fig.suptitle("Agent-Level Savings Distribution",
                 color=TEXT, fontsize=13, fontweight="bold")

    # Left: pie chart of agent composition
    ax = axes[0]
    ax.set_facecolor(AX_BG)
    type_counts = {t: len(model.agents_by_type(t))
                   for t in ("family", "working", "workplace")}
    wedge_colors = [COLORS[t] for t in type_counts]
    wedges, texts, autotexts = ax.pie(
        type_counts.values(),
        labels=[TYPE_LABELS[t] for t in type_counts],
        colors=wedge_colors,
        autopct="%1.0f%%",
        startangle=90,
        textprops={"color": TEXT, "fontsize": 9},
    )
    for at in autotexts:
        at.set_color(TEXT)
        at.set_fontweight("bold")
    ax.set_title("Agent Composition", color=TEXT, fontsize=10)

    # Right: histogram of per-agent daily savings %
    ax2 = axes[1]
    for atype in ("family", "working", "workplace"):
        group = model.agents_by_type(atype)
        if group:
            pcts = [a.savings_pct() for a in group]
            ax2.hist(pcts, bins=15, color=COLORS[atype],
                     label=TYPE_LABELS[atype], edgecolor=AX_BG)
    ax2.set_xlabel("Daily savings (%)", color=TEXT)
    ax2.set_ylabel("Number of agents", color=TEXT)
    ax2.set_title("Distribution of Savings % per Agent", color=TEXT, fontsize=10)
    leg = ax2.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "05_agent_distribution.png"))


# --------------------------------------------------------------------------
# Plot 6: Network hourly total load — baseline vs intervention
# --------------------------------------------------------------------------

def plot_hourly_line_comparison(model, out_dir):
    hours      = list(range(24))
    base_total = model.network_hourly_total("baseline")
    intv_total = model.network_hourly_total("intervention")

    fig, ax = _dark_fig(figsize=(12, 5))
    ax.set_title("Network Hourly Load: Baseline vs. Intervention",
                 color=TEXT, fontsize=12, fontweight="bold")

    ax.plot(hours, base_total, color=COLORS["baseline"],
            linewidth=2, marker="o", markersize=4, label="Baseline")
    ax.plot(hours, intv_total, color=COLORS["savings"],
            linewidth=2, marker="o", markersize=4, label="With Interventions")

    ax.set_xlabel("Hour of day", color=TEXT)
    ax.set_ylabel("Total network consumption (L/h)", color=TEXT)
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}h" for h in hours], rotation=45,
                       ha="right", fontsize=7)
    ax.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "06_hourly_line_comparison.png"))


# --------------------------------------------------------------------------
# Plot 7: Network mean pressure — baseline vs intervention
# --------------------------------------------------------------------------

PRESSURE_COLORS = ["#38bdf8", "#a78bfa", "#34d399", "#f59e0b", "#fb923c",
                   "#f87171", "#60a5fa", "#c084fc", "#4ade80", "#facc15"]

def _build_node_pressure_profiles(model):
    """Return {scenario: {leaf_id: [pressure_bar x 24]}} for both scenarios."""
    hours = list(range(24))
    net = model.pipe_network
    profiles = {}
    for scenario in ("baseline", "intervention"):
        profiles[scenario] = {}
        for h in hours:
            p, _, _ = net.solve(h, scenario)
            for lid in net.leaf_ids:
                profiles[scenario].setdefault(lid, []).append(p[lid])
    return profiles

def plot_network_pressure(model, out_dir):
    hours = list(range(24))
    hour_labels = [f"{h:02d}h" for h in hours]
    leaf_ids = model.pipe_network.leaf_ids
    profiles = _build_node_pressure_profiles(model)

    mean_base = [np.mean([profiles["baseline"][lid][h] for lid in leaf_ids])
                 for h in hours]
    mean_intv = [np.mean([profiles["intervention"][lid][h] for lid in leaf_ids])
                 for h in hours]

    fig, ax = _dark_fig(figsize=(12, 5))
    ax.set_title("Network Mean Pressure (bar): Baseline vs. Intervention",
                 color=TEXT, fontsize=12, fontweight="bold")

    ax.plot(hours, mean_base, color=COLORS["baseline"],
            linewidth=2, marker="o", markersize=4, label="Baseline")
    ax.plot(hours, mean_intv, color=COLORS["savings"],
            linewidth=2, marker="o", markersize=4, label="With Interventions")

    ax.set_xlabel("Hour of day", color=TEXT)
    ax.set_ylabel("Mean leaf node pressure (bar)", color=TEXT)
    ax.set_xticks(hours[::2])
    ax.set_xticklabels(hour_labels[::2], rotation=45, ha="right", fontsize=8)
    ax.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "07_network_pressure.png"))


# --------------------------------------------------------------------------
# Plot 8: Per-node hourly load — all leaf nodes, baseline vs intervention
# --------------------------------------------------------------------------

def plot_node_load_profiles(model, out_dir):
    hours = list(range(24))
    net = model.pipe_network
    leaf_ids = net.leaf_ids

    node_profiles = {}
    for scenario in ("baseline", "intervention"):
        node_profiles[scenario] = {}
        for h in hours:
            for lid in leaf_ids:
                agents = net.node_agents.get(lid, [])
                fn = "hourly_baseline" if scenario == "baseline" else "hourly_intervention"
                q = sum(getattr(a, fn)(h) for a in agents)
                node_profiles[scenario].setdefault(lid, []).append(q)

    fig, ax = _dark_fig(figsize=(13, 5))
    ax.set_title("Per-Node Hourly Load: Baseline vs. Intervention",
                 color=TEXT, fontsize=12, fontweight="bold")

    for i, lid in enumerate(leaf_ids):
        col = PRESSURE_COLORS[i % len(PRESSURE_COLORS)]
        ax.plot(hours, node_profiles["baseline"][lid],
                color=col, linewidth=1.2, alpha=0.7, label=f"Node {lid} — base")
        ax.plot(hours, node_profiles["intervention"][lid],
                color=col, linewidth=1.2, alpha=0.7, linestyle=":",
                label=f"Node {lid} — intv")

    ax.set_xlabel("Hour of day", color=TEXT)
    ax.set_ylabel("Node consumption (L/h)", color=TEXT)
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}h" for h in hours], rotation=45,
                       ha="right", fontsize=7)
    # Compact legend: one solid + one dotted entry only
    solid = mpatches.Patch(color="#94a3b8", label="Baseline")
    dotted = plt.Line2D([0], [0], color="#94a3b8", linestyle=":", label="Intervention")
    ax.legend(handles=[solid, dotted], facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "08_node_load_profiles.png"))


# --------------------------------------------------------------------------
# Plot 9: Per-node pressure profiles — all leaf nodes, baseline vs intervention
# --------------------------------------------------------------------------

def plot_node_pressure_profiles(model, out_dir):
    hours = list(range(24))
    hour_labels = [f"{h:02d}h" for h in hours]
    leaf_ids = model.pipe_network.leaf_ids
    profiles = _build_node_pressure_profiles(model)

    fig, ax = _dark_fig(figsize=(13, 5))
    ax.set_title("Per-Node Pressure (bar): Baseline vs. Intervention",
                 color=TEXT, fontsize=12, fontweight="bold")

    for i, lid in enumerate(leaf_ids):
        col = PRESSURE_COLORS[i % len(PRESSURE_COLORS)]
        ax.plot(hours, profiles["baseline"][lid],
                color=col, linewidth=1.2, alpha=0.7)
        ax.plot(hours, profiles["intervention"][lid],
                color=col, linewidth=1.2, alpha=0.7, linestyle=":")

    ax.set_xlabel("Hour of day", color=TEXT)
    ax.set_ylabel("Node pressure (bar)", color=TEXT)
    ax.set_xticks(hours[::2])
    ax.set_xticklabels(hour_labels[::2], rotation=45, ha="right", fontsize=8)
    solid = mpatches.Patch(color="#94a3b8", label="Baseline")
    dotted = plt.Line2D([0], [0], color="#94a3b8", linestyle=":", label="Intervention")
    ax.legend(handles=[solid, dotted], facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "09_node_pressure_profiles.png"))


# --------------------------------------------------------------------------
# Master call
# --------------------------------------------------------------------------

def generate_all_plots(model, out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nGenerating plots → {out_dir}/")
    plot_hourly_profiles(model, out_dir)
    plot_per_type_comparison(model, out_dir)
    plot_network_totals(model, out_dir)
    plot_intervention_breakdown(model, out_dir)
    plot_agent_savings_distribution(model, out_dir)
    plot_hourly_line_comparison(model, out_dir)
    plot_network_pressure(model, out_dir)
    plot_node_load_profiles(model, out_dir)
    plot_node_pressure_profiles(model, out_dir)
    print("All plots saved.\n")
