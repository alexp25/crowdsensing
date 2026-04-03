"""
run.py
Entry point for the Water Consumption ABM Simulator.

Built on Mesa (pip install mesa).
Mirrors the run.py / main.py convention used in alexp25/crowdsensing.

Usage
-----
    python run.py                          # default scenario
    python run.py --scenario all           # run all preset scenarios
    python run.py --scenario custom        # edit CUSTOM_CONFIG below

Scenarios
---------
  baseline        – No interventions (reference)
  fixtures        – Toilet + shower reducer only
  digital         – Smart meter + leak detection only
  full_package    – All 8 interventions
  behaviour_led   – Education program + smart meter only
  infra_heavy     – Rainwater tank + pressure limiter
"""

import argparse
import os
from model import WaterNetworkModel
from agents import INTERVENTIONS
from visualization import generate_all_plots

# ---------------------------------------------------------------------------
# Network composition (shared across all scenarios)
# ---------------------------------------------------------------------------
NETWORK = dict(
    n_families=300,
    n_working=400,
    n_workplaces=100,
    seed=42,
)

# ---------------------------------------------------------------------------
# Predefined intervention scenarios
# ---------------------------------------------------------------------------
SCENARIOS = {
    "baseline": {
        "label": "Baseline — No interventions",
        "interventions": [],
    },
    "fixtures": {
        "label": "Fixture upgrades only",
        "interventions": ["efficient_toilet", "shower_reducer"],
    },
    "digital": {
        "label": "Digital monitoring only",
        "interventions": ["smart_meter", "leak_detection"],
    },
    "behaviour_led": {
        "label": "Behaviour & education led",
        "interventions": ["education_program", "smart_meter"],
    },
    "infra_heavy": {
        "label": "Infrastructure heavy",
        "interventions": ["rainwater_tank", "pressure_limiter"],
    },
    "full_package": {
        "label": "Full intervention package (all 8)",
        "interventions": list(INTERVENTIONS.keys()),
    },
}

# ---------------------------------------------------------------------------
# Custom scenario — edit here or pass --scenario custom
# ---------------------------------------------------------------------------
CUSTOM_CONFIG = {
    "label": "Custom scenario",
    "interventions": ["efficient_toilet", "smart_meter", "education_program"],
}


# ---------------------------------------------------------------------------
def print_summary(model, label):
    summary = model.per_type_summary()
    breakdown = model.intervention_savings_breakdown()

    print("\n" + "═" * 62)
    print(f"  SCENARIO: {label}")
    print("═" * 62)
    print(f"  Network: {model.n_families} family | "
          f"{model.n_working} working | {model.n_workplaces} workplace agents")
    print(f"  Active interventions: "
          f"{model.interventions if model.interventions else '(none)'}")
    print()

    # Per-type table
    hdr = f"  {'Type':<22} {'Agents':>6}  {'Baseline':>10}  {'w/ Interv.':>10}  {'Savings':>8}"
    print(hdr)
    print("  " + "-" * 58)
    for atype, s in summary.items():
        label_str = {"family": "Family w/ Children",
                     "working": "Working Adults",
                     "workplace": "Workplace/School"}[atype]
        print(f"  {label_str:<22} {s['count']:>6}  "
              f"{s['avg_baseline_L']:>9.1f}L  "
              f"{s['avg_intervention_L']:>9.1f}L  "
              f"{s['avg_savings_pct']:>7.1f}%")

    print()
    print(f"  {'NETWORK TOTAL':<22} "
          f"{'':>6}  "
          f"{model.total_daily_baseline():>9.0f}L  "
          f"{model.total_daily_intervention():>9.0f}L  "
          f"{model.network_savings_pct():>7.1f}%")
    print()

    if breakdown:
        print("  Intervention contributions:")
        for inv_id, v in breakdown.items():
            print(f"    [{v['category']:<14}]  {v['label']:<35}  "
                  f"{v['savings_L']:>7.0f} L/day  ({v['savings_pct']:.1f}%)")

    print()


# ---------------------------------------------------------------------------
def run_scenario(name, config, out_root="output"):
    label = config["label"]
    interventions = config["interventions"]

    model = WaterNetworkModel(
        **NETWORK,
        interventions=interventions,
    )

    print_summary(model, label)

    out_dir = os.path.join(out_root, name)
    generate_all_plots(model, out_dir)
    return model


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Water Consumption ABM Simulator"
    )
    parser.add_argument(
        "--scenario",
        default="full_package",
        choices=list(SCENARIOS.keys()) + ["all", "custom"],
        help="Which scenario to run (default: full_package)",
    )
    parser.add_argument(
        "--out", default="output",
        help="Output directory for plots (default: output/)",
    )
    args = parser.parse_args()

    if args.scenario == "all":
        for name, config in SCENARIOS.items():
            run_scenario(name, config, out_root=args.out)
    elif args.scenario == "custom":
        run_scenario("custom", CUSTOM_CONFIG, out_root=args.out)
    else:
        run_scenario(args.scenario, SCENARIOS[args.scenario], out_root=args.out)


if __name__ == "__main__":
    main()
