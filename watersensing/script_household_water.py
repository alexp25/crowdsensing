import matplotlib.pyplot as plt
import numpy as np

from household_model import HouseholdProfile, HouseholdWaterNetworkModel


def summarize_metrics(series_map):
    demand = series_map["active_demand"]
    queue = series_map["queued_households"]
    consumption = series_map["consumption_lpm"]
    service_zone_peak_load = series_map["max_service_zone_load"]
    pressure = series_map["pressure_index"]
    pressure_deficit = series_map["pressure_deficit"]
    min_pressure = series_map["min_service_zone_pressure"]
    return {
        "avg_active_demand": float(np.mean(demand)),
        "avg_queue": float(np.mean(queue)),
        "avg_consumption_lpm": float(np.mean(consumption)),
        "avg_pressure_index": float(np.mean(pressure)),
        "avg_pressure_deficit": float(np.mean(pressure_deficit)),
        "peak_active_demand": float(np.max(demand)),
        "peak_queue": float(np.max(queue)),
        "peak_consumption_lpm": float(np.max(consumption)),
        "peak_service_zone_load": float(np.max(service_zone_peak_load)),
        "worst_service_zone_pressure": float(np.min(min_pressure)),
    }


def expected_household_flow(profiles):
    total_share = sum(p.share for p in profiles)
    if total_share <= 0:
        return 0.0
    return sum((p.share / total_share) * p.flow_lpm_per_household for p in profiles)


def run_behavior_scenario(total_households: int = 1200, day_minutes: int = 24 * 60, seed: int = 42):
    """Scenario 1: three clearly separated group types."""
    profiles = [
        HouseholdProfile(
            name="Families With Children",
            share=0.42,
            dwell_range=(7, 13),
            flow_lpm_per_household=6.6,
            start_window=(0, day_minutes),
            start_interval=10,
            group_size_range=(4, 12),
            start_windows=[(6 * 60, 9 * 60), (12 * 60, 14 * 60), (18 * 60, 22 * 60)],
        ),
        HouseholdProfile(
            name="Working Adults Households",
            share=0.33,
            dwell_range=(5, 9),
            flow_lpm_per_household=5.3,
            start_window=(0, day_minutes),
            start_interval=10,
            group_size_range=(2, 8),
            start_windows=[(6 * 60, 9 * 60), (18 * 60, 22 * 60)],
        ),
        HouseholdProfile(
            name="Workplaces And Schools",
            share=0.25,
            dwell_range=(8, 12),
            flow_lpm_per_household=4.8,
            start_window=(0, day_minutes),
            start_interval=5,
            group_size_range=(10, 35),
            start_windows=[(8 * 60, 17 * 60)],
        ),
    ]

    model = HouseholdWaterNetworkModel(
        total_households=total_households,
        profiles=profiles,
        day_minutes=day_minutes,
        num_service_zones=8,
        service_zone_capacity_range=(30, 65),
        seed=seed,
        validate_state=True,
    )
    series = model.run()
    return series, summarize_metrics(series), profiles


def run_intervention_portfolio_scenario(total_households: int = 1200, day_minutes: int = 24 * 60, seed: int = 42):
    """Scenario 2: same three groups after targeted interventions."""
    profiles = [
        HouseholdProfile(
            name="Families With Children (Efficient Fixtures + Dishwasher Shift)",
            share=0.42,
            dwell_range=(6, 10),
            flow_lpm_per_household=5.1,
            start_window=(0, day_minutes),
            start_interval=10,
            group_size_range=(4, 12),
            start_windows=[(6 * 60, 9 * 60), (12 * 60, 14 * 60), (18 * 60, 22 * 60)],
        ),
        HouseholdProfile(
            name="Working Adults Households (Smart Meter Nudges)",
            share=0.33,
            dwell_range=(4, 7),
            flow_lpm_per_household=4.4,
            start_window=(0, day_minutes),
            start_interval=10,
            group_size_range=(2, 8),
            start_windows=[(6 * 60, 9 * 60), (18 * 60, 22 * 60)],
        ),
        HouseholdProfile(
            name="Workplaces And Schools (Retrofit + Leak Repair)",
            share=0.25,
            dwell_range=(6, 10),
            flow_lpm_per_household=3.9,
            start_window=(0, day_minutes),
            start_interval=5,
            group_size_range=(10, 35),
            start_windows=[(8 * 60, 17 * 60)],
        ),
    ]

    model = HouseholdWaterNetworkModel(
        total_households=total_households,
        profiles=profiles,
        day_minutes=day_minutes,
        num_service_zones=8,
        service_zone_capacity_range=(30, 65),
        seed=seed,
        validate_state=True,
    )
    series = model.run()
    return series, summarize_metrics(series), profiles


def print_summary(title: str, metrics: dict):
    print(title)
    print(f"  Avg Active Demand: {metrics['avg_active_demand']:.2f}")
    print(f"  Avg Queue: {metrics['avg_queue']:.2f}")
    print(f"  Avg Consumption (L/min): {metrics['avg_consumption_lpm']:.2f}")
    print(f"  Avg Pressure Index (0..1): {metrics['avg_pressure_index']:.3f}")
    print(f"  Avg Pressure Deficit: {metrics['avg_pressure_deficit']:.3f}")
    print(f"  Peak Active Demand: {metrics['peak_active_demand']:.2f}")
    print(f"  Peak Queue: {metrics['peak_queue']:.2f}")
    print(f"  Peak Consumption (L/min): {metrics['peak_consumption_lpm']:.2f}")
    print(f"  Peak Service zone Load: {metrics['peak_service_zone_load']:.2f}")
    print(f"  Worst Service zone Pressure (0..1): {metrics['worst_service_zone_pressure']:.3f}")


def print_delta(reference: dict, candidate: dict):
    print("Impact vs Scenario 1 (Behavior Baseline)")
    for key in [
        "avg_active_demand",
        "avg_queue",
        "avg_consumption_lpm",
        "avg_pressure_index",
        "avg_pressure_deficit",
        "peak_active_demand",
        "peak_queue",
        "peak_consumption_lpm",
        "peak_service_zone_load",
        "worst_service_zone_pressure",
    ]:
        delta = candidate[key] - reference[key]
        pct = (delta / reference[key] * 100.0) if reference[key] else 0.0
        print(f"  {key}: {delta:+.2f} ({pct:+.2f}%)")


def run_multi_seed_check(num_runs: int = 8):
    keys = [
        "avg_active_demand",
        "avg_queue",
        "avg_consumption_lpm",
        "avg_pressure_index",
        "avg_pressure_deficit",
        "peak_active_demand",
        "peak_queue",
        "peak_consumption_lpm",
        "peak_service_zone_load",
        "worst_service_zone_pressure",
    ]
    s1_vals = {k: [] for k in keys}
    s2_vals = {k: [] for k in keys}
    for seed in range(1, num_runs + 1):
        _, m1, _ = run_behavior_scenario(seed=seed)
        _, m2, _ = run_intervention_portfolio_scenario(seed=seed)
        for k in keys:
            s1_vals[k].append(m1[k])
            s2_vals[k].append(m2[k])

    print(f"Multi-seed check ({num_runs} runs)")
    for k in keys:
        mean1 = float(np.mean(s1_vals[k]))
        mean2 = float(np.mean(s2_vals[k]))
        delta = mean2 - mean1
        pct = (delta / mean1 * 100.0) if mean1 else 0.0
        print(f"  {k}: S1={mean1:.2f}, S2={mean2:.2f}, delta={delta:+.2f} ({pct:+.2f}%)")


def plot_results(series_a, series_b, profiles_a, profiles_b):
    fig = plt.figure(figsize=(14, 10))

    plt.subplot(3, 2, 1)
    plt.plot(series_a["active_demand"], label="Scenario 1")
    plt.plot(series_b["active_demand"], label="Scenario 2")
    plt.title("Active Household Demand")
    plt.xlabel("Time Step (min)")
    plt.ylabel("Households")
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(series_a["queued_households"], label="Scenario 1")
    plt.plot(series_b["queued_households"], label="Scenario 2")
    plt.title("Queued Households")
    plt.xlabel("Time Step (min)")
    plt.ylabel("Households")
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(series_a["consumption_lpm"], label="Scenario 1")
    plt.plot(series_b["consumption_lpm"], label="Scenario 2")
    plt.title("Estimated Consumption")
    plt.xlabel("Time Step (min)")
    plt.ylabel("Liters / minute")
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(series_a["pressure_index"], label="Scenario 1")
    plt.plot(series_b["pressure_index"], label="Scenario 2")
    plt.title("Network Pressure Index")
    plt.xlabel("Time Step (min)")
    plt.ylabel("Pressure (0..1)")
    plt.ylim(0, 1.02)
    plt.legend()

    plt.subplot(3, 2, 5)
    for p in profiles_a:
        key = f"profile_active::{p.name}"
        plt.plot(series_a[key], alpha=0.9, label=f"S1 {p.name}")
    for p in profiles_b:
        key = f"profile_active::{p.name}"
        plt.plot(series_b[key], linestyle="--", alpha=0.8, label=f"S2 {p.name}")
    plt.title("Per-Group Active Demand")
    plt.xlabel("Time Step (min)")
    plt.ylabel("Households")
    plt.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    fig.savefig("output/household_water_scenarios.png", dpi=fig.dpi)
    if "agg" in plt.get_backend().lower():
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    scenario1_series, scenario1_metrics, scenario1_profiles = run_behavior_scenario()
    scenario2_series, scenario2_metrics, scenario2_profiles = run_intervention_portfolio_scenario()

    print(f"Expected per-household flow (Scenario 1): {expected_household_flow(scenario1_profiles):.2f} L/min")
    print(f"Expected per-household flow (Scenario 2): {expected_household_flow(scenario2_profiles):.2f} L/min")
    print_summary("Scenario 1: Behavioral Segments", scenario1_metrics)
    print_summary("Scenario 2: Intervention Portfolio", scenario2_metrics)
    print_delta(scenario1_metrics, scenario2_metrics)
    plot_results(scenario1_series, scenario2_series, scenario1_profiles, scenario2_profiles)
    run_multi_seed_check()
