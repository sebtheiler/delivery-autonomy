"""Synthetic route comparison: invented campus routes and travel times, no ROS.
32 half-hour departure windows, 06:00 inclusive to 22:00 exclusive.
Same 200 origin/destination pairs per window; three candidate paths per pair.
365 simulated history days and 60 independent test days.
Streams 7,008,000 historical observations to disk to keep memory bounded.
"""
import csv
import math
import random
from pathlib import Path
from statistics import mean, stdev

OUTPUT = Path(__file__).resolve().parent / "large_dataset"
SPEED = 1.5
HISTORY_DAYS = 365
TEST_DAYS = 60
PAIR_COUNT = 200


def activity(minute):
    # Illustrative assumptions, not measured WashU traffic.
    if minute >= 17 * 60:
        return 0.06 if minute < 19 * 60 else 0.025
    if minute < 8 * 60:
        return 0.08 if minute < 7 * 60 else 0.22
    lunch = 0.65 * math.exp(-((minute - 12.5 * 60) / 65) ** 2)
    class_change = 0.25 if minute % 60 == 0 else 0.05
    return 0.45 + lunch + class_change


def make_pairs():
    rng = random.Random(73)
    pairs = []
    for i in range(PAIR_COUNT):
        base = rng.uniform(110, 290)
        # Segment fields: distance, crowd exposure, crossing delay multiplier.
        entry = (base * 0.15, 0.12, 0)
        exit_segment = (base * 0.15, 0.12, 0)
        routes = {
            "direct": [entry, (base * 0.70, rng.uniform(0.45, 1.5), 1), exit_segment],
            "sidewalk_detour": [entry, (
                base * 0.70 + rng.uniform(18, 65), 0.18, 0.5), exit_segment],
            "outer_loop": [entry, (
                base * 0.70 + rng.uniform(70, 125), 0.025, 0), exit_segment],
        }
        pairs.append((f"A{i+1:02d}->B{i+1:02d}", routes))
    return pairs


def distance(segments):
    return sum(s[0] for s in segments)


def travel_time(segments, minute, rng, day_factor):
    crowd = activity(minute) * day_factor
    total = 0
    for length, exposure, crossing in segments:
        total += length / SPEED * (1 + crowd * exposure)
        total += crossing * (2 + 16 * crowd)
        if rng.random() < min(0.7, crowd * exposure * 0.30):
            total += rng.uniform(5, 22)
    return max(0.1, total * rng.uniform(0.96, 1.04))


def save_csv(name, rows):
    path = OUTPUT / name
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def clock(minute):
    return f"{minute // 60:02d}:{minute % 60:02d}"


def run(observations):
    pairs = make_pairs()
    train_rng, test_rng = random.Random(42), random.Random(2026)
    history_factors = [train_rng.uniform(0.75, 1.25) for _ in range(HISTORY_DAYS)]
    test_factors = [test_rng.uniform(0.75, 1.25) for _ in range(TEST_DAYS)]
    decisions, summaries = [], []
    observation_count = 0
    previous = {}
    print("SYNTHETIC DATA ONLY - invented paths and traffic; no robot integration")
    print(f"Same {PAIR_COUNT} A/B pairs each window; campus quiets after 17:00.")
    print(f"History: {HISTORY_DAYS} synthetic days; test: {TEST_DAYS} separate days.")
    print("Recommend only if estimated savings >=5 seconds AND >=5%,")
    print("with a positive approximate 95% lower bound on paired savings.")
    print("This uncertainty filter is illustrative, not calibrated validation.\n")
    print("Window       Faster alt  Recommend  Since prior  Test saving*")

    for minute in range(360, 1320, 30):
        label = f"{clock(minute)}-{clock(minute + 30)}"
        batch = []
        for pair, routes in pairs:
            samples = {}
            for route, segments in routes.items():
                samples[route] = []
                for day, factor in enumerate(history_factors, 1):
                    actual = travel_time(segments, minute, train_rng, factor)
                    samples[route].append(actual)
                    observations.writerow({
                        "synthetic": True, "history_day": day, "window": label,
                        "pair": pair, "route": route,
                        "distance_m": round(distance(segments), 3),
                        "distance_eta_seconds": round(distance(segments) / SPEED, 3),
                        "observed_seconds": round(actual, 3),
                    })
                    observation_count += 1
            baseline = min(routes, key=lambda r: distance(routes[r]))
            fastest = min(routes, key=lambda r: mean(samples[r]))
            differences = [a-b for a, b in zip(samples[baseline], samples[fastest])]
            saving = mean(differences)
            fraction = saving / mean(samples[baseline])
            lower = saving - 1.96 * stdev(differences) / math.sqrt(HISTORY_DAYS)
            recommend = (fastest != baseline and saving >= 5
                         and fraction >= 0.05 and lower > 0)
            selected = fastest if recommend else baseline

            # Evaluate only AFTER selecting; test data never selects the route.
            # Unchanged routes must have exactly zero counterfactual savings.
            test_savings = []
            for factor in test_factors:
                base_time = travel_time(routes[baseline], minute, test_rng, factor)
                chosen_time = (base_time if selected == baseline else
                               travel_time(routes[selected], minute, test_rng, factor))
                test_savings.append(base_time - chosen_time)
            row = {
                "synthetic": True, "window": label, "pair": pair,
                "distance_best": baseline, "data_best": fastest, "selected": selected,
                "faster_alternative": fastest != baseline, "recommend_change": recommend,
                "changed_since_previous_window": (
                    pair in previous and previous[pair] != selected),
                "samples_per_route": HISTORY_DAYS,
                "baseline_distance_m": round(distance(routes[baseline]), 2),
                "selected_distance_m": round(distance(routes[selected]), 2),
                "baseline_history_seconds": round(mean(samples[baseline]), 2),
                "fastest_history_seconds": round(mean(samples[fastest]), 2),
                "fastest_estimated_saving_seconds": round(saving, 2),
                "fastest_estimated_saving_percent": round(fraction * 100, 2),
                "approx_lower_saving_bound_seconds": round(lower, 2),
                "test_mean_saving_seconds": round(mean(test_savings), 2),
            }
            previous[pair] = selected
            decisions.append(row)
            batch.append(row)
        count = sum(r["recommend_change"] for r in batch)
        summary = {
            "synthetic": True, "window": label,
            "activity_factor": round(activity(minute), 3),
            "pairs_evaluated": PAIR_COUNT,
            "faster_alternatives": sum(r["faster_alternative"] for r in batch),
            "recommended_changes": count,
            "recommended_percent": round(100 * count / PAIR_COUNT, 2),
            "changes_since_previous_window": sum(
                r["changed_since_previous_window"] for r in batch),
            "test_mean_saving_per_request_seconds": round(mean(
                r["test_mean_saving_seconds"] for r in batch), 2),
        }
        summaries.append(summary)
        print(f"{label}   {summary['faster_alternatives']:>3}/{PAIR_COUNT}     {count:>3}/{PAIR_COUNT}"
              f"         {summary['changes_since_previous_window']:>2}"
              f"          {summary['test_mean_saving_per_request_seconds']:>6.1f}s",
              flush=True)

    print(f"\n* Savings on {TEST_DAYS} separate synthetic test days, averaged over")
    print(f"ALL {PAIR_COUNT} requests, including unchanged routes.")
    print("Recommend: alternative to distance routing in THIS window.")
    print("Since prior: selected routes changed since the prior window, including")
    print("switching back to the shortest route. First window = 0.")
    print(f"\nTotal: {sum(s['recommended_changes'] for s in summaries)} recommendations"
          f" across {len(decisions)} pair/window decisions (not unique routes).")
    expected_count = 32 * PAIR_COUNT * 3 * HISTORY_DAYS
    assert observation_count == expected_count
    assert len(decisions) == 32 * PAIR_COUNT
    print(f"Historical observation rows: {observation_count:,}")
    print(f"Independent test comparisons: {32 * PAIR_COUNT * TEST_DAYS:,}")
    save_csv("demo_route_decisions.csv", decisions)
    save_csv("demo_route_intervals.csv", summaries)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "demo_route_observations.csv"
    fields = ["synthetic", "history_day", "window", "pair", "route",
              "distance_m", "distance_eta_seconds", "observed_seconds"]
    # Stream directly rather than keeping millions of dictionaries in RAM.
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        run(writer)
    print(f"Saved: {path} ({path.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
