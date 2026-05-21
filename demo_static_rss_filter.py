"""Demo for Progress-Aware RSS Action Projection.

Run:
    python demo_static_rss_filter.py
"""

import argparse
import csv
from pathlib import Path
from pprint import pprint

from ppl.utils.static_rss_filter import StaticRSSConfig, StaticRSSFilter


def make_state(with_obstacle=True, left_available=True, right_available=False):
    lanes = {
        "current": {
            "lane_id": "center",
            "width": 3.5,
        }
    }
    if left_available:
        lanes["left"] = {
            "lane_id": "left",
            "width": 3.5,
            "available": True,
            "drivable": True,
        }
    if right_available:
        lanes["right"] = {
            "lane_id": "right",
            "width": 3.5,
            "available": True,
            "drivable": True,
        }

    state = {
        "ego": {
            "x": 0.0,
            "y": 0.0,
            "heading": 0.0,
            "speed": 8.0,
            "lane_id": "center",
            "lane_width": 3.5,
        },
        "static_obstacles": [
            {
                "x": 30.0,
                "y": 0.0,
                "length": 4.5,
                "width": 2.0,
                "heading": 0.0,
                "lane_id": "center",
            }
        ],
        "vehicles": [],
        "lanes": lanes,
    }
    if not with_obstacle:
        state["static_obstacles"] = []
    return state


def run_case(title, filter_, state, u_nom):
    u_safe, info = filter_.filter_action(state, u_nom)

    print("\n=== {} ===".format(title))
    print("mode:", info["mode"])
    print("u_nom:", u_nom)
    print("u_safe:", u_safe)
    print("d_obs:", info.get("d_obs"))
    print("d_brake:", info.get("d_brake"))
    print("left_feasible:", info.get("left_feasible"))
    print("right_feasible:", info.get("right_feasible"))
    print("candidate scores:")
    for candidate in info.get("candidates", []):
        pprint(
            {
                "mode": candidate["mode"],
                "safe": candidate["safe"],
                "action": candidate["action"],
                "progress_score": candidate["progress_score"],
                "intervention_cost": candidate["intervention_cost"],
                "margins": candidate["margins"],
            }
        )
    selected = info.get("selected", {})
    summary_row = {
        "scenario": title,
        "nominal_action_acc": u_nom[0],
        "nominal_action_steer": u_nom[1],
        "filtered_action_acc": u_safe[0],
        "filtered_action_steer": u_safe[1],
        "selected_mode": info["mode"],
        "filter_applied": info["mode"] != "normal",
        "obstacle_detected": info.get("obstacle_detected", False),
        "distance_to_obstacle_m": info.get("d_obs"),
        "rss_brake_distance_m": info.get("d_brake"),
        "rss_margin_m": info.get("rss_margin"),
        "left_feasible": info.get("left_feasible"),
        "right_feasible": info.get("right_feasible"),
        "selected_progress_score": selected.get("progress_score"),
        "selected_intervention_cost": selected.get("intervention_cost"),
        "selected_score": info.get("selected_score"),
        "decision_meaning": describe_decision(info),
    }
    candidate_rows = []
    for candidate in info.get("candidates", []):
        margins = candidate.get("margins", {})
        candidate_rows.append(
            {
                "scenario": title,
                "candidate_mode": candidate.get("mode"),
                "selected": candidate.get("mode") == info["mode"],
                "safe": candidate.get("safe"),
                "action_acc": candidate.get("action", [None, None])[0],
                "action_steer": candidate.get("action", [None, None])[1],
                "progress_score": candidate.get("progress_score"),
                "intervention_cost": candidate.get("intervention_cost"),
                "h_stop_min": margins.get("h_stop_min"),
                "obstacle_clearance_margin": margins.get("obstacle_clearance_margin"),
                "lane_boundary_margin": margins.get("lane_boundary_margin"),
                "progress_margin": margins.get("progress_margin"),
                "lateral_progress_margin": margins.get("lateral_progress_margin"),
                "front_vehicle_rss_margin": margins.get("front_rss_margin"),
                "rear_vehicle_rss_margin": margins.get("rear_rss_margin"),
                "steer_direction_margin": margins.get("steer_direction_margin"),
            }
        )
    return summary_row, candidate_rows


def describe_decision(info):
    mode = info["mode"]
    if mode == "normal":
        return "No static obstacle ahead; keep nominal policy action."
    if mode == "left_bypass":
        return "Static obstacle ahead; left bypass is safe and gives more progress than stop."
    if mode == "right_bypass":
        return "Static obstacle ahead; right bypass is safe and gives more progress than stop."
    if mode == "stop":
        return "Static obstacle ahead; no safe bypass candidate, so use RSS stop."
    return "No safe candidate; use fallback maximum braking."


def write_csv(rows, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Run the static RSS filter demo.")
    parser.add_argument(
        "--csv",
        default="evaluation_results/static_rss_filter_summary.csv",
        help="Path to save scenario-level filtered-action summary CSV.",
    )
    parser.add_argument(
        "--candidate_csv",
        default="evaluation_results/static_rss_filter_candidates.csv",
        help="Path to save one-row-per-candidate action comparison CSV.",
    )
    args = parser.parse_args()

    rss_filter = StaticRSSFilter(StaticRSSConfig())
    u_nom = [1.0, 0.0]
    summary_rows = []
    candidate_rows = []

    normal_state = make_state(with_obstacle=False, left_available=True, right_available=False)
    summary, candidates = run_case("no obstacle", rss_filter, normal_state, u_nom)
    summary_rows.append(summary)
    candidate_rows.extend(candidates)

    left_state = make_state(with_obstacle=True, left_available=True, right_available=False)
    summary, candidates = run_case("left bypass available", rss_filter, left_state, u_nom)
    summary_rows.append(summary)
    candidate_rows.extend(candidates)

    stop_state = make_state(with_obstacle=True, left_available=False, right_available=False)
    summary, candidates = run_case("no bypass available", rss_filter, stop_state, u_nom)
    summary_rows.append(summary)
    candidate_rows.extend(candidates)

    summary_path = write_csv(summary_rows, args.csv)
    candidate_path = write_csv(candidate_rows, args.candidate_csv)
    print("\nSummary CSV saved to: {}".format(summary_path))
    print("Candidate CSV saved to: {}".format(candidate_path))


if __name__ == "__main__":
    main()
