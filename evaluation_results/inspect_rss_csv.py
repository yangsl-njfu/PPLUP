import pandas as pd

path = r"evaluation_results/rss_type_debug\checkpoint_3000_static_rss_steps.csv"
df = pd.read_csv(path)

print("Mode counts:")
print(df["mode"].value_counts())

fb = df[df["mode"] == "fallback_no_safe_candidate"]
print("\nFallback rows:", len(fb))
print("\nColumns:")
print(df.columns.tolist())

cols = [
    "step", "mode", "reason",
    "obstacle_detected", "dynamic_vehicle_detected",
    "d_obs", "d_brake", "rss_margin",
    "clearance_margin", "risk_step",
    "left_feasible", "right_feasible",
    "num_static_obstacles", "num_dynamic_vehicles",
]
cols = [c for c in cols if c in df.columns]

print("\nFallback sample:")
print(fb[cols].head(30).to_string())

num_cols = [c for c in ["d_obs", "d_brake", "rss_margin", "clearance_margin"] if c in fb.columns]
print("\nFallback numeric summary:")
print(fb[num_cols].describe())