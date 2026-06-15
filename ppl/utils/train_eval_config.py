baseline_eval_config = dict(
    manual_control=False,
    use_render=False,
    main_exp=False,
    start_seed=1000,
    horizon=1500,
    traffic_density=0.08,
    accident_prob=0.0,
    crash_vehicle_done=True,
    crash_object_done=True,
)
baseline_train_config = dict(manual_control=False, use_render=False, main_exp=False, horizon=1500)
