def test_more_finetune_helpers_exist():
    import vkb.finetune as ft
    for name in (
        '_mlflow_log_params',
        '_init_progress',
        '_setup_viz_and_telemetry',
        '_epoch_after',
    ):
        assert hasattr(ft, name), f"missing helper: {name}"

