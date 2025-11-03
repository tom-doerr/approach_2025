def test_helpers_exist():
    import vkb.finetune as ft
    for name in (
        '_prepare_data',
        '_print_device',
        '_print_hparams',
        '_make_loaders',
        '_init_model_and_optim',
        '_save_artifacts',
    ):
        assert hasattr(ft, name), f"missing helper: {name}"

