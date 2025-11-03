def test_train_epoch_split_helpers_exist():
    import vkb.finetune as ft
    for name in (
        '_log_aug_samples_if_first_batch',
        '_forward_backward_step',
        '_print_batch_progress',
    ):
        assert hasattr(ft, name), f"missing helper: {name}"

