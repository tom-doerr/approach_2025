from rich.console import Console


def test_dl_hparams_key_order():
    # Construct a minimal args object with DL params
    from vkb.finetune import _print_hparams
    class A: pass
    a = A()
    # expected fields present in stable order
    a.backbone = 'mobilenetv3_small_100'
    a.epochs = 1
    a.batch_size = 8
    a.lr = 1e-4
    a.wd = 1e-4
    a.eval_split = 0.2
    a.eval_mode = 'tail'
    a.drop_path = 0.25
    a.dropout = 0.0
    a.aug = 'rot360'
    a.brightness = 0.15
    a.warp = 0.2
    a.sat = 0.0; a.contrast = 0.0; a.wb = 0.0; a.hue = 0.0; a.rot_deg = 0.0
    a.val_eq_eps = 0.002
    a.label_smoothing = 0.05

    cons = Console(record=True)
    _print_hparams(cons, a)
    text = cons.export_text()

    expected_order = [
        'backbone', 'epochs', 'batch_size', 'lr', 'weight_decay', 'eval_split',
        'eval_mode', 'optimizer', 'label_smoothing', 'drop_path', 'dropout',
        'aug', 'brightness', 'warp', 'sat', 'contrast', 'wb', 'hue', 'rot_deg', 'val_eq_eps'
    ]
    # Extract keys sequence from the printed table
    keys = []
    for line in text.splitlines():
        if not line.strip() or 'HParams' in line or line[0] in ('┏','┗','┡','└','┫','┣'):
            continue
        parts = [p.strip() for p in line.split('│') if p.strip()]
        if len(parts) >= 2:
            keys.append(parts[0])
    # Keep only those that are in our expected list
    seen = [k for k in keys if k in expected_order]
    # Assert expected sequence is a subsequence in order
    pos = -1
    for k in expected_order:
        try:
            i = seen.index(k, pos+1)
        except ValueError:
            assert False, f"missing key in HParams output: {k}"
        pos = i
