import io
from contextlib import redirect_stdout

from rich.console import Console


def test_label_smoothing_arg_parses_and_prints():
    from train_frames import parse_args
    args = parse_args(["--clf","dl","--label-smoothing","0.12"])  # minimal DL args
    assert hasattr(args, "label_smoothing") and abs(args.label_smoothing - 0.12) < 1e-9

    # Ensure hparams print reflects the value
    from vkb.finetune import _print_hparams
    cons = Console(record=True)
    _print_hparams(cons, args)
    out = cons.export_text()
    assert "label_smoothing" in out
    assert "0.12" in out

