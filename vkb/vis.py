"""Visdom utilities (moved out of vkb.finetune).

We expose:
- setup_visdom(args, cons)
- visdom_prepare(viz)
- viz_scalar(viz, wins, series, x, y, title)
- viz_aug_text(viz, wins, args, ep)
"""

def setup_visdom(args, cons):
    vcount = int(getattr(args, 'visdom_aug', 0) or 0)
    want_metrics = bool(getattr(args, 'visdom_metrics', False))
    # If user didn't request any Visdom features, do not touch Visdom at all.
    if vcount <= 0 and not want_metrics:
        return None
    auto_try = False
    import os as _os
    if _os.getenv('VKB_VISDOM_DISABLE', '') == '1':
        return None
    try:
        from visdom import Visdom
    except Exception:
        if not auto_try:
            cons.print("[yellow]Visdom not installed; skipping logging (pip install visdom).[/]")
        return None
    env = str(getattr(args, 'visdom_env', _os.getenv('VKB_VISDOM_ENV', 'vkb-aug')))
    port = int(_os.getenv('VKB_VISDOM_PORT', getattr(args, 'visdom_port', 8097)))
    server = _os.getenv('VKB_VISDOM_SERVER', None)
    viz = Visdom(port=port, env=env) if not server else Visdom(server=server, port=port, env=env)
    try:
        ok = bool(getattr(viz, 'check_connection', lambda: True)())
    except Exception:
        ok = True
    if not ok:
        cons.print(f"[yellow]Visdom server not reachable (env={env} port={port}); no logging.[/]")
        return None
    cons.print(f"[dim]Visdom: logging to env={env} port={port}[/]")
    return viz


def visdom_prepare(viz):
    import os as _os
    if viz is None:
        return
    # Do not clear existing Visdom windows during tests or when opted out
    if _os.getenv('PYTEST_CURRENT_TEST') or _os.getenv('VKB_VISDOM_NO_CLEAR', '') == '1':
        return
    try:
        for w in ("train_acc", "val_acc"):
            viz.close(win=w)
    except Exception:
        pass


def viz_scalar(viz, wins: dict, series: str, x: float, y: float, title: str):
    if viz is None:
        return
    win_id = 'vkb_acc'
    if 'acc' not in wins:
        wins['acc'] = win_id
        viz.line(X=[x], Y=[y], win=win_id, name=series,
                 opts={"title": title, "xlabel": "epoch", "ylabel": "acc", "legend": ["train","val"]})
    else:
        viz.line(X=[x], Y=[y], win=win_id, name=series, update="append")


def viz_aug_text(viz, wins: dict, args, ep: int):
    if viz is None:
        return
    txt = (
        f"epoch {ep}/{getattr(args,'epochs',ep)}: "
        f"aug={getattr(args,'aug','none')} "
        f"brightness={getattr(args,'brightness',0.0)} "
        f"warp={getattr(args,'warp',0.0)} "
        f"sat={getattr(args,'sat',0.0)} "
        f"contrast={getattr(args,'contrast',0.0)} "
        f"wb={getattr(args,'wb',0.0)} "
        f"hue={getattr(args,'hue',0.0)} "
        f"rot_deg={getattr(args,'rot_deg',0.0)} "
        f"drop_path={getattr(args,'drop_path',0.0)} "
        f"dropout={getattr(args,'dropout',0.0)}"
    )
    win_id = 'vkb_policy'
    try:
        if 'aug_text' not in wins:
            wins['aug_text'] = win_id
            viz.text(txt, win=win_id, opts={"title": "Aug/Reg Policy"})
        else:
            viz.text("\n" + txt, win=win_id, append=True)
    except Exception:
        pass
