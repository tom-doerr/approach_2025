from __future__ import annotations

from typing import Any, Optional


class Telemetry:
    def __init__(self, console: Any, viz: Optional[Any] = None, mlflow: Optional[Any] = None):
        self.cons = console
        self.viz = viz
        self.mlflow = mlflow
        self._wins: dict[str, str] = {}

    # Scalars ---------------------------------------------------------------
    def metric(self, name: str, value: float, step: Optional[int] = None):
        if self.mlflow is not None:
            try:
                self.mlflow.log_metric(name, float(value), step=step)
            except Exception:
                pass

    def scalar(self, series: str, step: float, value: float, title: str = "Train/Val Acc", win: str = "vkb_acc"):
        if self.viz is None:
            return
        try:
            if 'acc' not in self._wins:
                self._wins['acc'] = win
                self.viz.line(X=[step], Y=[value], win=win, name=series,
                              opts={"title": title, "xlabel": "epoch", "ylabel": "acc", "legend": ["train","val"]})
            else:
                self.viz.line(X=[step], Y=[value], win=win, name=series, update="append")
        except Exception:
            pass

    def scalar2(self, win_key: str, series: str, step: float, value: float, title: str, ylabel: str, win: str):
        if self.viz is None:
            return
        try:
            if win_key not in self._wins:
                self._wins[win_key] = win
                self.viz.line(
                    X=[step], Y=[value], win=win, name=series,
                    opts={
                        "title": title,
                        "xlabel": "epoch",
                        "ylabel": ylabel,
                        "legend": [series],
                        "ytype": "log",
                    },
                )
            else:
                self.viz.line(X=[step], Y=[value], win=win, name=series, update="append")
        except Exception:
            pass

    # Images ---------------------------------------------------------------
    def images(self, imgs, title: str = "Aug Samples", win: str = "vkb_aug"):
        # imgs: torch tensor [N,C,H,W] in [0,1]
        N = int(getattr(imgs, 'shape', [0])[0] or 0)
        if N <= 0:
            return
        # Visdom fanout
        if self.viz is not None:
            try:
                self.viz.images(imgs.detach().cpu().numpy(), nrow=min(N, 8), win=win,
                                opts={"title": title} if 'aug_images' not in self._wins else None)
                self._wins['aug_images'] = win
            except Exception:
                pass
        # MLflow artifact
        if self.mlflow is not None:
            try:
                import os, tempfile, numpy as _np, cv2 as _cv
                N, C, H, W = imgs.shape
                ncol = min(N, 8)
                rows = (N + ncol - 1)//ncol
                grid = _np.zeros((rows*H, ncol*W, 3), dtype=_np.uint8)
                arr = (imgs.detach().cpu().numpy()*255.0).clip(0,255).astype('uint8')
                for i in range(N):
                    r, c = i//ncol, i % ncol
                    im = arr[i].transpose(1,2,0)  # HWC RGB
                    grid[r*H:(r+1)*H, c*W:(c+1)*W] = im[:, :, ::-1]  # BGR for imwrite
                tmpdir = tempfile.mkdtemp(prefix='mlflow_aug_')
                outp = os.path.join(tmpdir, 'aug.png')
                _cv.imwrite(outp, grid)
                self.mlflow.log_artifact(outp)
            except Exception:
                pass

    # Text -----------------------------------------------------------------
    def text(self, text: str, win: str = 'vkb_policy', title: str = 'Aug/Reg Policy'):
        if self.viz is None:
            return
        try:
            if 'aug_text' not in self._wins:
                self._wins['aug_text'] = win
                self.viz.text(text, win=win, opts={"title": title})
            else:
                self.viz.text("\n" + text, win=win, append=True)
        except Exception:
            pass
