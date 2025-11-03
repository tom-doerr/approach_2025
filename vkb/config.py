from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass
class Config:
    # Core
    data: str
    embed_model: str
    backbone: str = "mobilenetv3_small_100"
    clf: str | None = None
    # DL
    epochs: int = 0
    batch_size: int = 0
    lr: float = 0.0
    wd: float = 0.0
    device: str = "auto"
    require_cuda: bool = False
    eval_split: float = 0.0
    eval_mode: str = "tail"
    # Aug/Reg
    aug: str = "none"
    brightness: float = 0.0
    warp: float = 0.0
    shift: float = 0.0
    drop_path: float = 0.0
    dropout: float = 0.0
    class_weights: str = "none"
    # Loader
    workers: int = 0
    prefetch: int = 1
    persistent_workers: bool = False
    sharing_strategy: str = "auto"
    # Telemetry
    visdom_aug: int = 0
    visdom_metrics: bool = False
    visdom_env: str = "vkb-aug"
    visdom_port: int = 8097
    rich_progress: bool = False
    # MLflow
    mlflow: bool = False
    mlflow_uri: str | None = None
    mlflow_exp: str = "vkb"
    mlflow_run_name: str | None = None
    # Models
    models_dir: str = "models"


def make_config(args) -> Config:
    # Environment overlays (explicit, minimal)
    visdom_env = os.getenv('VKB_VISDOM_ENV', getattr(args, 'visdom_env', 'vkb-aug'))
    visdom_port = int(os.getenv('VKB_VISDOM_PORT', getattr(args, 'visdom_port', 8097)))
    models_dir = os.getenv('VKB_MODELS_DIR', 'models')
    return Config(
        data=getattr(args, 'data', 'data'),
        embed_model=getattr(args, 'embed_model', 'mobilenetv3_small_100'),
        backbone=getattr(args, 'backbone', 'mobilenetv3_small_100'),
        clf=getattr(args, 'clf', None),
        epochs=int(getattr(args, 'epochs', 0) or 0),
        batch_size=int(getattr(args, 'batch_size', 0) or 0),
        lr=float(getattr(args, 'lr', 0.0) or 0.0),
        wd=float(getattr(args, 'wd', 0.0) or 0.0),
        device=str(getattr(args, 'device', 'auto')),
        require_cuda=bool(getattr(args, 'require_cuda', False)),
        eval_split=float(getattr(args, 'eval_split', 0.0) or 0.0),
        eval_mode=str(getattr(args, 'eval_mode', 'tail')),
        aug=str(getattr(args, 'aug', 'none')),
        brightness=float(getattr(args, 'brightness', 0.0) or 0.0),
        warp=float(getattr(args, 'warp', 0.0) or 0.0),
        shift=float(getattr(args, 'shift', 0.0) or 0.0),
        drop_path=float(getattr(args, 'drop_path', 0.0) or 0.0),
        dropout=float(getattr(args, 'dropout', 0.0) or 0.0),
        class_weights=str(getattr(args, 'class_weights', 'none')),
        workers=int(getattr(args, 'workers', 0) or 0),
        prefetch=int(getattr(args, 'prefetch', 1) or 1),
        persistent_workers=bool(getattr(args, 'persistent_workers', False)),
        sharing_strategy=str(getattr(args, 'sharing_strategy', 'auto')),
        visdom_aug=int(getattr(args, 'visdom_aug', 0) or 0),
        visdom_metrics=bool(getattr(args, 'visdom_metrics', False)),
        visdom_env=visdom_env,
        visdom_port=visdom_port,
        rich_progress=bool(getattr(args, 'rich_progress', False)),
        mlflow=bool(getattr(args, 'mlflow', False)),
        mlflow_uri=getattr(args, 'mlflow_uri', None),
        mlflow_exp=str(getattr(args, 'mlflow_exp', 'vkb')),
        mlflow_run_name=getattr(args, 'mlflow_run_name', None),
        models_dir=models_dir,
    )
