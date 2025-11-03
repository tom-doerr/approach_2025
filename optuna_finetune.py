#!/usr/bin/env python3
"""
Minimal Optuna driver for DL finetuning.
Searches a tiny space (lr, wd, drop_path, dropout) and maximizes val_acc
recorded by vkb.finetune via the sidecar JSON.

Usage (example):
  .venv/bin/python optuna_finetune.py --data data_small --trials 5 --epochs 1 --device cpu
"""
import argparse, json
from types import SimpleNamespace as NS


def _objective_factory(cli):
    import optuna
    from vkb.finetune import finetune

    def obj(trial: "optuna.trial.Trial") -> float:
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        wd = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
        dp = trial.suggest_float("drop_path", 0.0, 0.30)
        do = trial.suggest_categorical("dropout", [0.0, 0.1])
        args = NS(
            data=cli.data,
            embed_model=cli.embed_model,
            eval_split=cli.eval_split,
            eval_mode="tail",
            epochs=cli.epochs,
            batch_size=cli.batch_size,
            lr=lr,
            wd=wd,
            device=cli.device,
            drop_path=dp,
            dropout=do,
        )
        path = finetune(args)
        with open(path + ".meta.json") as f:
            meta = json.load(f)
        val = float(meta["val_acc"])  # expect present; no fallbacks by design
        return val

    return obj


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--data", default="data_small")
    ap.add_argument("--embed-model", default="mobilenetv3_small_100")
    ap.add_argument("--eval-split", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--study-name", default=None)
    cli = ap.parse_args()

    import optuna
    study = optuna.create_study(direction="maximize", study_name=cli.study_name)
    study.optimize(_objective_factory(cli), n_trials=cli.trials)
    print(f"best_value={study.best_value:.3f}")
    print("best_params=", study.best_params)


if __name__ == "__main__":
    main()

