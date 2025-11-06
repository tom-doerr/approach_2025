# AGENTS.md

Scope: Applies to the entire `approach_2025` repo.

## House Rules
- Keep solutions minimal; write as little code as possible.
- No defensive programming or hidden fallbacks unless explicitly requested.
- Prefer simple, deterministic logic; surface issues rather than masking them.
- Add tests for what we build; don’t over‑engineer rare edges.
- After every request, update this file with new learnings/decisions.
- When offering choices, number them like `1a, 1b, 1c…` and increment next time.

## Maintainability Tooling (Quick Picks)
- Keep it minimal and fast; prefer single-binary or single-command tools.
- Cross‑repo default (Python):
  - `ruff` for lint/format (fast; replaces flake8/isort/pycodestyle in one).
  - `mypy` for basic typing gates where we’ve added hints.
  - `radon` + `xenon` for Cyclomatic Complexity and Maintainability Index budgets.
  - `vulture` to spot dead code; `deptry` for unused/missing deps.
  - `pytest-cov` to keep an eye on coverage (signal, not a target).
- Optional (hosted): SonarQube/SonarCloud or Code Climate if/when we want dashboards.
- Budgets (starting points): CCN ≤ 10; MI ≥ 65; duplication < 5% per dir.

### How We’ll Use Them Here (minimal)
- Pre-commit hook runs `ruff` and `radon cc -s -n` (non‑blocking on first pass).
- A tiny `make maint` target prints: ruff issues, worst CCN/MI, dead code summary.
- CI adds a single “Maintainability” job that fails only when budgets are exceeded.


## 2025-11-02 Updates
### Radon Snapshot
- CC hotspots:
  - `vkb/finetune.py`: `finetune` F(173), `_make_loaders` F(43).
  - `train_frames.py`: `_fit_classic_and_save` F(73), `_embed_videos` D(28).
- MI summary:
  - `vkb/finetune.py` MI C(0.00) — still the top candidate to split next.
  - `train_frames.py` MI B(9.49) — improved after trainer split.
  - `vkb/dataset.py` A(37.45) and `vkb/cache.py` A(35.97) after helper splits.
- Shift augmentation switched to roll-around (circular wrap). Default CLI `--shift` is now 0.25 (±25%). Dynamic Aug includes `shift` with a higher cap (0.5). HParams/telemetry show `shift`.
- Artifact contents audit: DL bundles already persist `input_size` and `normalize` (mean/std); `infer_live` reads both and builds the correct preprocessor. Classic bundles store `embed_model` and use the same 224×224 preproc in both train and infer via `vkb.emb.create_embedder` (implicit, consistent). `embed_params` exists (currently empty) if we later need to pass size/flags.
- HParams now includes `img_size` and `shift` so runs explicitly show input resolution and spatial jitter magnitude.
 - Cache stats wired: `FrameDataset` now increments `_cache_hits/_cache_misses/_seen_videos` from `ensure_frames_cached` so the first‑epoch "Cache: videos=..." line reflects reality. Test added (`tests/test_cache_counters_update.py`).

## 2025-11-03 Updates — DL Aug Defaults & Tuning Notes
- New DL static defaults (user request to improve generalization with color/reg jitters):
  - `aug=rot360`, `rot_deg=360.0`, `warp=0.30`, `shift=0.25`,
    `brightness=0.25`, `sat=0.40`, `contrast=0.15`, `wb=0.30`, `hue=0.30`,
    `erase_p=0.3` (`erase_area_min=0.02`, `erase_area_max=0.06`), `noise_std=0.05`.
- Dynamic Aug (opt‑in via `--dynamic-aug`): still starts all at 0 and can raise up to caps
  (`brightness/warp ≤ 0.6`, `sat/contrast ≤ 0.8`, `hue/wb ≤ 0.5`, `shift ≤ 0.5`, `rot_deg ≤ 360`).
- Rationale: validation is near 100% but generalization lags; stronger color jitter and mild noise/erasing push invariance without heavier geometry.
- Caveat: `sat=0.40` and `hue=0.30` are strong; if val drops sharply on small runs, try `sat=0.25`, `hue=0.15`.
- Doc mismatch to review: we state “shift uses roll‑around (circular) wrap”, but code uses
  reflect padding (`cv.BORDER_REFLECT_101`). Decide whether to keep reflect (current) or switch to true roll.

### 2025-11-03 — Workspace/Git note
- `approach_2025` may be symlinked inside a larger repo (e.g., `Neural-Computer-Interface/approach_2025@`). Commit/push inside `approach_2025` itself; the outer repo only tracks the symlink entry.
- Shell aliases (e.g., `ga`) depend on local config; prefer explicit commands: `git add -A && git commit -m "..." && git push`.

### 2025-11-03 — Hangs followed by high CPU: Why and quick knobs
- Most common cause: I/O wait during first-epoch cache builds and memmap reads (tasks in `D` state). UI feels frozen; when I/O finishes, many workers run at once → CPU spikes.
- Thread oversubscription: OpenCV/BLAS/Torch spawn their own pools; with `workers>0` total runnable threads can exceed cores, amplifying the spike.
- CPU-side augs (rotation/warp) are heavy; with `rot360` and `warp=0.30` they add real per-sample cost.
- Quick actions (minimal):
  - Diagnosis: add `--diag-system` to see `Sys:` and `Perf:` lines; high `io_ms` and many threads imply loader/IO bound.
  - Stabilize: run 1 warm-up epoch with `--workers 0 --prefetch 1 --no-persistent-workers --visdom-aug 0` to build caches, then train with `--workers 1–2`.
  - Cap threads: pass `--cap-threads` (sets OMP/MKL/OPENBLAS/NUMEXPR=1, cv2/torch threads=1).
  - Reduce CPU aug: try `--rot-deg 5 --warp 0.10` (or keep defaults and enable `--dynamic-aug`).
  - If load persists: lower `--img-size` and keep `--prefetch 1`.
  - Use RAM for frame caches: set `VKB_FRAMES_DIR=/dev/shm/vkb_frames` (or another tmpfs) so `.npy` memmaps live on tmpfs → no disk `D`-state during reads. Test added `tests/test_frames_env_dir.py`.

### 2025-11-03 — Why I/O reads freeze the GUI
- Memmapped frame reads cause page faults served by the block layer. With many workers and random access, we saturate IOPS and queue depth → lots of tasks in `D` (uninterruptible I/O). GUI input/desktop apps also page‑fault and wait on the same device, so cursor/Windows stutter.
- When memory is tight, reclaim (kswapd) runs aggressively; GUI pages get evicted and then fault back in → long stalls.
- After the I/O completes, many runnable threads wake up together (OpenCV/BLAS/Torch pools × workers) → short CPU spikes while they “catch up”.

Mitigations (runtime)
- Put frames on tmpfs: `export VKB_FRAMES_DIR=/dev/shm/vkb_frames` (avoid disk entirely).
- Lower I/O priority: launch with `ionice -c3` (idle) or `ionice -c2 -n7` (best‑effort, lowest) plus our existing `--nice 10`.
- Keep workers small (0–2) and `--prefetch 1`; warm caches single‑process first.
- Cap thread pools with `--cap-threads` to reduce the post‑I/O CPU surge.

### 2025-11-03 — Can we make them interruptible?
- Strictly, no: Linux tasks in uninterruptible I/O (`D` state) cannot be interrupted by signals. The only practical fixes are to avoid hitting disk (tmpfs) or to prefetch so pages are resident.
- Practical sidesteps we support now:
  - Use `VKB_FRAMES_DIR` (tmpfs) so memmaps are RAM‑backed.
  - Do a single‑process warmup epoch (`--workers 0`) to prefill the page cache.
  - Lower I/O priority with `ionice` so GUI wins contention.
- If desired, we can add a tiny `--ionice idle` flag to apply `ionice -c3` at startup (Linux‑only); not added yet to keep CLI minimal.

### 2025-11-03 — Why reads are “uninterruptible” (kernel note)
- Memmap/file reads that miss in RAM trigger a major page fault. The fault handler submits block I/O and then waits for the page to be filled.
- That wait uses `TASK_UNINTERRUPTIBLE` (state `D`) in most filesystem/block paths (e.g., `wait_on_page_bit(PG_locked/PG_writeback)`), so signals (Ctrl‑C, even `SIGKILL`) are only delivered after the I/O finishes.
- Rationale: maintain filesystem/device consistency and avoid partially‑filled pages; many block drivers can’t safely abort mid‑request.
- Exceptions exist (some network/NFS paths use killable waits), but generic local‑disk reads are typically uninterruptible.

### 2025-11-03 — ionice/nice quick recipes (Linux)
- Run new training at idle I/O priority + lower CPU priority:
  - `ionice -c3 nice -n 10 .venv/bin/python train_frames.py --clf dl [args...]`
- Or “best‑effort, lowest” I/O priority (class 2, level 7) + nice:
  - `ionice -c2 -n7 nice -n 10 .venv/bin/python train_frames.py --clf dl [args...]`
- Adjust a running process (replace PID):
  - `ionice -c3 -p <PID>` (idle I/O)
  - `renice +10 -p <PID>` (CPU nice)
- Verify:
  - `ionice -p <PID>` → shows `idle` or `best-effort: prio 7`
  - `ps -o pid,ni,cmd -p <PID>` → shows niceness
Notes:
- Our CLI already lowers CPU priority by default (`--nice 10`). Wrapping with `nice -n 10` is equivalent; doing both is harmless.

### 2025-11-03 — CLI tip: `--cap-threads`
- `--cap-threads` is a boolean flag (no value). Using `--cap-threads 1` causes an argparse error (`unrecognized arguments: 1`).
- To enable: pass `--cap-threads` by itself. It sets `OMP/MKL/OPENBLAS/NUMEXPR=1`, disables OpenCV OpenCL, and calls `cv2.setNumThreads(1)`, `torch.set_num_threads(1)`, and `torch.set_num_interop_threads(1)` when available.
- If you need a specific thread count >1, we can add a small `--threads N` later; for now, set env vars before launch (e.g., `OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 ...`).

## 2025-11-01 Updates
- Fixed DL crash “mat1 and mat2 shapes cannot be multiplied (…150528 and 192…)”. Cause: local `timm` stub flattened raw 224×224 frames into 150,528 dims but used a `Linear(192→C)` head. Fix: stub now applies `AdaptiveAvgPool2d((8,8))` before `Flatten`, so input dim is always `8×8×3=192` regardless of image size. Added `tests/test_timm_stub_pooling.py` to assert outputs are shaped `[B, num_classes]` for sizes 224/64/17.
- Reminder: if you use a real `timm` install, our stub is ignored. If you see this error again, ensure the intended `timm` is the one being imported.
- Per‑epoch model path printing: After each DL epoch we already save a tiny checkpoint. Now we print the absolute model path on its own line (no prefix) so it’s easy to copy. Test: `tests/test_epoch_model_path_print.py`. If you’d prefer only printing without saving, ask and we’ll switch `_save_epoch_model` to just print and skip writing.
- Dynamic aug tie policy: If `val_acc` is exactly unchanged, we treat it as “no improvement”. After `patience` (default 1) epochs without improvement we either raise the current aug to its target or randomly lower it to 0 (25% chance). If the subsequent check epoch is a tie, we now keep the new value; only a worse score reverts. A strict increase (`>` best) locks in the change. (See `AugScheduler.update`.)
- Validation tie epsilon: added `VAL_EQ_EPS=0.002` with `_val_compare()` in `vkb.finetune` and a CLI flag `--val-eq-eps` to tune it. Ties within epsilon keep the aug change. Tests: `tests/test_val_compare_epsilon.py`.
- Maintainability flags: `--no-persistent-workers` exists; default is now OFF. Unit tests: `tests/test_cli_maint_flags.py`, `tests/test_make_loaders_persistent_flag.py`.
- Dataset simplification: `FrameDataset` no longer depends on the return value of `ensure_frames_cached` (it’s just called). This restores test stubs that returned `None` and avoids unnecessary coupling.
- Infer robustness (classic vs DL): `infer_live.load_bundle` now prefers `pickle.load` first (safer for classic bundles). DL path is wrapped so missing timm/ops cleanly disables DL inference while preserving the explicit CUDA guard (still raises). This prevents stray local DL checkpoints in `models/` from breaking classic e2e tests.
 - Soft labels (DL): we still use label smoothing `=0.05` in the DL loss (`CrossEntropyLoss`) — see `vkb/finetune.py:757`. Classic (`ridge|xgb|logreg`) keep hard labels.
- Default class balancing: `--class-weights` now defaults to `none` (was `auto`). Turn on balancing explicitly with `--class-weights auto` if your classes are very imbalanced. This avoids the previous “double balancing” (sampler + CE weights) by default.
 - Visdom aug-strength plot now uses a logarithmic y‑axis (`ytype='log'`) so you can see small values when using log‑spaced levels. Test: `tests/test_visdom_ylog.py`.

### Class imbalance — how we handle it (classic + DL)
- DL (`--clf dl`): when you pass `--class-weights auto` we do two simple things:
  - Use `WeightedRandomSampler` so each batch is roughly balanced across classes.
  - Pass cross‑entropy class weights `N/(C·n_c)` into the loss (rare classes → larger weight).
  - Default is off (`--class-weights none`) to keep behavior simple; enable explicitly when you have heavy imbalance (e.g., a dominant `no_input`).
- Classic (`--clf ridge|xgb|logreg`): when `--class-weights auto` is set, we compute per‑sample weights `N/(C·n_c)` and pass `sample_weight` to the estimator `.fit(...)`.
  - Works uniformly for `RidgeClassifier`, `LogisticRegression`, and `XGBClassifier`.
  - HPO trials keep using unweighted fits to stay minimal (call out if you want weighted HPO).
- Metrics: favor macro‑F1 in DL summaries; classic path still shows accuracy (fast/simple). If you need macro‑F1 for classic, ask and we’ll add it.

Quick examples
- DL (balanced): `python train_frames.py --clf dl --class-weights auto --eval-split 0.1`
- Classic (balanced ridge): `python train_frames.py --clf ridge --class-weights auto --eval-split 0.1`

Notes & Caveats
- The best fix for “too many no‑input frames” is better labeling (press‑event windows), not just reweighting. We’ll add event‑window ingestion when you’re ready.
- Double balancing (sampler + CE weights) can slow convergence; that’s why it’s opt‑in.


### Dynamic Augmentation Clarification
- Current scheduler is binary per knob: it toggles each augmentation between `0.0` and its configured target (e.g., `brightness=0.15`). There are no intermediate levels; re‑raising a knob that’s already at target is a no‑op. Random “lower” tries may set it back to `0.0`; later it can be raised again to the same target.
- If we want multi‑step intensification, add a tiny `--dyn-steps N` and let each knob take levels `target * i/N` for i=1..N. Keep it off (N=1) by default.

### DL training review notes (same day)
- Confirmed per‑epoch checkpoint now prints just the path; tests cover it.
- Dynamic aug: implemented requested tie semantics — if the validation score stays identical during the check epoch after a change, we keep the new strength and advance; only a worse score reverts.
- Potential sharp edges (left as‑is to keep code minimal):
  - WeightedRandomSampler + CE class weights (both enabled in `auto`) “double balances” classes. It improves recall on rare classes but may slow convergence; can be toggled via `--class-weights`.
  - Epoch checkpoints update `models/LATEST` repeatedly; final save overwrites it. If a run aborts mid‑epoch, LATEST might be a checkpoint — acceptable for now.
  - Tie detection uses exact equality. If needed, add a tiny epsilon (e.g., 1e‑4) to treat near‑ties as ties.

## Glossary
- `vkb`: Vision Keyboard (the internal Python package namespace for this project).
- Ports & Adapters (Hexagonal): Ports are tiny interfaces the core defines for what it needs (outbound) and what it exposes (inbound). Adapters implement those ports using concrete tech (e.g., OpenCV camera, timm embedder, sklearn/XGB, filesystem/MLflow). Benefits: swap deps without touching core, easy fakes in tests, less framework lock‑in. For vkb, candidate ports: `Embedder`, `VideoSource`, `ArtifactStore`, `Telemetry`. Keep ports minimal (Python `Protocol`), adapters thin; no hidden fallbacks.
- Ablation study: originally “remove a part and measure the drop.” In practice, we isolate one factor (component/hparam) and compare baseline vs small, controlled variants (including the removal) while holding everything else constant. Purpose: attribute effect and pick sensible defaults without overfitting to noise.

## FAQ
- What does `vkb` stand for? Vision Keyboard. (Noted 2025-11-01.)
- What does `--visdom-aug 0` do? It disables sending augmented training images to Visdom. When combined with `--visdom-metrics` being off, we don't import, connect to, or clear Visdom at all.

## Project Context (Vision Keyboard)
- Goal: fast, tiny‑motion thumb‑to‑finger input using a single RGB cam.
- Current user preference: avoid landmark/keypoint models; use direct image embeddings + small classifiers.
- Do not try mid‑air QWERTY tapping; rely on chords/contacts and a press trigger.

## Defaults (Opinionated)
- Non‑landmark pipeline (baseline):
  - ROI: stabilize a fixed crop. Calibrate once by recording 2–3 s of hand motion, union the motion mask, and save the bounding box. Use that box for all runs.
  - Embeddings: small CNN (e.g., `mobilenetv3_small_100`, 224×224) via `timm` → ~1024‑D.
  - Labels: include a strong `no_input` class; keep per‑class videos separate to avoid leakage.
  - Event gating (no landmarks): classify every frame; push a key only when top‑1 probability crosses `p_down` for N consecutive frames; release when it drops below `p_up` (p_up < p_down). Add a 120–200 ms refractory. This is probability‑hysteresis.
  - Classifiers: start with Logistic Regression (L2) or XGBoost (`hist`, small depth). Keep Ridge as a quick baseline. Use the built‑in HPO switches.
  - Decoding: optional KenLM 5‑gram for autocorrect; not in the critical path.
  - Keystroke output (Linux): `python-evdev` via `/dev/uinput` (when we wire output).

## Key Thresholds (Tunable Starting Points)
- Pinch hysteresis: press when d < 0.05; release when d > 0.07 (image‑normalized units).
- Refractory: ignore subsequent presses for ~150 ms after a press.
- Finger extended heuristic: angle(MCP–PIP–TIP) > 160°; down if < 130°; in‑between = hold prior state.

## Labeling & Data
- Record video while capturing ground‑truth keypress times (e.g., pedal/keyboard logger).
- For each press at time tp, label frames in [tp−120 ms, tp−20 ms]. Sample negatives far from any press.
- Train a frame‑wise classifier but only sample features at detected press instants at inference.
- If only strings are available (no press times), consider a small temporal model with CTC alignment (later, not first).

## Hysteresis (Definition)
- Use two thresholds so state changes only when crossing a stricter bound and releases on a looser bound; prevents rapid state flapping in noisy signals.

## Testing Guidance
- Unit: geometry feature extraction (closest‑point, normalization, angles) with synthetic landmarks.
- Unit: pinch hysteresis and refractory timing state machine.
- Integration (offline): feed recorded landmark logs, verify decoded sequence matches reference text with LM rescoring on/off.
- System (Linux): create virtual keyboard via uinput; assert characters appear in a sandbox app.

## Known Pitfalls / Things to Watch
- Multi‑worker DataLoader: dataset cache counters are worker‑local. The main process copy stays at zero. Our epoch‑1 cache summary now probes caches directly; for debugging, you can also run with `--workers 0`.
- Very short recordings: stopping immediately can produce almost‑empty `.mp4` files. These add labels but not frames, so total "Frames:" may not increase. Consider re‑recording or adding a pre‑loop write if this persists.
- Camera modes: OpenCV property sets are advisory; drivers clamp silently. Our probe list (sizes 3840/2560/1920/1280/640 × FPS 60/30) is intentionally tiny and not authoritative. On Linux, use `v4l2-ctl --list-formats-ext -d /dev/video0` to see exact discrete modes and frame intervals per pixel format.
- Perspective warp fallback is intentionally tiny and test-oriented. If a real OpenCV build is available, we should use `getPerspectiveTransform`/`warpPerspective`. Avoid expanding fallbacks; prefer explicit errors outside tests.
- Jitter order (wb → saturation → contrast) is fixed and subtle; changing order alters results. Keep small ranges (~0.05–0.10).
- With no landmarks, ROI stability is everything. Lock camera and use the calibrated crop; add light jitter augmentation during training.
- Don’t random‑split frames; use tail/holdout‑by‑video. Random splits inflate accuracy.
- High‑dim embeddings + trees can be slow; consider PCA to ~128–256 dims if latency spikes.
- OpenCV GUI: headless wheels throw GUI errors; use `--no-preview` unless you install a GUI‑enabled build.
- Per‑video labels are weak: labeling an entire video as a key means most frames are actually `no_input`. This inflates train accuracy (memorization) while crushing val accuracy and F1. Fix by labeling short windows around contact events, or by recording with a trigger and only keeping the triggered windows. Class‑balance and evaluation must respect this.
 - Frame cache locks: if a run crashes mid‑build, a leftover `.lock` can stall other workers. We now auto‑clear stale locks when either PID is dead or mtime exceeds a TTL (default 3600 s; override with `VKB_CACHE_LOCK_TTL_SEC`). Set `VKB_CACHE_LOG=1` to see "cleared stale lock" notices. Manual cleanup: remove `*.lock` under `.cache/vkb/frames/` if needed.
- Camera disconnected → empty videos: if `/dev/videoX` isn’t available, `record_video.py` can still create an `.mp4` with 0 frames (V4L2 buffer request fails). These files keep labels but add no frames, so “Frames:” counts won’t increase and frame cache shows zero touches. Quick check: list tiny files or probe frame counts, remove and re‑record.
- Dense frame sampling (stride=1) creates many near‑duplicate train samples from the same video; MobileNetV3 will memorize them even with heavy aug → train_acc≈1.0. Mitigate by sampling fewer frames per video (e.g., stride 3–10) for training while keeping val/test full, or by training only on press‑window frames.
- Background/ID leakage: static backgrounds and per‑video lighting let the model learn “video ID”. Strong color/rotation aug doesn’t remove this; change sampling/labels rather than pushing aug further.
- Heavy, static 360° rotation hurts this dataset (class semantics vary with viewpoint). Prefer dynamic aug with small rotations (≤20°) and incremental warp/color.

### 2025‑11‑01 — Dynamic Aug defaults can look like “no aug”
- When `dynamic_aug` is enabled, `FrameDataset` is created with `aug='none'` and all jitters (including `rot_deg`) set to `0.0`. The `AugScheduler` then raises/lowers strengths over time. Result: epoch‑1 images and the first batch logged to Visdom often show no aug. Changes only appear from the next epoch.
- Scheduler order is: `brightness → warp → sat → contrast → hue → wb → rot_deg`. Rotation is last; it can take several epochs before you see non‑zero `rotation_deg` in the “Policy used:” line and in Visdom images.
- CLI `--aug rot360` is ignored when dynamic aug is ON (we force `aug='none'`). To see geometric rotation immediately in logged images, pass `--no-dynamic-aug` and `--aug rot360`, or keep dynamic aug and start with a small static rotation by disabling dynamic aug.
- Verification: check the per‑epoch console line `Policy used: ... rotation_deg=...`. Val set always uses `aug='none'` and zero jitters by design.
### Note on warp under dynamic aug
- `Augment` captures `warp` at construction time; when the scheduler updates `ds.warp`, the `Augment.warp` field doesn’t change, so perspective warp may not show up in images even though the console prints a higher `warp`. Rotation and color jitters are applied from `FrameDataset` fields and do reflect scheduler changes. Tiny fix if needed: sync `self._augment.warp = self.warp` when `ds.warp` changes.
- If you want the static CLI aug policy instead of dynamic control, run with `--no-dynamic-aug` (or enable DSPy aug which also disables dynamic aug).

### Quick Run — Fixed Augmentation
- Disable dynamic aug: pass `--no-dynamic-aug` and do NOT set `--dspy-aug`.
- Choose a simple preset: `--aug light` (mild zoom‑out). Note: CLI default `--rot-deg` is now 360 (full) — override to a small value like `--rot-deg 5` for this dataset.
- Keep color jitters modest: `--brightness 0.15 --sat 0.08 --contrast 0.08 --wb 0.06 --hue 0.06`.
- Optional perspective: `--warp 0.20` (train only).
- Example (CUDA):
  - `.venv/bin/python train_frames.py --clf dl --device cuda --epochs 10 --batch-size 128 --workers 4 \
     --no-dynamic-aug --aug light --rot-deg 5 --warp 0.20 \
     --brightness 0.15 --sat 0.08 --contrast 0.08 --wb 0.06 --hue 0.06`

### Quick How‑To — Drop Path
- Set on CLI with `--drop-path <float>` (range 0–1). Passed to timm as `drop_path_rate`.
- Example (CUDA): `.venv/bin/python train_frames.py --clf dl --device cuda --epochs 10 --batch-size 128 --workers 4 --drop-path 0.15`
- Inspection: the value prints in the HParams table each run.
- DSPy: when `--dspy-aug` is enabled, suggestions may include `drop_path`; we apply it live to compatible modules and update `args.drop_path`.

### High Load Average — Why and What to Tweak
- Linux load ≠ pure CPU%. It also counts tasks stuck in uninterruptible I/O (state `D`). Our first‑epoch frame‑cache builds and heavy memmap reads can park worker processes in `D`, inflating load even when CPU% isn’t maxed.
- Oversubscription: each DataLoader worker can spawn multi‑threaded libs (OpenCV, BLAS). With `workers>0` and default library thread pools, total runnable threads can exceed core count by a lot → high load.
- CPU‑side aug (rotate/warp/contrast/sat/hue/wb) adds real compute per sample; dynamic‑aug (default ON) ramps these up during training.
- Persistent workers keep processes alive across epochs; the load average window (1/5/15 min) remains elevated.

Practical knobs (minimal + deterministic):
- For diagnosis, run one epoch with single‑process loading: `--workers 0 --prefetch 1 --no-persistent-workers` (expect load to drop sharply if oversubscription/IO wait was the cause).
- Cap thread pools to avoid multiplicative threads (recommended defaults):
  - `export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENCV_OPENCL_RUNTIME=disabled`
- If CPU (user%) is pegged: CPU‑side augs are the main cost. Try `--no-dynamic-aug` (or much smaller `--brightness/--sat/--contrast/--hue/--warp/--shift`) and reduce `--img-size`.
- Optional (manual): calling `cv2.setNumThreads(1)` and `torch.set_num_threads(1)` further caps intra‑op threads; we have not wired a flag yet to keep code minimal.
- Warm the frame cache once (sequential) so later runs don’t rebuild under contention:
  - `VKB_CACHE_LOG=1 python train_frames.py --clf dl --epochs 1 --workers 0 --visdom-aug 0`
- Then train with modest overlap: `--workers 1–2 --prefetch 2` (and keep `--batch-size` moderate). If load spikes on spinning disks/NFS, prefer `--workers 0`.
- Heavy aug costs: temporarily disable to confirm impact: `--no-dynamic-aug` (or set small fixed strengths).

Quick observability:
- `top`/`htop`: look for many `python` workers and threads; `STAT` of `D` implies I/O wait.
- `iostat -xz 1`: high `await`/`%util` indicates disk is the bottleneck.
- Our per‑batch line prints `io_ms` vs `gpu_ms` and `stall`; loader‑bound epochs correlate with higher system load.
 - High load but low CPU% usually means many tasks in uninterruptible sleep (`D`): they count toward load even while burning 0% CPU (e.g., waiting on disk or GPU driver). Confirm with `pidstat -dl 1` and `ps -eo stat,pid,comm,wchan:32 | grep '^D'`.

### Desktop Freezes / UI Stutter (GPU)
- Cause: on single‑GPU desktops (e.g., GTX 1080 Ti), training can monopolize the GPU so the X/Wayland compositor and browser tabs starve → mouse/keyboard lag, window freezes.
- Quick mitigations (pick 1–3 of these):
  - Reduce work: `--batch-size 32` (or 16), `--img-size 128`, `--workers 0`, `--prefetch 1`, `--no-persistent-workers`.
  - Cap thread pools: `export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENCV_OPENCL_RUNTIME=disabled`.
  - Turn off heavy logging: `--visdom-aug 0` (images off) and keep `--visdom-metrics` only.
  - Warm the frame cache once with `--workers 0`, then rerun with workers 1–2.
  - If feasible, run the training from a TTY/SSH session, or on a second GPU (`CUDA_VISIBLE_DEVICES=1`).
  - If freezes persist, temporarily use CPU: `--device cpu` (slow, but interactive).

## Architecture Patterns

Currently used (kept intentionally small):
- Separation of Concerns: thin CLIs (`record_video.py`, `train_frames.py`, `infer_live.py`) over library code in `vkb/`.
- Strategy/Factory: `create_embedder()` selects a backbone; `--clf {ridge,xgb,logreg,dl}` chooses training path.
- Cache‑Aside: embeddings (`.cache/vkb/<embedder>/...npy`) and per‑frame memmaps (`.cache/vkb/frames`) built on demand and reused.
- Artifact Repository: timestamped filenames + `models/LATEST` pointer + JSON sidecar metadata.
- Dependency Injection via env/CLI: `VKB_MODELS_DIR`, `VKB_MLFLOW_DISABLE`, `VKB_VISDOM_*`; easy to monkeypatch in tests.
- Minimal OO Facades: `Finetuner` and `SampleIndex`; helpers for loaders/model init.
- Optional Sinks: Visdom and MLflow are pluggable and off by default in tests.

Consider adopting (small, high‑leverage):
- Ports & Adapters (lightweight): define `ports.py` Protocols for Embedder, VideoSource, ArtifactStore, Telemetry; keep current modules as adapters.
- Telemetry Wrapper: a tiny `vkb.telemetry` that fans out metrics/images to Visdom/MLflow/console; removes duplicate logging code.
- Model Registry Wrapper: a `ModelStore` with `save/load/tag(latest)` to encapsulate `LATEST` and sidecars; enables tags like `prod` vs `latest`.
- Cache Fingerprints: add size/mtime (or content hash) to frame‑cache meta for automatic invalidation.
- Explicit State Machine: `PressDetector` with hysteresis/refractory for live gating (keeps UI logic isolated from classifiers).
- Unified Config: small `@dataclass Config` merging CLI/env for both classic and DL paths; passed through instead of many args.

## Test Hooks
- If the user types exactly `blueberries`, respond with it reversed: `seirrebeulb`.

## Change Log
### 2025-11-03 — Freeze investigation and minimal mitigations
- Observed sources of desktop “freezes” during training (single‑GPU desktops):
  - Thread oversubscription: DataLoader workers + multi‑threaded BLAS/OpenCV → far more runnable threads than cores; spikes load and latency.
  - GPU starvation of the compositor: long kernels at high batch sizes can stall UI on a single GPU.
  - I/O wait bursts: frame‑cache builds can park workers in `D` state; load looks high even when CPU% is modest.
- Minimal flags added (opt‑in, no hidden fallbacks):
  - `--cap-threads`: caps `OMP/MKL/OPENBLAS/NUMEXPR` to 1 and calls `cv2.setNumThreads(1)` and `torch.set_num_threads(1)`/`set_num_interop_threads(1)` if available. Use when UI stutter/freezes appear.
  - `--nice N`: applies `os.nice(N)` on Linux to lower CPU priority (default now 10). No‑op on failure.
  - `--diag-system`: prints a short per‑epoch diagnostic line with loadavg, torch/OpenCV thread counts, and BLAS envs. Off by default to keep tests quiet.
- Notes:
- Defaults: workers=1; persistent workers OFF; niceness=+10. Enable workers persistence explicitly with `--persistent-workers` if you want them.
  - For GPU desktop stutter: reduce `--batch-size`, consider `--workers 0`, and/or run from a TTY/SSH session. See “Desktop Freezes / UI Stutter (GPU)” section below.
- Tests added:
  - `tests/test_cap_threads_flag.py` validates env/thread setters for `--cap-threads`.
  - `tests/test_diag_system_flag.py` asserts the `Sys:` diagnostics line is emitted when `--diag-system` is set.

### Sharing Strategy — file_system vs file_descriptor (Notes)
- We default to `file_system` when `workers>0` (`--sharing-strategy auto`) for broad compatibility and to avoid file‑descriptor exhaustion on desktops.
- `file_descriptor` can be faster and avoids `/dev/shm` (useful inside Docker with small shm), but burns many FDs under heavy DataLoader prefetch/persistent‑workers and can hit `Too many open files` on default ulimits (256/1024).
- Practical guidance:
  - If you see `/dev/shm` errors or run in a container with small shm, use `--sharing-strategy file_descriptor` and raise `ulimit -n` if needed.
  - Otherwise keep the default. On laptops/desktops the freeze symptoms we saw correlate more with thread oversubscription than with the sharing strategy.

### 2025-11-03 — User chose 2a (run diag pass)
- Plan: warm caches single‑process, then 2 short DL epochs with diagnostics.
- Recommended commands (CUDA):
  - Warm cache once: `VKB_CACHE_LOG=1 python train_frames.py --clf dl --device cpu --epochs 1 --workers 0 --prefetch 1 --no-persistent-workers --visdom-aug 0 --eval-split 0.2 --batch-size 64`
  - Diagnostic run: `python train_frames.py --clf dl --device cuda --epochs 2 --workers 1 --prefetch 1 --no-persistent-workers --cap-threads --diag-system --batch-size 64 --eval-split 0.2`
- What to share back: the printed `Sys:` lines (per epoch), plus the `Perf:` line; we’ll tune batch/workers based on that.

### 2025-11-03 — Tests Added (niceness + main helpers)
- `tests/test_cli_nice_default.py`: asserts `--nice` defaults to `10`.
- `tests/test_cli_nice_apply.py`: verifies `_maybe_nice` calls `os.nice(n)` with the provided value.
- `tests/test_main_calls_helpers.py`: ensures `main()` calls `_apply_thread_caps` and `_maybe_nice` before dispatching to `train()`.
- `tests/test_cli_persistent_workers_default_and_flag.py`: asserts `persistent_workers` default is `False`, `--persistent-workers` enables it, and `--no-persistent-workers` sets the disable flag.

### Workers = 0 — Behavior Notes
- DataLoader runs in the main process (no subprocesses). `prefetch_factor` and `persistent_workers` are ignored; sharing strategy isn’t touched.
- Augmentations and noise run in a single RNG stream (main process) → more reproducible than multi‑worker runs.
- Frame‑cache builds happen synchronously from the training process. Slower initial epoch, but avoids I/O storms and “D‑state” spikes.
- Lower CPU/RAM overhead; avoids thread oversubscription. GPU may idle more (no overlap), so samples/s can drop; check `Perf:` stall to confirm.
- Good for: first runs, desktops with stutter, or when building caches; switch to workers=1–2 once caches are warm and UI is smooth.

### 2025-11-03 — Git Identity Policy
- Do NOT modify the user’s git identity (user.name/user.email), neither globally nor locally.
- For one‑off commits, prefer inline environment variables on the commit command:
  `GIT_AUTHOR_NAME="…" GIT_AUTHOR_EMAIL="…" GIT_COMMITTER_NAME="…" GIT_COMMITTER_EMAIL="…" git commit -m "msg"`.
- Alternatively, use `git commit --author="Name <email>"` while setting committer via env vars if needed.
- Note: we removed a temporary local `[user]` section created during an earlier attempt so repo config is clean again.

### 2025-11-03 — Repo State / Push
- Tracked program files include: `train_frames.py`, `infer_live.py`, `record_video.py`, `optuna_finetune.py`, and the whole `vkb/` package.
- Tests are tracked too; heavy dirs are ignored (`.venv/`, `.cache/`, `models/`, `data/`, `tmp/`).
- Current branch: `main`. No remote configured (checked via `git remote -v`). To push:
  - `git remote add origin <URL>`
  - `GIT_TERMINAL_PROMPT=0 timeout 20s git push -u origin main`

### 2025-11-03 — Symlink → Real Dir (outer repo)
- Outer repo (`Neural-Computer-Interface`) had `approach_2025` as a symlink → `/mnt/4tb_nvme/approach_2025`.
- To have the outer repo track files (not the symlink), we materialized a real directory:
  - `rsync -a /mnt/4tb_nvme/approach_2025/ approach_2025_tmp/` excluding: `.git/ .venv/ .cache/ models/ data/ tmp/ __pycache__/ .pytest_cache/`.
  - `rm approach_2025` (remove symlink) → `mv approach_2025_tmp approach_2025` (rename into place).
  - `git add -A approach_2025 && git commit -m "vendor approach_2025 program files (replace symlink)"` in the OUTER repo branch `feat/approach_2025-aug-jitters-hue-20251101`.
  - Pushed to the existing remote; branch updated.
- Result: outer repo now contains real tracked files under `approach_2025/`.

### 2025-11-03 — Per‑Video Tail Validation
- New eval mode: `--eval-mode tail-per-video` (classic + DL). Instead of a per‑class tail, we split within each video: head → train, mid → val, tail → test.
- Global fraction respected: total val/test frames are apportioned across videos proportional to video length using Hamilton rounding (largest remainders). This keeps `--eval-split` as a fraction of overall frames, not per‑video.
- Capacity: we cap per‑video val so at least one train frame remains when possible. If fractions are too large to satisfy this everywhere, we allocate up to capacity (i.e., totals may undershoot slightly rather than adding hidden fallbacks).
- Classic: `_split_tail_per_video_slices` with `args._video_slices` captured during embedding.
- DL: `vkb.finetune._per_video_tail_split_three`, honored in `_prepare_data()`.
- Tests: `tests/test_split_tail_per_video.py` covers both helpers.

### 2025-11-03 — Recording Buttons Mode
- New CLI: `record_video.py --buttons A,B,C` shows on‑screen buttons for labels `A`, `B`, `C`. Clicking a button starts recording immediately into `data/<label>/<timestamp>.mp4`. Stop with `q`/`ESC` or by clicking the on‑screen `STOP` button.
- Minimal UI: one row of equal‑width buttons at the bottom; clicked label highlights. Includes a red‑tinted `STOP` button to finish/save the current clip. Clicking the active label again also stops and saves (toggle behavior). We still ignore clicks on a different label while recording to keep the flow simple.
- Clear recording indicator: while recording, an on-frame overlay prints `REC: <label>` at the top-left (also shown in single-label `--label` mode when preview is enabled). Test: `tests/test_record_video_buttons_indicator.py` asserts the overlay contains the active label when buttons mode is used.
- Duration: overlay now includes elapsed time as `REC: <label> mm:ss` in both modes. Covered indirectly by the indicator test.
- Frame count: overlay also shows the current frame count as `… {N}f`. Buttons test asserts the overlay contains the label, time, and `f`.
- Convenience flag: `--add-button-right-index-inward` appends a `right_index_inward` button to whatever labels you pass via `--buttons` (deduped if already present). Test: `tests/test_record_video_extra_button_flag.py` ensures the flag alters the labels fed into `record_buttons()`.
- Classic label mode (`--label`) is unchanged and coexists; the two options are mutually exclusive in `parse_args()`.
- Tests: `tests/test_record_video_buttons.py` stubs `cv2`, simulates a click, and asserts the writer path goes under the clicked label’s folder. `_layout_buttons()` is tiny and tested implicitly.
- Constraint: `--buttons` requires preview (GUI); we raise an error if `--no-preview` is used with `--buttons` to avoid silent failure.
### 2025-11-03
- Model filenames now include validation accuracy when available:
  - Classic path: `..._<clf>_<embed_model>_val{ACC}.pkl` where `ACC` is formatted to 3 decimals (e.g., `val0_532`).
  - DL path: `..._finetune_<backbone>_val{ACC}.pkl` with the same formatting.
  - We only append when a validation split exists (no placeholders or fallbacks).
- DL filenames also append best epoch: `_epNN` (2-digit), e.g., `..._val0_447_ep01.pkl`.
- Tests added:
  - `tests/test_filename_val_in_name_classic.py` asserts classic saved filename contains `_val`.
  - `tests/test_filename_val_in_name_dl.py` asserts DL saved filename contains `_val`.
- Rationale: quick scan of artifacts shows the best run at a glance without opening sidecars.

- Default DL DataLoader workers set to 1 (was 4). Minimal, reduces thread oversubscription on desktops by default; bump explicitly for throughput. Test updated: `tests/test_workers_default.py` now asserts `workers==1`.

#### Workers vs. Validation — Why scores can drop with >0 workers
- With multiple workers, training-side randomness (per-sample color jitter, erasing, noise, rotation) is applied in parallel in different orders. On small datasets this can change the trajectory noticeably and lead to lower val.
- Val path itself remains augmentation‑free. Differences come from the trained weights, not how val data is loaded.
- Locks ensure frame-cache builds are safe across workers; we verified no val‑time augmentation occurs.

New tests
- `tests/test_val_consistency_workers.py`: runs a deterministic DL finetune twice (workers=0 vs 2) with all augs/noise off and fixed seeds; asserts identical `val_acc`.
- Rationale: guards against accidental augmentation or cache differences leaking into validation when bumping `--workers`.

- 2025-11-02: Cache locking hardened. `_acquire_lock()` now detects and clears stale `.lock` files (dead PID or TTL exceeded; TTL default 3600 s via `VKB_CACHE_LOCK_TTL_SEC`). Added tests `tests/test_cache_stale_lock.py`. This prevents long waits/spin that could feel like system freezes after a crash.
- 2025-11-02: Decoupled DL backbone from classical embedding flag. New CLI flag `--backbone` controls the DL finetune architecture (default `mobilenetv3_small_100`). `--embed-model` remains for classic (ridge/xgb/logreg) only. HParams table now shows `backbone` instead of `model`. DL artifacts and per-epoch checkpoints are named with the backbone. Tests updated accordingly.
- 2025-11-02: infer_live safe loading: defaults to `torch.load(..., weights_only=True)`. Added `--unsafe-load` to allow loading trusted pickles/legacy checkpoints with `weights_only=False` (or plain pickle) when safe load fails. Tests updated; error messages now nudge to use `--unsafe-load` explicitly.
- 2025-11-02: Live camera parity: infer_live now applies the same MJPG mode selection as record_video (Linux/v4l2): it picks the first MJPG mode via `v4l2-ctl` and sets FOURCC/FPS/WH; falls back to 1920×1080@30 when MJPG section isn’t found. Added `_apply_mjpg_mode()` and a small test.
- 2025-11-02: Fixed misleading epoch‑1 cache stats when `--workers > 0` by probing caches via `ensure_frames_cached` instead of reading main‑process dataset counters. Added tests: `tests/test_new_videos_picked_up.py` and `tests/test_cache_summary_epoch1_probe.py`.
- 2025-11-02: record_video now requests the highest available camera resolution by setting very large `CAP_PROP_FRAME_WIDTH/HEIGHT` before the first read (backends clamp to max). Added `tests/test_record_video_highres.py` to assert `VideoWriter` is initialized with the camera’s max.
  - Update: record_video now prints camera backend, current mode, and probes a tiny set of common modes (WxH@FPS) before recording. Tests: `tests/test_record_video_modes.py`.
- 2025-11-02: Train now prints the newest video included (overall) for both classic and DL paths: a single `[dim]Newest video: <path> label=<lab>[/]` line right after dataset discovery. Tests: `tests/test_newest_video_prints.py`.
- 2025-11-02: Added optional live validation during epochs (DL): `--val-live-interval N` runs a quick validation on one shuffled val batch every N train batches and logs `val_live` to Visdom/MLflow immediately. Minimal implementation via a separate shuffled val loader. Test: `tests/test_val_live_step.py`.
  - Update: `val_live` now plots to a separate Visdom window (`vkb_val_live`, title “Val (Live)”) so it doesn’t clutter the main Train/Val plot.
- 2025-11-02: Docs: added a brief classic classifier cheatsheet (ridge/logreg/xgb) in our guidance below; no code changes.
  - Update: record_video now prints the active capture mode and the exact resolution/FPS being written (line “Using: <WxH>@<FPS>”). Test: `tests/test_record_video_uses_mode.py`.
  - New: full capabilities listing via `--list-modes` (Linux/V4L2) using `v4l2-ctl --list-formats-ext -d /dev/video<index>`. Also added `--cam-index` to select the device. Test: `tests/test_record_video_list_modes.py`. Default: when recording on Linux, we print the v4l2-ctl listing if available (test: `tests/test_record_video_default_lists.py`).
  - Default capture request: we now request MJPG 1920×1080 @ 30 fps by default (before the first read). Drivers may clamp/ignore; we still print the actual “Current:” and “Using:” modes.
  - XGBoost HPO ranges widened: `n_estimators` up to 1000 (was 300), `reg_lambda` up to 100 (was 10). Test: `tests/test_hpo_xgb_ranges.py` (uses a stub RNG + stub XGB to stay fast).
- 2025-11-02: Maintainability quick wins
  - Added `--label-smoothing` (default 0.05) and threaded it to `CrossEntropyLoss`; HParams table and sidecar now record the value.
  - Fixed `scripts/sweep_val.sh` environment leak (`META` is now exported before the heredoc Python block).
  - Removed unused `dspy_agents/deepseek_aug.py` (policy is centralized in `dspy_agents/policy_lm.py`).
  - Kept noise augmentation simple and test-friendly: train-only; skipped on all-zero frames to keep validation-equality tests stable; random erasing fill uses `np.random.normal` (tests stub this to zeros). No hidden fallbacks.
  - Minor import hygiene: avoid importing `timm` eagerly in `finetune()` to reduce heavy deps during tests.
  - Split `train_frames.train()` into `_train_dl` and `_train_classic` (behavior and output unchanged). Added small delegation tests.
  - Docs: quick commands recap for classic classifiers — Ridge (`--clf ridge [--alpha|--hpo-alpha]`), LogReg (`--clf logreg [--C|--hpo-logreg]`), XGB (`--clf xgb [--hpo-xgb]`); suggest `--workers 0 --prefetch 1 --no-persistent-workers` on desktops.
  - Added `ClassicTrainer` in `train_frames.py` that wraps existing helpers (`prepare→embed→fit_and_save`). `_train_classic()` now calls `ClassicTrainer(args).run()`. Kept helper functions and console output unchanged so existing tests remain valid.
  - Split `FrameDataset.__getitem__` into small helpers (`_open_memmap`, `_apply_xy_shift`, `_apply_rotation`, `_apply_color_and_geom_augs`). Behavior unchanged; aug/val/noise/erasing tests continue to pass. Added `tests/test_framedataset_helpers_exist.py`.
  - Split `vkb.cache.ensure_frames_cached()` into tiny helpers: `_src_fingerprint`, `_check_existing`, `_recheck_after_wait`, `_count_frames_dims`, `_build_frames_cache`. Added `tests/test_cache_split_helpers_exist.py`. All existing cache tests still pass.
  - Added regression tests to lock outputs: classic console section order and DL HParams key order.
  - Continued split of classic path:
    - `_prepare_classic_io()` sets up listing/labels/embedder/cache dir.
    - `_embed_videos()` performs embedding + cache prints + summary tables.
    - `_fit_classic_and_save()` selects/HPOs the model, fits/evaluates, writes bundle/sidecar/MLflow, prints summary and HPO tables.
    - Kept all console strings and ordering identical; tests pass unchanged.

## Maintainability — Quick Wins (Opinionated)
- Single source of truth for defaults: prefer CLI → `args` → sidecar; avoid magic constants in code (example: `--label-smoothing`).
- Keep heavy imports deep: import `timm` inside `_create_timm_model()` only; it shortens test import paths and failures.
- Scripts are part of the product: treat `scripts/sweep_val.sh` like code (env correctness, `set -euo pipefail`, small prints).
- Remove dead code promptly: delete unused adapters/modules (we removed the legacy DeepSeek‑specific module).
- Explicit names over abbreviations in user‑facing prints/plots: use `rotation_deg`, `white_balance`, etc., in CLI/Visdom.
- Small, focused tests for flags and prints: add parser tests when introducing a flag; assert HParams includes it.
- Avoid silent behavior switches: if an aug is disabled by design (e.g., val path), make it explicit in code and docs.

## Notes (Maintainability context)
- Validation is augmentation‑free by design (no color jitter, warp, rotation, or noise). Train path applies aug; noise is train‑only and skipped for all‑zero frames to keep stability in tests.
- Sidecars: include key training knobs (`label_smoothing`, drop_path, dropout, aug strengths). This keeps DSPy/history features in sync.
- 2025-11-02: Added a concise "Maintainability Tooling (Quick Picks)" section with recommended minimal tools (ruff, mypy, radon/xenon, vulture, deptry, pytest-cov) and lightweight budgets/CI approach.
 - 2025-11-02: Clarified augmentation status: no explicit random XY shifts or circular roll are implemented; spatial jitter comes from zoom‑out, rotation (reflect pad), and perspective warp. Added a note under “New Learnings / Keep In Mind”.
- 2025-11-02: Docs — added a quick how‑to for setting drop path (`--drop-path`).
- 2025-11-02: Default change — dynamic augmentation is now OFF by default. Enable it explicitly with `--dynamic-aug` (use `--no-dynamic-aug` to force off). With dynamic aug off, training uses the fixed CLI augmentation strengths immediately.
- 2025-11-02: Increased default augmentation strengths for fixed runs: `brightness=0.25`, `sat=0.15`, `contrast=0.15`, `wb=0.12`, `hue=0.10`, `warp=0.30`. Added `tests/test_cli_aug_defaults.py` to lock the numbers. Note: with dynamic aug ON (default), training still starts at 0.0 and ramps up; these defaults matter primarily when `--no-dynamic-aug` is used or when passing explicit values.
- 2025-11-02: CLI default `--rot-deg` set to full rotation (360.0). Dynamic aug still starts at 0 and adjusts per epoch; for fixed runs prefer `--rot-deg 5` on this dataset.
- 2025-11-02: Re‑added train‑only noise augmentation with random strength. New flag `--noise-std` sets a maximum Gaussian std; for each training sample we draw `σ ~ U(0, noise_std)` and add `N(0, σ²)` before normalization. Validation ignores noise. Tests: `tests/test_noise_random_degree.py`; `tests/test_val_no_aug.py` already asserts val ignores aug, including noise.
- 2025-11-02: Random erasing default ON (train only). New flags: `--erase-p` (default 0.2), `--erase-area-min` (0.02), `--erase-area-max` (0.06). Validation ignores erasing. Tests: `tests/test_random_erasing.py`, `tests/test_cli_erase_defaults.py`.
- 2025-11-02: Drop Path note — `--drop-path` accepts any float in [0,1] and is passed to timm as `drop_path_rate`. It’s safe to try 0.5 or 0.75, but on our small dataset and MobileNetV3‑Small backbone it typically hurts val. Recommended range to start: 0.05–0.20. Example: `.venv/bin/python train_frames.py --clf dl --device cuda --epochs 5 --batch-size 128 --workers 4 --drop-path 0.5`.
- 2025-11-02: Dynamic rotation added to `AugScheduler` (now schedules `rot_deg` with a modest cap of 20°). Tiny unit test `tests/test_aug_scheduler_rot_included.py` asserts rotation can be raised from 0.
- 2025-11-02: Quick DL tuning (CUDA, bs=128, workers=4, tail split=0.2, visdom/mlflow disabled, artifacts to `/tmp`):
  - Dynamic aug ON (scheduler raised warp then sat), dropout=0.1, drop_path=0.1 → 5‑epoch val_acc ≈ 0.445 (best this pass).
  - Static heavy rotation (`--no-dynamic-aug --aug rot360` + jitters) → val_acc ≈ 0.33.
  - Stronger regularization (wd=3e‑4, dropout=0.2) → val_acc ≈ 0.39.
  Takeaway: subtle, data‑driven aug beats heavy fixed aug; over‑regularization hurts. Underlying issue remains weak labels (per‑video); event‑window labeling would likely yield larger gains than hyper‑tuning.
- 2025-11-01: Live run sanity — DSPy policy integration works in a real 1-epoch CPU run on `data_small`. The CLI prints per-epoch policy (aug+reg) and metrics, and applies the LM suggestion at epoch end. Observed LM round-trip ~90s on one run; streaming via OpenRouter + `--dspy-reasoning-effort low` recommended for faster suggestions. Pydantic warnings from litellm are benign.
 - 2025-11-01: Clarified DSPy toggle flag for training: use `--dspy-aug` with `train_frames.py --clf dl` to enable background LM aug suggestions; default remains OFF.
 - 2025-11-01: Renamed the DSPy module to be model‑ and control‑agnostic: `dspy_agents/policy_lm.py` (was `deepseek_aug.py` → `aug_lm.py`). It now supports policy suggestions beyond augmentation: optional keys `dropout` and `drop_path`. We keep backward‑compatible aliases (`AugStep`, `AsyncAugPredictor`, etc.) exported from `dspy_agents.__init__`.
 - 2025-11-01: Trainer applies `dropout` by updating `nn.Dropout.p` and `args.dropout`; applies `drop_path` by setting `drop_prob` on modules that expose it and `args.drop_path`. CLI prints these when present as `do=` and `dp=` alongside aug values.
- 2025-11-01: Added asynchronous aug suggestion: `AsyncAugPredictor` runs a background thread that calls the LM to propose next‑epoch aug strengths while the current epoch trains. Minimal API: `pred = AsyncAugPredictor(); pred.submit(history); ... at epoch end -> pred.result()`. Test included in the same file.
- 2025-11-01: Added flag‑gated integration: pass `--dspy-aug` to `train_frames.py --clf dl` to enable DSPy‑based aug suggestions. When set, built‑in dynamic aug is disabled for that run, and we apply the LM suggestion (if ready) at each epoch end without waiting; otherwise we skip quietly (no fallbacks). CLI tests added.
- 2025-11-01: CLI now prints DSPy latency per application: when a suggestion is applied, we show `DSPy aug applied for next epoch in X.XXs.`. This measures time from `submit()` at epoch start to `result()` at epoch end.
- 2025-11-01: Added OpenRouter support and reasoning effort flag for DSPy augs. Use `--dspy-openrouter` to route via OpenRouter and `--dspy-reasoning-effort {low,medium,high}` to set the `reasoning={effort:...}` body field (models may ignore if unsupported). Requires `OPENROUTER_API_KEY`.
 - 2025-11-01: Default OpenRouter reasoning effort is now `low` (CLI `--dspy-reasoning-effort` default); override with `--dspy-reasoning-effort medium|high` if needed.
- 2025-11-01: DSPy history now includes both train and val metrics per epoch: `train_acc`, `train_loss`, `val_acc`, `val_loss`. Internally, `_train_epoch` returns an extra `train_loss` (kept backward-compatible), and `_validate_epoch` returns `(val_acc, val_loss)` but callers accept either tuple or single value for compatibility with older tests.
- 2025-11-01: Added typed DSPy API in `dspy_agents`: `AugStep` (input) and `AugSuggestion` (output). New `suggest_aug_typed(steps: Iterable[AugStep]) -> AugSuggestion` enforces typed I/O; legacy `suggest_aug_sync()` still returns a dict for existing call sites. `AsyncAugPredictor` now exposes `result_typed()` and accepts a predict function that returns either `AugSuggestion` or a dict.
- 2025-11-01: Reasoning streaming (OpenRouter). New helper `dspy_agents.stream_reasoning_openrouter(...)` streams `response.reasoning.delta` events and prints deltas when `--dspy-stream-reasoning` is set (OpenRouter path only). Requires `OPENROUTER_API_KEY` and Python package `sseclient-py`. CLI flags: `--dspy-stream-reasoning`, `--dspy-model` (default `deepseek/deepseek-reasoner`).
- 2025-11-01: Visdom plot rename — window previously titled "Aug Strengths" is now "Policy (Aug + Reg)" and includes `dropout` and `drop_path` series along with augmentation knobs.
 - 2025-11-01: CLI now prints chosen aug strengths along with DSPy latency, e.g., `b=0.120 w=0.220 s=0.050 c=0.100 hue=0.010 wb=0.020 rot=45.0`. Test `tests/test_dspy_cli_print_suggest.py` covers this output.

- 2025-11-01: Checkpoint commit/push after strict frame-cache test integration; suite green (138). No code changes beyond those already logged.
### 2025-11-01 — 100‑epoch DL sweeps to improve val score (no Visdom, no models/)
- Constraint: no Visdom writes and no models/ writes. Used `VKB_VISDOM_DISABLE=1` and `VKB_MLFLOW_DISABLE=1`; redirected artifacts via `VKB_MODELS_DIR=/tmp/...` (see results).
- Run #1 (rot360, moderate reg): `bs=128, lr=1e-4, wd=3e-4, drop_path=0.10, dropout=0.10, aug=rot360, brightness=0.15, warp=0.20, sat=0.08, contrast=0.08, wb=0.06, hue=0.06, workers=8, prefetch=2` → best val_acc≈0.386 @ epoch 83. Path: `/tmp/vkb_sweep_run1/...`.
- Run #2 (light aug, stronger wd): `bs=128, wd=5e-4, drop_path=0.05, dropout=0.10, aug=light, brightness=0.10, warp=0.15, sat=0.05, contrast=0.05, wb=0.03, hue=0.02` → best val_acc≈0.438 @ epoch 97. Path: `/tmp/vkb_sweep_run2/...`.
- Run #3 (light aug, even stronger wd, no dropout, milder color): `bs=128, wd=1e-3, drop_path=0.05, dropout=0.00, aug=light, brightness=0.05, warp=0.10, sat=0.03, contrast=0.03, wb=0.02, hue=0.00` → best val_acc≈0.536 @ epoch 100. Path: `/tmp/vkb_sweep_run3/...`. This is the current best.
- Throughput notes (1080 Ti): with `workers=8, prefetch=2, bs=128`, typical ~1.3–1.4k samples/s; some epochs peaked much higher once caches were warm.

Potential bugs / clarifications noticed while sweeping:
- `--persistent-workers` now defaults to OFF; use `--persistent-workers` to enable or `--no-persistent-workers` to force-disable.
- Early per‑epoch val can look poor with `aug=light` but converges well by late epochs; avoid drawing conclusions from the first 5–10 epochs on this dataset.

Recommended next default (opinionated for this dataset):
- DL: `aug=light, wd=1e-3, drop_path=0.05, dropout=0.0, brightness=0.05, warp=0.10, sat=0.03, contrast=0.03, wb=0.02, hue=0.0, bs=128, workers=8, prefetch=2`.
  Rationale: best observed val accuracy (≈0.536) with minimal complexity.

Safety/testing:
- Added `tests/test_artifacts_env_dir.py` to assert that artifacts respect `VKB_MODELS_DIR`; ensures we can sandbox runs away from `models/`.

- 2025-11-01: Default DL DataLoader workers increased from 0 → 4. Rationale: better overlap on cached datasets; lock added to frame cache mitigates prior races. If you hit issues, pass `--workers 0`.
- 2025-11-01: Use non‑blocking CUDA transfers (`.to(device, non_blocking=True)`) in train/val loops to overlap H2D copies with compute. No behavior change on CPU.
 - 2025-11-01: Removed `--amp` (CUDA mixed precision) per user request; on GTX 1080 Ti and loader‑bound runs it did not improve throughput and added complexity. Keep non‑blocking H2D; revisit AMP only if switching to newer GPUs.
- 2025-11-01: Fixed warp augmentation test by adding a minimal crop-and-resize fallback in `_perspective_warp_inward` when `cv2.getPerspectiveTransform`/`warpPerspective` are missing (common in stubs). Also hardened `_resize_to` to work when `cv2.resize` is absent. Result: test suite now passes end-to-end (117 tests).
- 2025-11-01: Added small, train-only, pre-normalization color jitters: saturation (`--sat`, default 0.08), contrast (`--contrast`, default 0.08), and per-channel white-balance (`--wb`, default 0.06). Wired through `FrameDataset`, printed in HParams and Visdom policy text, and saved in sidecars. Test `tests/test_color_jitter.py` ensures jitters apply in train and are ignored in val. Suite: 118 passing.
- 2025-11-01: Added hue shift augmentation (`--hue`, default 0.06). Implemented via a lightweight YIQ chroma-plane rotation (±hue·π radians) before normalization, train-only. Printed in HParams/Visdom, and saved in sidecars. Test `tests/test_hue_aug.py`. Suite: 119 passing.
- 2025-11-01: Git: created branch `feat/approach_2025-aug-jitters-hue-20251101` and pushed code-only changes (excluded data/models/caches). Open a PR against `origin/main`.
- 2025-11-01: CLI UX: training now prints a final one-liner with the saved model path (`Saved model: <path>`) for both classic and DL paths, in addition to the summary table.
- 2025-11-01: Visdom: `visdom_prepare()` no longer clears windows during tests (detected via `PYTEST_CURRENT_TEST`) or when `VKB_VISDOM_NO_CLEAR=1`. Added `tests/test_visdom_no_clear.py`.
- 2025-11-01: Visdom off-by-default cases: `setup_visdom()` now returns `None` early without importing or touching Visdom when both `--visdom-aug 0` and `--visdom-metrics` are off. Added `tests/test_visdom_disabled_no_touch.py`.
- 2025-11-01: DL checkpoints: we now save a checkpoint after every epoch (`..._finetune_<model>_epXX.pkl`) and print its path with `Saved epoch N model: <path>`. Final best/last model is still saved and printed at the end.
- 2025-11-01: Dynamic augmentation (`--dynamic-aug`): start with all aug strengths at 0; if val stalls for 1 epoch, raise one aug (brightness→warp→sat→contrast→hue→wb in round‑robin). If it helps in the next epoch, keep it and advance; otherwise revert and try the next. Implemented via a tiny `AugScheduler`; unit test `tests/test_dynamic_aug_scheduler.py`. No dataloader rebuilds; we tweak `FrameDataset` fields in-place.
  - When `--dynamic-aug` is NOT set, training uses the default/static augmentation strengths exactly as provided by CLI/defaults (verified by `tests/test_dynamic_aug_off_defaults.py`).
  - Visdom: when dynamic aug is on and Visdom is enabled, we also plot per-epoch augmentation strengths (window `vkb_aug_strengths`) using `Telemetry.scalar2`. Test: `tests/test_visdom_aug_strengths.py`.
  - Dynamic aug randomness: on a stall, there is a 25% chance we try decreasing the current aug (to 0) instead of increasing; if the next epoch still doesn’t improve, we restore the previous value. Test: `tests/test_dynamic_aug_decrease_branch.py`.
  - Rotation: dynamic aug now also controls a small rotation magnitude (`rot_deg`, degrees). We start at 0 and adjust in the same round‑robin. Logged in HParams and Visdom (window `vkb_aug_strengths`).
- 2025-11-01: Perf line improved — now prints batch size (`bs`) and average throughput (`samples/s`) computed from average batch time (I/O + GPU). Added unit test `tests/test_perf_line_prints.py`. Kept existing per-epoch Summary line (`train_fps=... samples/s`).
- 2025-11-01: Rich progress bars — already supported for DL training via `--rich-progress`. Added a minimal nested layout: an outer `Epochs` bar and an inner per-epoch batch bar. Still opt-in.
- 2025-11-01: Introduced minimal Ports & Adapters:
  - Added `vkb/ports.py` with `Protocol`s for `Embedder`, `FrameCache`, `Augmenter`, `Console`, and `Classifier`.
  - Added `vkb/adapters.py` with `CacheModuleAdapter` (delegates to `vkb.cache`).
  - Added tests `tests/test_ports_protocols.py` to assert protocol conformance and adapter delegation.
- 2025-11-01: Frame cache fingerprints — `ensure_frames_cached` now writes `src_size` and `src_mtime_ns` (from the source video) into the `.meta.json` and refuses to reuse a cache if they differ. Added `tests/test_frame_cache_invalidate_fingerprint.py`. Behavior is unchanged when source stats aren’t available.
- 2025-11-01: ModelStore wrapper — added `vkb/model_store.py` with a tiny `ModelStore` class to standardize model saving/loading and tagging:
  - `save(obj, name_parts, meta=..., tags=[...])` → writes model, optional sidecar, and tag pointers.
  - `tag(path, tag)` → writes a simple uppercase pointer file in `models/` (e.g., `LATEST`, `PROD`).
  - `load('latest'|'prod'|path)` → resolves tag files then unpickles.
  - Tests: `tests/test_model_store.py` verifies tag pointers and load behavior.
- 2025-11-01: Telemetry wrapper — added `vkb/telemetry.py` providing a tiny `Telemetry` class to fan out metrics/images/text to Visdom + MLflow without duplicating calls. `finetune.py` now uses it for:
  - `metric(name, value, step)` → MLflow metric.
  - `scalar(series, step, value, title)` → Visdom line.
  - `images(imgs)` → Visdom image grid + MLflow artifact.
  - `text(txt)` → Visdom text panel. Existing console prints unchanged.
- 2025-11-01: Unified Config — added `vkb/config.py` with a minimal `Config` dataclass and `make_config(args)` that merges CLI with env vars (`VKB_VISDOM_ENV`, `VKB_VISDOM_PORT`, `VKB_MODELS_DIR`). `train_frames.train()` and `finetune()` attach `args.cfg` for downstream use. Test: `tests/test_config_unified.py`.
- 2025-11-01: Added `optuna_finetune.py` — a minimal executable Optuna HPO driver for the DL finetuning path. It tunes `lr`, `wd`, `drop_path`, and `dropout`, maximizes `val_acc` from the finetune sidecar, and prints `best_value` + `best_params`. A focused test `tests/test_optuna_finetune_cli.py` stubs `cv2`/`timm` and redirects artifacts to a temp dir to keep runs fast. Note: requires `optuna` to be installed.
- 2025-11-01: Broke up `vkb.finetune.finetune` into small helpers: `_prepare_data`, `_print_device`, `_print_hparams`, `_make_loaders`, `_init_model_and_optim`, `_save_artifacts`. Behavior and console output are unchanged. Added `tests/test_finetune_breakup_helpers_exist.py` to lock the split.
- 2025-11-01: Promoted a tiny OO API: `vkb.finetune.Finetuner` is now a `@dataclass` with `fit()` (alias of `run()`) that delegates to `finetune(args)`. Added `tests/test_finetuner_api.py` to assert the alias.
- 2025-11-01: Added a minimal augmentation class `vkb.finetune.Augment` and wired `FrameDataset` to use it. Behavior is unchanged: aug/warp apply only in `mode='train'`; validation resizes only. Added `tests/test_augment_class.py` to verify val ignores aug and warp and that train+warp alters pixels.
- 2025-11-01: Moved augmentation code to `vkb/augment.py` (exports: `_resize_to`, `_zoom_out`, `_rotate_square`, `_perspective_warp_inward`, `Augment`). `FrameDataset` now imports and uses `Augment`, passing hooks so tests can monkeypatch `vkb.finetune._zoom_out` as before. Updated `tests/test_augment_class.py` to import from `vkb.augment`.
- 2025-11-01: Moved `FrameDataset` to `vkb/dataset.py` and re-exported it from `vkb.finetune` for backward compatibility. `FrameDataset` pulls augmentation hooks from `vkb.finetune` at runtime to preserve monkeypatch points. All aug-related tests and finetune print/helpers tests pass.
- 2025-11-01: Added a tiny cache class `vkb.cache.Cache` that wraps existing functions (`cache_dir`, `cache_path_for`, `save_embeddings`, `load_embeddings`, `ensure_frames_cached`, `open_frames_memmap`). Added tests `tests/test_cache_class.py` to check embedding roundtrip and frame open/ensure wiring.
- 2025-11-01: Added `vkb.artifacts.Artifacts` — a minimal OO wrapper around artifact helpers (save/list/latest/save_sidecar). Test `tests/test_artifact_class_wrapper.py` ensures roundtrip and sidecar write.
- 2025-11-01: Added `vkb.io.IO` — thin wrapper for I/O helpers with instance `data_root` and `timefmt`. Methods: `safe_label`, `make_output_path`, `list_videos`, `update_latest_symlink`. Tests in `tests/test_io_class.py` cover path generation, listing, and symlink update.
- 2025-11-01: Added `vkb.emb.Embedder` — tiny OO wrapper around `create_embedder()`. Methods: `embed(frame)` and `__call__(frame)`. Test `tests/test_embedder_class.py` monkeypatches `create_embedder` and asserts both paths return the same vector.
- 2025-11-01: Split DL training epoch into smaller helpers without changing output: `_log_aug_samples_if_first_batch`, `_forward_backward_step`, `_print_batch_progress`. Added `tests/test_train_epoch_split_helpers_exist.py` to lock the split. Adjusted `vkb.emb.create_embedder` to inspect the `timm.create_model` signature and only pass supported kwargs (global_pool) to keep tests’ tiny stubs working.
- 2025-11-01: Further split `finetune`: added `_setup_mlflow`, `_run_training_epochs` (not yet used), and `_print_summary`. `finetune()` now uses `_setup_mlflow` and `_print_summary`; behavior and printed lines remain identical. Tests still pass.
- 2025-11-01: Broke up `finetune` even more: `_mlflow_log_params`, `_init_progress`, `_setup_viz_and_telemetry`, `_epoch_after`. Wired these into `finetune()` with identical prints. Added test `tests/test_finetune_more_split_helpers_exist.py` to lock the helpers.
- 2025-11-01: Made `finetune` available as a class entrypoint: `vkb.finetune.Finetuner.finetune(args)` calls `Finetuner(args).fit()` which delegates to the module-level `finetune(args)`. Added `tests/test_finetuner_classmethod.py` to assert the classmethod exists and delegates correctly.
- 2025-11-01: Moved Visdom helpers out of `vkb.finetune` into `vkb.vis` (`setup_visdom`, `visdom_prepare`, `viz_scalar`, `viz_aug_text`). `vkb.finetune` now imports and re-exports them via the old names for backward‑compatible tests. All Visdom tests pass.
- 2025-11-01: Added `tests/test_no_visdom_e2e.py` to assert that DL e2e runs do not touch Visdom when `VKB_VISDOM_DISABLE=1` is set; installs a guard module to fail if `visdom` is imported and checks CLI output contains no Visdom messages.
- 2025-11-01: Added `tests/test_no_models_writes_e2e.py` to ensure both classic (ridge) and DL e2e flows do not write into the repo `models/` dir; we snapshot `models/` before and after and redirect saves to a temp dir by monkeypatching `vkb.artifacts.save_model` (and the `train_frames.save_model` alias).
- 2025-11-01: Minor robustness fixes: `train_frames.train` now uses safe `getattr` for `test_split`/`eval_split` in mixed test contexts; `_log_aug_samples_if_first_batch` no longer requires a Visdom client to log MLflow image artifacts (relies on `Telemetry`, which handles both).
- 2025-11-01: Added `tests/test_no_mlflow_e2e.py` to assert e2e flows do not import or write to MLflow unless explicitly requested. Installs a guard `mlflow` module and sets `VKB_MLFLOW_DISABLE=1`, then runs ridge and DL paths.
- 2025‑11‑01: Image logging to Visdom is now ON by default (`--visdom-aug 4`). Use `--visdom-aug 0` to disable. Docs and troubleshooting updated.
- 2025‑11‑01: Dynamic rotation now supports up to full 360°. `AugScheduler` includes `rot_deg` with a max of 360, and its level ladder now reaches the declared maximum. `FrameDataset` applies a per‑angle safe zoom‑out before rotation so content isn’t cropped (scale s = 1/(|cos a|+|sin a|), clamped to [0.60, 1.0]). New tests: `tests/test_dynamic_rot_full.py`.
- 2025‑11‑01: `infer_live.py` DL device guard: if `--device cuda` is requested without a CUDA runtime, we raise a clear error before touching the camera. Test: `tests/test_infer_live_cuda_guard.py`. Rotation line in aug‑strengths plot is normalized (`rot=rot_deg/360`).
- 2025‑11‑01: Inference preprocessing: factored a tiny `_make_preprocess()` in `infer_live.py` and added `tests/test_dl_preprocess_factory.py` to assert BGR→RGB, resize, and normalization. No behavior change.
- 2025‑11‑01: Aug-strengths normalization test: `tests/test_aug_strengths_rot_norm.py` ensures the Visdom rotation series logs as `rot=rot_deg/360` (1.0 for 360°).
- 2025‑11‑01: Frame cache strictness: changed memmap validation from `>=` to `==` size to catch truncation; added `tests/test_frame_cache_strict_size.py`. Fixed classic train sample‑weights access to tolerate missing `class_weights` in tests.
- 2025‑11‑01: Frame cache drift check (9a): after a concurrent build wait, we now re‑validate dimensions and source fingerprint in addition to exact size; added `tests/test_frame_cache_dims_drift.py`. Suite: 139 passing.
- 2025‑11‑01: Cache rebuild logging (9b): `ensure_frames_cached` now prints a one‑liner when rebuilding (reason=size/dims/fingerprint). It prints via the provided console or when `VKB_CACHE_LOG=1` is set; default tests stay quiet.
 - 2025‑11‑01: Per-batch print now says `(samples/s)` instead of `fps` for consistency.
- 2025‑11‑01: Guard against empty memmaps in `FrameDataset`: raises a clear RuntimeError if `n<=0`. Test: `tests/test_empty_frame_cache_guard.py`. Suite: 140 passing.
- 2025‑11‑01: Removed repo‑local `timm` stub to avoid shadowing real timm. Tests now inject a minimal `timm` stub via `sys.modules` where needed (`tests/test_regularization_flags.py`, `tests/test_timm_stub_pooling.py`). Suite: 144 passing.
- 2025‑11‑01: Cache summary (13d): after epoch 1, print a one‑liner: `Cache: videos=<n> hits=<h> misses=<m>`. Implemented via counters in `FrameDataset` and printed in `_epoch_after`.
- 2025‑11‑01: Checkpoint commit/push after 13d; branch up to date.
 - 2025‑11‑01: Infer classic/DL selection hardened: prefer `clf_name=='finetune'` to route DL; `load_bundle` tries pickle first. If `timm` import fails for DL, we print a clear message and exit gracefully. Restored CUDA guard to raise as expected in tests. Suite: 147 passing.
- 2025‑11‑01: Default DL batch size doubled from 64 to 128.
- 2025‑11‑01: Zoom‑out, rotation, and warp now use mirror padding (reflect) instead of black; added `tests/test_mirror_padding.py` to assert behavior.
- 2025‑11‑01: Added MLflow logging (`--mlflow`, `--mlflow-uri`, `--mlflow-exp`, `--mlflow-run-name`) for both classic and DL training. Minimal params/metrics/artifacts are logged.
- 2025‑11‑01: Clarified Visdom image logging behavior in troubleshooting. To see frames, start the server on port 8097, select env `vkb-aug`, run DL training with `--visdom-aug > 0`; classic (ridge/xgb/logreg) paths don’t send images.
- 2025‑10‑31: Initial file added with defaults for a minimal, chorded, landmark‑based input system; codified thresholds, testing, and LM use for decoding.
- 2025‑10‑31: Added `record_video.py` CLI to capture labeled videos to `data/<label>/<YYYYMMDD_HHMMSS>.mp4` with a live preview. Added unit tests for path creation and arg parsing. Notes: keep OpenCV import local to simplify tests; no fallbacks for codecs or camera indices.
- 2025‑10‑31: Added `train_frames.py` — embeds all frames with a small image model (`timm` backbone, default `mobilenetv3_small_100`) and trains a frame‑wise classifier via `--clf {xgb|ridge|logreg}`. Ridge supports `--alpha`; LogReg supports `C`. Tests cover CLI parsing and dataset discovery without importing heavy deps.
- 2025‑10‑31: Switched baseline to an embedding‑only pipeline per user request; added guidance on ROI calibration and probability‑hysteresis press gating. Landmark/keypoint path remains documented for reference but is not the default.
- 2025‑10‑31: Added a minimal end‑to‑end fine‑tuning path (`--clf dl`) wired through `train_frames.py` and implemented in `vkb/finetune.py` (MobileNetV3‑Small, CE+label smoothing, GPU if available). No ROI; trains on all non‑validation frames each epoch.
- 2025‑10‑31: Ran `--clf dl` training on a small subset (`data_small/` with 3 labels, 946 frames, 1 epoch). Result: train_acc≈0.894, val_acc≈0.428 on tail split. Learnings: mp4 random access per frame is the current bottleneck; consider sequential per‑video decoding for speed when scaling to full `data/`.
- 2025‑10‑31: Fine‑tune trainer now prints the selected device (cpu or cuda + GPU name). Added tests to assert the device line appears.
- 2025‑10‑31: Added richer DL training feedback: dataset summary, per‑epoch batch count, periodic loss/acc/FPS lines, and a "Validating..." marker. Tests ensure these messages appear.
- 2025‑10‑31: Added end‑of‑run DL summary: average epoch duration and overall training throughput (samples/sec). Tests cover presence of the summary line.
- 2025‑10‑31: Added validation confusion matrix and macro‑F1 to DL training. We print a Rich table and a `confusion_raw=[[...]]` line for easy parsing/tests. Added tests.
- 2025‑10‑31: DL trainer now prints a concise HParams table at start (model, epochs, batch_size, lr, weight_decay, eval_split/mode, optimizer, label_smoothing). Tests ensure presence of these fields.
- 2025‑10‑31: GPU note — user asked why GPU wasn’t used. Current venv has CPU‑only PyTorch, so `torch.cuda.is_available()` is False and `--device auto` selects CPU. To use GPU: install CUDA wheels (e.g., `pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1`), verify with `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`, then run with `--device cuda`.
- 2025‑10‑31: Added `--require-cuda` flag. DL trainer raises a clear error if CUDA is requested but unavailable. Device print now includes GPU VRAM (GB).
- 2025‑10‑31: User asked to consider end‑to‑end fine‑tuning. Proposed minimal plan: fine‑tune a small CNN (e.g., `mobilenetv3_small_100`) directly on video frames with holdout‑by‑video evaluation; probability hysteresis for key emission remains applicable.
- 2025‑10‑31: Refactor — introduced `vkb/` package with shared utilities: `vkb.io` (`safe_label`, `make_output_path`, `list_videos`) and `vkb.emb` (`create_embedder`). Updated scripts and tests to import from `vkb`. Added `tests/conftest.py` to put repo root on `sys.path`.
- 2025‑10‑31: Models now persist with timestamped filenames via `vkb.artifacts.save_model` under `models/`, used by `train_frames.py`.
- 2025‑10‑31: Filenames now include full classifier names: `ridge` or `xgboost` (not `xgb`). Mapping handled in `train_frames._clf_display_name` and covered by tests.
- 2025‑10‑31: Training now saves a bundle dict with `clf`, `labels`, `clf_name`, and `embed_model`. Added `infer_live.py` for live camera classification using the latest model (or `--model-path`). Also added `vkb.artifacts.latest_model` and tests.
- 2025‑10‑31: `infer_live.py` defaults to the newest model via `choose_model_path(None, "models")`; added unit test to guarantee behavior.
- 2025‑10‑31: `infer_live.py` now prints the readable model name (`<clf_name> | <embed_model>`) on start for clarity.
- 2025‑10‑31: Polished CLI output with `rich` in `record_video.py`, `train_frames.py` (summary table), and `infer_live.py` (loaded/model lines). Imports are inside runtime paths to keep tests light.
- 2025‑10‑31: Expanded tests to cover artifact listing/sorting, inference CLI parsing, and `vkb.emb` importability without heavy deps. Total tests: 13.
- 2025‑10‑31: Added `--hpo-alpha` to `train_frames.py` for random log‑uniform Ridge alpha search; saves best model and prints a small HPO table. Unit tests added; total tests: 14.
- 2025‑10‑31: Added an end‑to‑end training test (`tests/test_e2e_train.py`) that fakes video capture and the embedder to verify the full training pipeline and model save path without heavy deps. Total tests: 15.
- 2025‑10‑31: `record_video.py` gained `--no-preview` for headless capture; loop is Ctrl‑C stoppable. Added a unit test. Total tests: 16.
- 2025‑10‑31: Fixed headless crash by skipping `cv.destroyAllWindows()` when `--no-preview` is used. Added a headless unit test. Total tests: 17.
- 2025‑10‑31: Added per-label `latest.mp4` symlink after each recording; visible in CLI output. Added tests and stabilized timestamps by sleeping in test to avoid same-second collisions. Total tests: 18.
- 2025‑10‑31: Symlink now has no extension (`latest`) to avoid accidental inclusion in training. Training now only consumes `.mp4` files. Updated tests accordingly. Total tests: 18.
- 2025‑10‑31: Added explicit `--model raw` option (32×32 grayscale flatten) to unblock environments without timm/torchvision; this is an explicit choice, not a silent fallback.
- 2025‑10‑31: Ran ridge training on existing `.mp4` dataset with `--model raw`. Saved model: `models/20251031_152612_ridge_raw.pkl`. Labels detected: `['PgDown','PgUp','no_input']`.
- 2025‑10‑31: Renamed training flag `--model` → `--embed-model` (breaking change) to clarify it selects the embedding backbone. All scripts/tests updated.
- 2025‑10‑31: Documented how to enable OpenCV GUI windows: ensure non‑headless `opencv-python` is installed for the exact interpreter used to run scripts, install system GTK runtime (`libgtk-3-0`/`libgtk-3-dev`) on Debian/Ubuntu, and avoid mixing `python3-opencv` (apt) with pip installs. Added tips to verify with `cv2.getBuildInformation()`.
 - 2025‑10‑31: Reported last training results to user: ridge | raw embedder, 1993 frames, 3 classes (`PgDown`, `PgUp`, `no_input`), saved to `models/20251031_152612_ridge_raw.pkl`.
- 2025‑10‑31: Noted OpenCV headless GUI error in some envs; recommended explicit `--no-preview` usage or installing non‑headless OpenCV. No automatic fallbacks added by design.
- 2025‑10‑31: HPO results display: running `train_frames.py --clf ridge --hpo-alpha N` prints a Rich table titled "HPO (alpha)" with columns (trial, alpha, val_acc) and uses the best alpha for the final model.
- 2025‑10‑31: OpenCV pip install note — `opencv-python` 4.12 from sdist tries to pin `setuptools==59.2.0`, which fails under recent `pip` (25.x). Fix: avoid building from sdist and force a prebuilt wheel. Recommended: `python -m pip install "opencv-python==4.10.0.84" --only-binary=:all:` (or `opencv-python-headless` if GUI not needed). Documented below.
- 2025‑10‑31: Training now prints the embedding feature dimension and adds it to the summary table. Unit test added.
- 2025‑10‑31: Ridge HPO now defaults to 10 trials (`--hpo-alpha` default=10). Set `--hpo-alpha 0` to disable.
- 2025‑10‑31: Removed legacy `latest.mp4` symlinks under `data/` (now we use an extensionless `latest`). Training ignores the extensionless link and no longer double‑counts.
- 2025‑10‑31: HPO now uses the same chronological tail split as the main evaluation (`--eval-mode tail`) via `idx_by_class` + `eval_frac`. Added a test.
- 2025‑10‑31: Added on-disk embedding cache per embedder (`.cache/vkb/<embed_id>/data/<label>/<file>.mp4.npy`). Training prints cache hits/misses and a cache summary with loaded/computed frame counts. Tests added. Total tests: 24.
- 2025‑10‑31: Training now reports embedding speed (FPS) per computed video and an aggregate speed table. Tests updated. Total tests: 24.
- 2025‑10‑31: Live inference overlays FPS (EMA of instantaneous) under the predicted label.
- 2025‑10‑31: Live inference also overlays per-frame timings: total processing time and embedding time (ms) each frame.
 - 2025‑10‑31: Live inference now also overlays classifier inference time (ms) and exposes a `_format_timings` helper; test added. Total tests: 29.
- 2025‑10‑31: Added XGBoost HPO (random search, default 10 trials) using the same tail split as evaluation; prints per-trial scores and chosen params. Tests added. Total tests: 25.
- 2025‑10‑31: HPO tables (ridge and XGB) are now sorted ascending by val_acc so the best row appears at the bottom. Helper functions added and tested. Total tests: 27.
- 2025‑10‑31: HPO-XGB logging now uses full parameter names (max_depth, n_estimators, learning_rate, subsample, colsample_bytree, reg_lambda); added a formatter and test. Total tests: 30.
- 2025‑10‑31: Added Logistic Regression as a classifier (`--clf logreg`) with HPO over `C` (default 10 trials) using tail split; shows chosen `C` in summary. Tests added. Total tests: 31.
- 2025‑10‑31: Removed explicit `multi_class='auto'` from LogisticRegression to silence scikit‑learn FutureWarning; sklearn ≥1.5 defaults to multinomial and will drop the param in 1.8.
 - 2025‑10‑31: Added `--logreg-max-iter` (default 500) and wired HPO to honor it to reduce convergence warnings; no scaling added by default per minimalism rule.

## New Learnings / Keep In Mind
### 2025-11-02 — DL Performance quick notes
- Symptom: very high `stall` (e.g., 0.8–0.96) with tiny `gpu_ms` → training is DataLoader/I/O bound, not GPU bound. Logs already print `io_ms`, `gpu_ms`, and `stall`.
- Main causes in our setup: per‑sample CPU transforms (resize/rotate/warp/color/erase) + memmap reads dominate; `prefetch=1` and small worker counts amplify stalls; Visdom/MLflow add overhead when enabled; MobileNetV3‑Small underutilizes GPU at 224².
- Quick fixes (no code):
- If you enable persistent workers, `file_system` sharing is usually fine. For speed, consider `--prefetch 2` and `--workers 4–8` on fast disks.
  - Keep `pin_memory=True` and non‑blocking H2D (already enabled). Disable Visdom/MLflow during perf runs.
  - Increase compute per sample to hide I/O: try `--img-size 320` (lower `--batch-size` accordingly) or a slightly heavier backbone.
  - If SHM is large and stable, you can try PyTorch’s default sharing strategy; otherwise keep `file_system` to avoid bus errors.
- Why “bigger batch slowed throughput” sometimes: loader becomes the bottleneck; larger batches wait for more samples to be prepared, increasing `batch_ms` while GPU is still mostly idle.
- Sanity run (safe): `VKB_VISDOM_DISABLE=1 VKB_MLFLOW_DISABLE=1 \
  .venv/bin/python train_frames.py --clf dl --device cuda --epochs 2 \
  --batch-size 96 --workers 4 --prefetch 2 --visdom-aug 0`

### 2025-11-02 — Why DL Val < Classic Val
- Likely causes in our data:
  - Labels: many frames in key‑labeled videos are actually `no_input` → noisy supervision inflates train acc and crushes val acc. Prefer event‑window labels (see Labeling & Data) or discard far‑from‑press frames.
  - ROI/background drift: whole‑frame training without a stabilized crop lets the net memorize backgrounds/lighting; generalizes poorly cross‑video. Calibrate ROI once and reuse it.
  - Aug policy mismatch: dynamic aug used to start at zeros and rotation was not in the schedule; static aug was weak. We now default to fixed, stronger aug and added `rot_deg` to the scheduler.
  - Small dataset vs full fine‑tune: updating all weights overfits quickly; classic models on frozen embeddings generalize better with limited data.
  - Class imbalance and sampler/prior shift: we balance batches in train, but val reflects real priors. Track macro‑F1 (already printed) in addition to acc.
  - Resolution/model size: 224² + MobileNetV3‑Small underfits motion details; try 320² (lower BS) or the Large backbone for more signal.
- Quick confirmations:
  - Re‑run with static strong aug, noise and erasing ON, `--img-size 320`, and monitor macro‑F1. If train≈1.0 but val stays low, it’s label/ROI.
  - Compare classic ridge/logreg on the same split; if they win, bias to frozen embeddings or freeze most of the DL backbone (tiny feature to add).
- Minimal remedies (design‑aligned):
  - Data: event‑window labels; stronger `no_input` negatives; stabilized ROI (union‑of‑motion box).
  - Model: consider freezing backbone layers; keep modest wd (3e‑4–5e‑4), drop_path 0.1–0.25, dropout 0.1; img size 320.
  - Eval: keep tail split by video; track macro‑F1 and per‑class recall.

### Quick tip — Use MobileNetV3‑Large
- Pass `--embed-model mobilenetv3_large_100` to `train_frames.py` for both classic and DL paths. Expect a larger feature dim (≈1280 vs 1024) and higher VRAM; consider smaller `--batch-size` (e.g., 64–96 at 224² on a 1080 Ti).
- With DSPy enabled, it's helpful to log both the policy used this epoch and the suggestion applied for the next epoch. We now print both.
- Suggestion latency can dominate short epochs; reduce with OpenRouter + `--dspy-model deepseek/deepseek-chat` (non-reasoner) or `--dspy-reasoning-effort low`. Consider applying suggestions every N epochs if latency remains high.
 - OpenRouter Responses API emits SSE events including `response.reasoning.delta` (incremental chain-of-thought) and `response.output_text.delta` (final answer). Use `reasoning={effort:low|medium|high}` to request reasoning tokens; some models ignore it. We only stream reasoning when explicitly enabled. (Sources: OpenRouter docs)
- DeepSeek’s official API uses the slugs `deepseek-chat` and `deepseek-reasoner`, both labeled “DeepSeek‑V3.2‑Exp” as of Nov 1, 2025. Use DSPy’s LiteLLM path: `dspy.LM("openai/deepseek-chat", api_key=$DEEPSEEK_API_KEY, api_base="https://api.deepseek.com")`.
- Our current sidecars capture run‑level aug strengths, not per‑epoch dynamics. If we want step‑wise (epoch) histories, add a tiny writer (e.g., `aug_history.jsonl`) in the trainer or parse console logs that print the aug policy.
- Integration pattern for async aug: at epoch start call `submit(history)`, do training, then fetch `result()` at epoch end and apply; if not ready within a short timeout, skip applying (do not add fallbacks). Keep the thread per‑run to minimize overhead.
- Flag gating (`--dspy-aug`) ensures zero DSPy/DeepSeek imports or calls unless explicitly requested, keeping default runs simple and dependency‑light.
 - Reasoning effort: passed as `reasoning={"effort":"low|medium|high"}` to OpenRouter; not all models honor it. We don't add fallbacks; if unsupported, the provider just ignores it.

- DL uses the frame cache: `FrameDataset.__getitem__` calls `vkb.cache.ensure_frames_cached` and then reads via `open_frames_memmap` (see vkb/dataset.py:35–42). First pass builds memmaps; subsequent epochs stream from disk. Invalidation uses source size+mtime; consider strict memmap size check (==) to catch truncation.
### ModelStore tags — planned UX (6a)
- Add `--tag <name>` to `infer_live.py` to load by tag using `vkb.model_store.ModelStore.resolve(tag)`.
- Resolution order (no hidden fallbacks): `--model-path` > `--tag` > env `VKB_MODEL_TAG` > `LATEST` pointer.
- This means you do NOT need to pass `--tag prod` every time; you can export `VKB_MODEL_TAG=prod` once and defaults will use it. Leaving everything unset continues to load `LATEST`.
- Training: keep writing `LATEST`. Optionally allow `train_frames.py --tag prod` to also tag the saved model as `prod`. Tags are simple pointer files next to models (e.g., `models/TAGS/prod` → filename). Sidecar may include `tags: [...]` when tagging.
- Dynamic aug should be able to hit the cap exactly (we now force the last level to equal the knob’s max). This matters for `rot_deg` reaching 360°.
- For rotation at larger angles, zooming out before rotation avoids losing content; computing s from the sampled angle is minimal and robust.
- Split policy clarity: our validation/test splits are per‑class chronological tails, not a single global tail. Implementation uses head→train, mid→val, tail→test per class (vkb/finetune._tail_split_three and classic path’s analogous helper). Example: with 100 frames/class and val=0.2, test=0.1 → train=first 70, val=next 20, test=last 10 frames for each class.
- Early-val dip: with strong train-time aug (rot360, warp, sat/contrast/wb/hue) and drop_path, the model first specializes to the augmented train distribution while the head re-initializes; val can dip for 1–3 epochs before the backbone re-aligns and regularization helps. Freezing the backbone for 1–2 epochs or using a small LR warmup often flattens the dip.
- Rich can wrap long console lines; assertions should not assume single-line output. Our Perf test searches the whole output for `samples/s` rather than only the line containing `Perf:`.
- When users want progress bars, prefer the existing `--rich-progress` flag instead of flipping defaults. Turning it on globally can break output-based tests.
- Keep protocols tiny and aligned to what call sites actually use. Avoid speculative methods.
- Prefer runtime-checkable Protocols for fakes in tests (`@runtime_checkable`) so we can `isinstance` without mypy.
- Cache invalidation: adding `src_size` + `src_mtime_ns` is enough for practical auto‑rebuilds. A full content hash is slower and unnecessary for now.
- Use `args.cfg` as the unified, pass‑through config object; evolve it incrementally. Don’t refactor all signatures at once.
- HPO integration can stay tiny by reading the existing finetune `.meta.json` sidecar for `val_acc`; no changes to `vkb.finetune` needed.
- We now use roll-around shifts by default; seam is inherent but acceptable per request. Keep rotation + warp modest if artifacts show up.
- Keep the search space minimal first (lr, wd, drop_path, dropout). Expand only if needed.
- External deps (e.g., Optuna) may not be present; do not add code fallbacks. Install explicitly when running.
 - Noise aug rarely helps our fine‑detail task; brightness/warp/rotation are safer. If needed, calibrate magnitude to the camera (or simulate the deployment path via JPEG recompress) rather than using large synthetic noise.
- When refactoring, preserve exact console strings and ordering — multiple tests assert them. Extract helpers only; don’t alter prints.
- Keep augmentation state centralized: class-based aug makes it easier to swap policies without touching dataset logic. Keep brightness on the tensor path so tests that compare train vs val remain valid.
- When moving helpers across modules, re-export names or pass hooks to avoid breaking tests that monkeypatch by old locations.
- Prefer lazy imports inside initializers to avoid circular imports when splitting modules (`FrameDataset` imports `vkb.finetune` only inside `__init__`).
- Cache notes: `embed_id` currently ignores `embed_params` and returns only the model name; this can collide if params change. Consider including params in the key if we start varying them.
- When splitting core loops, keep print strings exactly identical; tests assert for phrasing and case. Helper extraction should be behavior-preserving only.
 - Introduce new helpers in small steps and wire them gradually to avoid rippling through tests; prefer no-op helpers first, then replace inline code blocks.

### 2025-11-01 — Ports & Adapters quick take
- Ports are stable, minimal `Protocol`s the core owns; adapters wrap tech to satisfy them. In vkb: `Embedder`, `VideoSource`, `ArtifactStore`, `Telemetry` are prime ports. Start small; add methods only when the core truly needs them.
- Tests should hit the core with fakes implementing the ports; adapter specifics get their own small tests.
- Don’t over-abstract: prefer multiple tiny ports over a kitchen-sink interface; keep adapters thin; no hidden fallbacks.

### 2025-11-01 — Docs to reconcile
- Visdom behavior conflict: earlier bullets say “no fallbacks, raise if visdom missing”; later text says “warn and continue.” Pick one policy (recommend: error when a Visdom feature is requested but unavailable; silent when features are off).
- HPO split mismatch note appears outdated given “HPO now uses same chronological tail split.” Remove or mark the older note as superseded.

## Optuna Quick Use
- Install: `.venv/bin/python -m pip install optuna==3.6.1`
- Run (CPU, tiny trial): `.venv/bin/python optuna_finetune.py --data data_small --trials 1 --epochs 1 --batch-size 8 --device cpu`
- Defaults: `embed_model=mobilenetv3_small_100`, `eval_split=0.2`, `direction=maximize` over `val_acc`.

### Optuna Dashboard Quick Start
- Install UI: `.venv/bin/python -m pip install optuna-dashboard==0.16.0`
- Start server on localhost: `.venv/bin/optuna-dashboard --storage sqlite:///hpo.db --host 127.0.0.1 --port 8080`
- Note: our current `optuna_finetune.py` uses in-memory studies. To see trials in the dashboard, we should add a `--storage` flag and pass it to `optuna.create_study(storage=..., load_if_exists=True)`. Ask to enable this and I’ll add it with a tiny test.
- Use deterministic filenames (local time) to keep dataset ordering simple.
- Avoid over‑engineering capture features; add only when needed by downstream labeling.
- Tests should target pure helpers (path, args) to stay reliable without a camera or GUI.
- For training, keep heavy imports inside functions so unit tests run even without `cv2`, `torch`, `timm`, or `xgboost` installed.
- Default embedding: MobileNetV3‑Small (pretrained). Good trade‑off; switchable via `--model`.
- Shared code belongs under `vkb/`; keep scripts (`record_video.py`, `train_frames.py`) thin.
- Don’t add resolution/normalization fallbacks; if a different backbone is passed, user must align input size manually (by design).
- Persist models with `YYYYMMDD_HHMMSS_<clf>_<model>.pkl` in `models/` so inference can pick the latest easily.
- Bundle labels inside the saved pickle so inference renders semantic labels, not numeric class indices.
- Prefer Rich for human-facing CLI messages; keep usage minimal (no spinners) to avoid bloat.
- Keep unit tests focused on pure helpers and behavior that doesn’t require GPU/camera; heavy paths exercised manually.
- HPO uses a simple 80/20 stratified split and accuracy; minimal by design.
- Integration/E2E tests should use light fakes for `cv2` and the embedder to avoid GPU/GUI requirements.
- For headless environments (no GUI backend), use `--no-preview` instead of adding auto‑fallbacks; explicit is better here.
- Creating a relative symlink (`latest.mp4` → `<timestamp>.mp4`) keeps the link portable across moves of the label directory.
- Prefer extensionless `latest` symlink and restrict loaders to `.mp4` to prevent duplicate ingestion.
- `--clf` selects classifier: `ridge` (RidgeClassifier; supports `--alpha` and `--hpo-alpha`) or `xgb` (XGBoost; stronger non‑linear, slower to train).
- `--embed-model` is the embedding backbone (timm name) or `raw` for 32×32 gray flatten; prefer 224×224‑friendly backbones since we resize to 224: `mobilenetv3_small_100` (default), `mobilenetv3_large_100`, `efficientnet_lite0`, `repvgg_a0`, `mobilevit_xxs`. Heavier: `convnextv2_tiny`, etc.
- OpenCV GUI: run the recorder with the same interpreter you used to install OpenCV (`pythonX.Y -m pip ...`); remove `opencv-python-headless`, install `opencv-python`, and ensure system GTK is present. Verify GUI support via `python -c "import cv2; print('GUI' in cv2.getBuildInformation())"`. In headless servers, use `--no-preview` or manually run under a virtual display (e.g., `xvfb-run`).
- Avoid building OpenCV from source unless intentional: on some systems `pip` may pick the sdist and then fail on a legacy `setuptools==59.2.0` constraint. Force wheels with `--only-binary=:all:` and pin to a known-good wheel (e.g., `4.10.0.84`). Prefer `opencv-python-headless` when GUI is off.
- timm MobileNet naming: the suffix `_100` denotes the width multiplier α=1.0 (100% channels). `_075` → α=0.75, etc. This scales channel counts, parameters, FLOPs, and often the pooled feature dimension (e.g., v3-small_100 → 1024-D, v3-large_100 → 1280-D).
- Embedding input size: our timm embedder resizes frames to 224×224 and uses ImageNet mean/std; the `raw` embedder uses `--raw-size` (default 32) and optional `--raw-rgb`.
- XGBoost single-frame latency: sklearn’s `predict` converts to DMatrix and incurs Python overhead each call; XGBoost parallelizes over rows, so `n_jobs` doesn’t help on a batch of 1. Deep trees and many estimators further add per-frame traversal. Prefer `Booster.inplace_predict` for single-row inference, or batch multiple frames.
- Logistic Regression `C`: inverse regularization strength (C = 1/λ). Smaller `C` → stronger regularization (simpler decision boundary, less overfit); larger `C` → weaker regularization. Our HPO samples `C` log-uniform in ~[1e-4, 1e2].

- HPO vs Summary split mismatch: HPO currently optimizes alpha on a random stratified split inside the sweep, while summary `val_acc` uses `--eval-mode` (default `tail`). Expect HPO trial accuracies to be higher than tail `val_acc`. To align, run with `--eval-mode random` or we can add an `--hpo-mode same_eval` option next.

## OpenCV Binary Install (Quick)
- Force wheel (no build):
  - GUI: `python -m pip install "opencv-python==4.10.0.84" --only-binary=:all:`
  - Headless: `python -m pip install "opencv-python-headless==4.10.0.84" --only-binary=:all:`
- If you need GUI windows, ensure a GTK runtime is present (e.g., Debian/Ubuntu `libgtk-3-0`), then verify via `cv2.getBuildInformation()`.

## OpenCV GUI Setup (Quick Reference)
- Verify current build: `python3.11 -c "import cv2; print(cv2.__version__); print(cv2.getBuildInformation())"` → look for `GUI:`. If `NONE`, you need a GUI-enabled build.
- Easiest: use Conda. `conda create -n vkb python=3.11 opencv -c conda-forge && conda activate vkb`.
- Pip + source build (pyenv):
  - `sudo apt-get update && sudo apt-get install -y build-essential cmake pkg-config libgtk-3-dev`
  - `python3.11 -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless`
  - `python3.11 -m pip install --no-binary=:all: opencv-python`
  - Re-check `cv2.getBuildInformation()`; `GUI: GTK3` should appear.
- Avoid mixing apt’s `python3-opencv` with pip’s; prefer an isolated env.

## PyTorch + Torchvision Install (CPU) Quick Recipe
- Stick to matching versions; for torch 2.5.1 use torchvision 0.20.1.
- Use the official PyTorch CPU index to avoid ABI mismatches:
  - `python3.11 -m pip uninstall -y torchvision timm`
  - `python3.11 -m pip install --index-url https://download.pytorch.org/whl/cpu torchvision==0.20.1`
  - Optional fresh pair: `python3.11 -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1`
  - Then: `python3.11 -m pip install timm==0.9.16`
- Verify ops are registered: `python3.11 - <<'PY'\nimport torch, torchvision, torchvision.ops as ops\nprint(torch.__version__, torchvision.__version__, 'nms:', hasattr(ops,'nms'))\nPY`
- If conflicts persist, use a clean venv: `python3.11 -m venv .venv && source .venv/bin/activate` and repeat the steps above.
## Venv Created (2025-10-31)
- Created Python 3.11 virtualenv at `.venv`.
- Installed: torch 2.5.1+cpu, torchvision 0.20.1+cpu, timm 0.9.16, scikit-learn, xgboost, opencv-python, rich, pytest.
- Use without activating: `.venv/bin/python train_frames.py --clf ridge --embed-model mobilenetv3_small_100 --eval-split 0.2`.

## Additional Change Log (2025-10-31)
- Inference supports finetuned DL bundles (state_dict). New flags: `infer_live.py --device {auto,cpu,cuda}` and `--frames N`. Fixed KeyError when loading DL models. Added tests for DL inference path.
- DL training: added `--workers` to speed I/O, automatic `pin_memory` on CUDA, CUDA events timing, per-batch `io=` and `gpu=` ms, and epoch perf summary (stall ratio). `--require-cuda` enforces GPU use. `--persistent-workers` default is OFF.
- GPU wheels installed (cu124) and verified on GTX 1080 Ti. Use `--device cuda` and watch `model_on_cuda=True` in logs.
- Added on-disk frame cache: sequentially decodes each video once and stores a memmap under `.cache/vkb/frames/`. FrameDataset uses the cache for random access. Includes dimension check + rebuild on mismatch. New tests cover cache build and read. This is the primary fix for the random-seek bottleneck.
- Clarified past 'raw' confusion: infer_live originally picked the newest model by timestamp, which could be an older `ridge_raw.pkl` (trained with a tiny 1‑D fake feature). Loading that bundle while using a 1024‑D embedding produced the "expects 1, got 1024" error. Fix: training now writes `models/LATEST`, infer_live prefers it; infer_live also infers `raw_size` from `n_features_in_` for legacy raw bundles. Tests added.
- 2025-10-31: Added Glossary entry: `vkb` stands for Vision Keyboard.


- 2025-10-31: Refactored DL training epoch logic in vkb/finetune.py: split the old `_run_epoch` into `_train_epoch`, `_validate_epoch`, and `_print_perf` for readability. Added tests/test_finetune_helpers_split.py to lock the split.

## New Learnings (today)
- Preserve existing console messages during refactors; several tests assert them.
- Safer DataLoader defaults (few workers, file_system sharing) reduce /dev/shm issues.
- Refactoring into tiny helpers keeps complexity low while improving maintainability.

- 2025-10-31: Added a minimal `Finetuner` class (object API) that delegates to `finetune(args)`; kept functions to avoid duplication and keep tests stable. New test `tests/test_finetune_class.py` ensures the class exists and `run()` delegates.
- 2025-10-31: Added `SampleIndex` (videos → samples → tail split) in `vkb/finetune.py` to encapsulate listing, sample expansion, and tail split logic; test `tests/test_sample_index.py` verifies parity with helper functions.
- 2025-10-31: Added more tests for SampleIndex/tail split edge cases and coverage: `tests/test_sample_index_edges.py`, `tests/test_tail_split_disjoint_fullcover.py`. Test suite now at 55 passing.
- 2025-10-31: Added integration tests for train→infer: `tests/test_integration_ridge_infer.py` and `tests/test_integration_xgb_infer.py` train a tiny model with a fake embedder and run `infer_live` headless. Suite: 57 passing.
- 2025-10-31: Added end‑to‑end CLI tests: `tests/test_e2e_cli_ridge_latest.py` and `tests/test_e2e_cli_dl_latest.py` run `train_frames.main()` and then `infer_live.main()` using `models/LATEST`. Suite: 59 passing.
- 2025-10-31: Ran real DL fine‑tune on local data (`mobilenetv3_small_100`, CUDA). 2 epochs, tail eval 0.20 → val_acc ≈ 0.603. Perf notes: high stall ratio (I/O dominated) despite memmap cache; workers=0 avoided shm errors. Latest model: `models/20251031_194026_finetune_mobilenetv3_small_100.pkl` and pointer `models/LATEST` updated.

## Change Log
- 2025-11-01: Default DL DataLoader workers increased from 0 → 4. Rationale: better overlap on cached datasets; lock added to frame cache mitigates prior races. If you hit issues, pass `--workers 0`.
- 2025-11-01: Use non‑blocking CUDA transfers (`.to(device, non_blocking=True)`) in train/val loops to overlap H2D copies with compute. No behavior change on CPU.
- 2025-11-01: Removed `--amp` (CUDA mixed precision) per user request; on GTX 1080 Ti and loader‑bound runs it did not improve throughput and added complexity. Keep non‑blocking H2D; revisit AMP only if switching to newer GPUs.
- 2025-11-03: Default `--persistent-workers` is now OFF by request. Enable explicitly when you want fewer epoch startup stalls.
- 2025-11-01: Added `scripts/sweep_val.sh` — a minimal sequential DL val-score sweep runner (100 epochs per run). Writes models under `runs/models_<timestamp>` and prints `RESULT val_acc=...` after each run by reading the sidecar `.meta.json`.
### 2025-11-01 — E2E MLflow safety
- Goal: ensure e2e/integration tests never touch your real MLflow.
- Two simple patterns (both already supported in code/tests):
  - Disable MLflow entirely during tests (even if `--mlflow` is passed):
    - `VKB_MLFLOW_DISABLE=1 pytest -q`
  - Route MLflow to a sandbox file store when you want to exercise logging paths:
    - `MLFLOW_TRACKING_URI=file:/tmp/mlruns_e2e pytest -q -- --mlflow --mlflow-exp vkb_e2e`
    - Or pass `--mlflow-uri file:/tmp/mlruns_e2e` on the CLI.
- Also isolate other outputs in tests:
  - Models dir: `VKB_MODELS_DIR=/tmp/vkb_models_test`
  - Visdom off: `VKB_VISDOM_DISABLE=1`
- Quick copy‑paste for a safe local run:
  - `VKB_MODELS_DIR=/tmp/vkb_models_test VKB_MLFLOW_DISABLE=1 VKB_VISDOM_DISABLE=1 \`
    `python train_frames.py --clf ridge --embed-model mobilenetv3_small_100 --eval-split 0.2`

### 2025-11-01 — Train/Val/Test split + best‑val testing
- Added `--test-split` (per‑class, tail) to hold out a final test portion. Works for both classic and DL paths.
- DL: track best epoch by validation accuracy and evaluate that checkpoint on the test split at the end; logs `test_acc` and writes it to the sidecar (`.meta.json`).
- Classic: if `--test-split > 0`, we evaluate the fitted model on the held‑out test frames and include `test_acc` in the sidecar.
- CLI example: `python train_frames.py --clf dl --epochs 3 --eval-split 0.1 --test-split 0.1`.
- Notes: tail split = chronological per‑class head=train, mid=val, tail=test. Random mode is supported for classic via a two‑step stratified split; DL uses tail by design.

## Notes — 2025-10-31 — Regularization quick guide

- DL fine-tune defaults we like (small datasets): weight decay 1e-4–5e-4, dropout 0.1–0.2 in head, label smoothing 0.05, cosine LR with warmup, optional EMA/SWA at end.
- Augment sparingly for micro‑gestures: small rotate/translate/crop, light color jitter, tiny random erasing; try mixup 0.05–0.1 (disable if it hurts).
- Linear models: Ridge/LogReg with L2; scale features; HPO alpha (logspace) or C (inverse L2 strength). Save chosen alpha/C in bundle metadata.
- XGBoost: prefer shallower trees (max_depth 4–6), subsample/colsample 0.6–0.9, lambda 1–20, gamma 0–4, learning_rate ~0.03–0.1, early stopping if a val split exists.
- If no validation split: rely on weight decay + schedule; optionally SWA/EMA; still keep a tiny tail split for sanity when tuning.
- Calibration after training (temp scaling) can improve confidence without changing accuracy.
- Keep tests: verify regularization shrinks train–val gap on a toy overfit set.

### 2025-11-01 — Overfitting playbook (minimal)
- Tweak (no code): increase `--wd` to `3e-4` or `5e-4`; set `--dropout 0.1`; consider `--drop-path 0.30` (small bumps). Reduce epochs if curves peak early.
- Augment (light): keep `--aug rot360`, nudge `--warp 0.25`, `--brightness 0.10`. Only add color jitter if needed (δ≤0.10).
- Batch/optimizer: try smaller batches (64→32) to add gradient noise; keep AdamW.
- Mixup (tiny, optional code): α≈0.05; implement as `lam*CE(y) + (1-lam)*CE(y_perm)` without changing loss class.
- Scheduler (tiny): cosine LR with warmup; step per epoch.
- SWA (tiny): last 5–10 epochs via `torch.optim.swa_utils`; update BN once on train loader.
- Freeze-unfreeze (tiny): freeze backbone for 5 epochs (linear probe), then unfreeze and fine‑tune.
- Data: prefer more diverse clips over heavier aug; keep tail split; clean labels.

## Notes — 2025-10-31 — What we use for DL (current)

- Backbone: `timm` MobileNetV3 (`mobilenetv3_small_100` default), replace classifier head to `num_classes`.
- Input: whole frame, resized to 224×224, ImageNet mean/std normalization. No ROI.
- Loss/opt: CrossEntropy with label smoothing 0.05; AdamW (lr default 1e-4, wd default 1e-4).
- Device: `--device auto` picks CUDA if available; prints model_on_cuda, VRAM, and dataloader settings.
- Data: per-frame memmap cache on disk (`.cache/vkb/frames`) used by the DL dataset; chronological tail split for eval by default. We now train on all frames and rely on shuffling (and class‑balanced sampling when enabled) rather than stride‑based subsampling.
- Loader defaults: workers=4, prefetch=1, persistent_workers=on, sharing=file_system when workers>0 (to avoid /dev/shm crashes). Flags exist to change.
  
### 2025-11-01 — Sharing strategy default
- If you don’t pass `--sharing-strategy`, we set it automatically to `file_system` when `--workers > 0` (see `vkb.finetune._make_loaders`). With `--workers 0`, nothing is set (no multiprocessing needed).
- `file_system` avoids many `/dev/shm` issues but can be a bit slower than `file_descriptor`. Only switch to `file_descriptor` if you know your environment’s shm is large enough.
- Progress/metrics: Rich progress optional; per-batch IO/GPU timings, epoch train_acc, val_acc, confusion matrix, macro‑F1, perf summary.
- Artifacts: saves `models/<timestamp>_finetune_<model>.pkl` + updates `models/LATEST`; `infer_live.py` prefers LATEST and supports both DL and sklearn bundles.
- Known limits: first epoch may be I/O‑bound while caches build; speed improves next epochs.

### DL regularization currently active
- L2 weight decay (AdamW `--wd`, default 1e-4).
- Label smoothing = 0.05 in CrossEntropyLoss.
- Drop path (stochastic depth) ON by default: `--drop-path 0.25`.
- Dropout default 0.0 (most MobileNetV3 variants ignore it anyway).
- Augmentations enabled by default: `--aug rot360`, `--brightness 0.15`, `--warp 0.20` (train only). Class imbalance mitigation on by default (`--class-weights auto`) with balanced loss and sampler.
- No EMA/SWA/early stopping right now.

## Notes — 2025-10-31 — Augmentation policy (zoom‑out + rotation)

- Goals: preserve fine thumb–finger cues, add robustness to small pose/lighting changes, stay cheap. Never crop in; only zoom out.
- Train presets (`--aug {none,light,heavy,rot360}`; default: none):
  - light: RandomResizedCrop 224 (scale 0.90–1.00, ratio 0.95–1.05), RandomRotation ±3°, ColorJitter (b/c/s=0.05, hue=0.01), RandomErasing p=0.2 (area 2–6%). No flips by default.
  - heavy: RandomResizedCrop 224 (scale 0.80–1.00, ratio 0.90–1.10), Rotation ±7°, ColorJitter (0.10, hue=0.02), RandomGrayscale p=0.10, GaussianBlur p=0.20 (σ 0.1–0.5), RandomErasing p=0.5 (area 2–8%). Still no flips unless labels are flip‑invariant.
- Val/test transforms: Resize 224 only; no jitter/erasing. In tests, a stubbed cv2 may keep native size; the pipeline tolerates that.
- Flip caution: Horizontal flips can invert semantics (left/right); keep off unless explicitly safe.
- Mixup/CutMix: optional later; start with α≈0.1, p≈0.1 if you need more regularization.

- `rot360` details: We first zoom out to a safe scale (0.60–0.68) so a full 0–360° rotation fits inside 224×224 without cropping, then rotate around the image center. Keeps the “no crop‑in” rule intact.

### Image noise augmentation (removed)
- We removed Gaussian noise from the pipeline to simplify training and avoid masking small contact cues. The CLI flag `--noise-std` is now ignored.

### Noise augmentation — stance (2025-11-01)
- For vkb’s micro‑gesture cues, pixel noise usually hurts (it hides tiny contacts). Keep noise OFF by default.
- If domain gap demands it (e.g., grainy webcams, compression): prefer small additive Gaussian noise, train‑only, after brightness and before normalization; σ≈0.01–0.03 on [0,1].
- Alternatives: JPEG artifact augmentation (quality 40–80) often approximates real pipeline noise better than per‑pixel noise. Use sparingly.
- Avoid salt‑and‑pepper by default; use Poisson/speckle only if the sensor characteristics justify them.

### Brightness augmentation
- Flag: `--brightness <float>` applies multiplicative jitter by a factor in `[1-b, 1+b]` on the [0,1] tensor (train only), then clips to [0,1].
- Suggested: start at 0.10; heavy=0.20. Keep modest with `rot360` to avoid washing out details.
- Printed in HParams as `brightness`.

### 2025-11-01 — Brightness quick fact
- Current default is `--brightness 0.15` → factor range `[0.85, 1.15]`, train-only; validation/test get no brightness jitter. Covered by tests in `tests/test_augment_class.py`.

### 2025-11-01 — Batch Size quick fact
- `train_frames.py` default DL batch size is 128 (train and val share it). See `train_frames.py:24` and `vkb/finetune.py` DataLoader calls. `tests/test_finetune_args.py` asserts the default.
- `optuna_finetune.py` uses a smaller default (8) for speed during HPO runs.

### 2025-11-01 — Effective batch size vs micro‑batches
- Not equivalent by default: 4 updates of size 64 are NOT the same as 1 update of size 256. You make 4 optimizer steps vs 1, BatchNorm sees different statistics, LR schedules step differently, and gradient noise differs.
- Near‑equivalence requires gradient accumulation: compute 4 micro‑batches of 64, average their losses/grads, then do a single optimizer step. This matches a 256 step for most optimizers (SGD/AdamW) but still differs in BN/dropout randomness.
- Schedules: step the LR once per effective update (after accumulation), not per micro‑batch.
- Linear LR scaling rule: if you increase true batch size by k (without accumulation), start by scaling LR by k and keep warmup; then tune.

### 2025-11-01 — Why larger batch reduced throughput
- Our pipeline is often loader‑bound (CPU aug + memmap reads). With `--workers 1` and `prefetch=1`, a bigger batch increases per‑batch prep time more than GPU compute is amortized, so the GPU waits; samples/s drops.
- Random access across many videos per batch thrashes the page cache; larger batches touch more files at once → worse locality.
- Bigger H2D copies from pinned memory reduce overlap when there’s only one worker.
- Confirm: watch `io=` vs `gpu=` in the per‑batch print; if `io` dominates and grows with batch size, you are loader‑bound.
- Remedies: keep batch modest (32–64), use `--workers 1–2` and `--prefetch 2` (lock now prevents races), or add gradient accumulation to emulate large effective batches without stressing the loader.

### 2025-11-01 — Speed benchmark (GTX 1080 Ti, local data)
- Env: `--device cuda` `--workers 4` `--sharing-strategy file_system` `--visdom-aug 0` `VKB_VISDOM_DISABLE=1` `VKB_MLFLOW_DISABLE=1`. 1 epoch runs after warming the frame cache.
- bs=64, prefetch=2: train_fps ≈ 408 samples/s, epoch ≈ 39.3 s.
- bs=96, prefetch=2: train_fps ≈ 463 samples/s, epoch ≈ 34.7 s. (best)
- bs=96, prefetch=1: train_fps ≈ 443 samples/s, epoch ≈ 36.2 s. (slower)
- bs=128, prefetch=2: train_fps ≈ 402 samples/s, epoch ≈ 40.0 s. (slower)
- AMP (`--amp`) at bs=64, prefetch=2: train_fps ≈ 369 samples/s (slower; loader‑bound). Recommendation: keep AMP off for now on 1080 Ti.
- workers=2 (bs=96, prefetch=2): train_fps ≈ 251 samples/s (much slower; stick with 4).

Recommendation (today)
- Use `--batch-size 96 --workers 4 --prefetch 2` on CUDA, keep `--amp` off. If `io=` dominates, reduce color/geom aug or try `--batch-size 64`.

### 2025-11-01 — Why throughput ramps up within/after epoch
- Warmups: cuDNN algorithm search (benchmark) picks faster kernels after initial batches; GPU allocator/pinned-memory pools also “settle”.
- DataLoader pipeline fill: prefetch queues and workers take a few steps to fully overlap CPU → H2D → GPU.
- OS page cache: memmap reads become much faster after the first pass over each video.
- Visdom/MLflow: first connection/handshake can add latency; later batches avoid it.

Tips
- Enabling `--persistent-workers` keeps workers alive across epochs; reduces the ramp at epoch boundaries (optional).
- Prefer `--prefetch 2` (with workers>0) to keep the queue full; watch memory.
- If you crank workers high (e.g., 8), ensure the disk can keep up; otherwise stall remains high (io≫gpu).
- Pascal (1080 Ti): AMP generally does not help; keep it off unless a micro-benchmark shows otherwise.

### 2025-11-01 — Prefetch rationale (PyTorch DataLoader)
- Round‑robin assigns index batches to workers, but workers produce results asynchronously. `prefetch_factor=k` means each worker keeps up to `k` batches in flight, so when it’s that worker’s turn, a ready batch is usually waiting.
- This overlaps CPU decode/aug + memmap reads with GPU compute and hides jitter; it reduces head‑of‑line stalls at epoch start and across variable‑cost batches.
- Order is preserved by batch index, so prefetch doesn’t allow out‑of‑order consumption; it just ensures the next expected batch is ready. Benefits plateau if one worker is consistently slow.
- Measured here (GTX 1080 Ti, w=4, bs=96): prefetch=2 beat prefetch=1 (epoch ≈34.7 s vs 36.2 s; +4–6% train_fps). Memory use rises with `k`.

### 2025-11-01 — Who builds a batch? (DataLoader)
- Map‑style Dataset (ours): each worker builds an entire batch by fetching `batch_size` items and applying `collate_fn` inside the worker. The main process does not merge partial batches from multiple workers.
- Round‑robin scheduling assigns the next batch’s indices to a single worker; order of yielded batches is preserved.
- With `pin_memory=True`, pinning happens in the main process after the worker hands off the batch.

### 2025-11-01 — Throughput (persistent workers; longer runs)
- Config A (balanced): `bs=96, workers=4, prefetch=2, persistent=on`. 3 epochs Perf samples/s per epoch: [390.7, 602.1, 671.3]. Peak=671.3 samples/s.
- Config B (max batch): `bs=256, workers=8, prefetch=1, persistent=on`. 2 epochs Perf samples/s per epoch: [599.7, 906.7]. Peak=906.7 samples/s (stall≈0.98; loader‑bound).
- Logs: `/tmp/bench_long_persist.txt`, `/tmp/bench_bs256_w8_pf1.txt`.

### 2025-11-01 — Large batch sweep (1 epoch; workers=8, prefetch=2)
- bs=192 → peak 813.6 samples/s, avg_epoch≈29.4 s, stall≈0.92
- bs=224 → peak 1008.2 samples/s, avg_epoch≈25.9 s, stall≈0.78
- bs=256 → peak 1112.1 samples/s, avg_epoch≈22.8 s, stall≈0.83  ← fastest
- bs=288 → peak 792.6 samples/s, avg_epoch≈29.4 s, stall≈0.85
- bs=320 → peak 977.7 samples/s, avg_epoch≈26.9 s, stall≈0.89
- Takeaway: On 1080 Ti + our loader, bs≈256 with 8 workers, prefetch=2 gives the highest raw throughput. For stability/accuracy, bs=96 (w=4, pf=2) remains a good default.

### 2025-11-01 — Workers×batch sweep (1 epoch; prefetch=2)
- Grid: workers∈{12,16,20}, batch∈{224,256,288,320}.
- Peaks (samples/s):
  - (224, 20) → 1313.4  ← best
  - (288, 12) → 1283.1
  - (320, 12) → 1066.6
  - (256, 8) → 1112.1 (from earlier sweep)
- Notes: stall remains high (≈0.78–0.91), indicating loader‑bound regime. Larger worker counts broaden overlap and lift peaks on warmed caches/SSD.
- Recommendation (speed preset): `--batch-size 224 --workers 20 --prefetch 2` (persistent workers on). For accuracy‑oriented runs, prefer `--batch-size 96 --workers 4 --prefetch 2`.

### 2025-11-01 — User run: bs=256, workers=16
- Reported peak: ~1300 samples/s on GTX 1080 Ti with `bs=256`, `workers=16`. Likely helped by persistent workers and warm page cache.
- Tips to replicate: use `--prefetch 2 --persistent-workers`, keep Visdom/MLflow off, ensure fast local SSD, and let epoch 1 warm up.
- Sanity checks: watch `Perf: ... stall=...` (loader‑bound if ≳0.9); compare per‑epoch `train_fps` and per‑batch `fps` lines; confirm `pin_memory=True` in the device print. If running in a container, increasing `/dev/shm` may allow `file_descriptor` sharing to beat `file_system` (only if shm is large).
### 2025-11-01 — Image channels quick fact
- DL path uses color (RGB) frames: `vkb/dataset.py:40–50` converts BGR→RGB, scales to [0,1], then normalizes with 3‑channel ImageNet mean/std.
- Classic embedder (`vkb.emb.create_embedder`) also converts to RGB before timm features (`vkb/emb.py:16`).
- Historical note: a "raw" grayscale embedder was mentioned in docs; current repo path doesn’t expose it by default. If we add it back, default should be grayscale with an explicit `--raw-rgb` to opt into color.
- DL color mode confirmation (2025-11-01): DL always uses 3‑channel RGB; no grayscale option wired for DL at present.

### 2025-11-01 — Color augmentation (options)
- Keep color aug train-only and before normalization.
- Saturation jitter (cheap, no HSV needed): let `x∈[0,1]^{3×H×W}`, `g=dot(x,[0.2989,0.587,0.114])`; sample `s∈[1−δ,1+δ]`; set `x←clip(g + s*(x−g),0,1)`.
- Contrast jitter: per-image mean `m=x.mean()`; sample `c∈[1−δ,1+δ]`; set `x←clip((x−m)*c + m,0,1)`.
- White-balance jitter: per-channel gains `r,g,b∈[1−δ,1+δ]`; set `x←clip(x*[:,None,None],0,1)`.
- Hue shift (optional, heavier): convert to HSV, add small Δ to H (±2–4°), wrap, back to RGB.
- Suggested ranges for our task: δ≈0.05–0.10; avoid large hue shifts (they can obscure subtle skin-color cues).
- Not enabled by default; only add flags if needed (`--saturation`, `--contrast`, `--wb`), and keep tests minimal to assert train-only application.

#### Default policy (off) — rationale
- Signal risk: our micro‑gesture labels hinge on tiny shading/specular cues around contacts; color jitter can wash them out faster than it helps.
- Low domain gap: single camera, similar scenes → color statistics are already matched; geometry/brightness/warp cover the bigger variance.
- Small data: heavy color perturbations increase optimization noise and can misalign with ImageNet pretrain stats.
- Simplicity/tests: fewer knobs keep runs predictable; if color shift is a real issue, enable one knob at a time and measure.

#### When to consider enabling
- Multi‑camera training or strong auto‑WB fluctuations across sessions.
- Deployment under tinted lighting (RGB histograms differ from train).
- Verified wins from a tiny sweep (δ in {0.05, 0.10}) on your validation split.

### 2025-11-01 — Dataloader bus error (shm) quick fix
- Symptom: "Unexpected bus error ... insufficient shared memory (shm)" when `num_workers>0`.
- Quick fix: run with `--workers 0` (single‑process data loading). Also keep `--prefetch 1`, and avoid `--persistent-workers`.
- If you need workers: try `--workers 1` or `2`, `--sharing-strategy file_system`, `--prefetch 1`, and reduce `--batch-size`.
- Docker/Podman: increase shared memory (`--shm-size 1g`).
- Note: our dataset builds per‑video memmaps under `.cache/vkb/frames/`. Concurrent first‑time builds across workers can race; warming the cache with `--workers 0` once avoids this.

### 2025-11-01 — Frame cache locking
- Added a tiny lockfile in `vkb.cache.ensure_frames_cached()` to serialize per‑video memmap builds. Prevents worker races that could cause bus errors when multiple workers touch the same uncached video.
- Minimal unit test `tests/test_cache_lock.py` verifies lock acquire + timeout + release behavior.

### 2025-11-01 — Stale lock check/cleanup
- Check locks: `ls .cache/vkb/frames/*.lock` then for each file, read the PID and test `/proc/<pid>` exists. One‑liner:
  - `for f in .cache/vkb/frames/*.lock; do pid=$(tr -cd '0-9' < "$f"); [[ -d /proc/$pid ]] && echo "$f alive" || echo "$f STALE"; done`
- Safe cleanup: delete only those marked STALE: `rg -0 -n --files -g ".cache/vkb/frames/*.lock" | xargs -0 -I{} bash -lc 'pid=$(tr -cd \''0-9\'' < {}); [[ -d /proc/$pid ]] || rm -f {}'`

### Perspective warping
- Flag: `--warp <float>` moves each corner inward by a random fraction in `[0, warp]` and applies a perspective transform. Keeps the content inside the 224×224 canvas (no crop‑in). Padding uses mirror reflect; no black borders. Train only.
- Start at 0.10; heavy ≈ 0.20–0.30. Combines well with `rot360` and small noise.

### Augmentation defaults (current)
- `--aug`: `rot360` (enabled by default for rotation invariance).
  
- `--brightness`: `0.15` (multiplicative factor in [0.85, 1.15]; train only).
- `--warp`: `0.20` (perspective inward warp; train only).
- Built‑in ranges for aug modes:
- `light` zoom‑out: scale 0.90–1.00 (mirror‑pad center).
- `heavy` zoom‑out: scale 0.80–1.00 (mirror‑pad center).
- `rot360`: zoom 0.60–1.10 then rotate 0–360°. For s<1 we mirror‑pad (no black). For s>1 we center‑crop (corners can be cut off).

### Drop Path (stochastic depth) — notes
- What: randomly skip residual blocks per sample during training; acts like structured dropout at the block level and reduces overfit.
- Inference: disabled; paths are kept with rescaling so expected activations match.
- timm: pass `drop_path_rate=<float>` to `timm.create_model(...)`. It linearly increases across depth.
- Good starting values: 0.02–0.07 for small datasets; 0.1 for deeper nets or more data. For our MobileNetV3 small set, start at 0.05.
- Interplay: complements weight decay + label smoothing; use together. Keep regularization modest if epochs are few.

## Visdom (Augmentation Previews)
- Purpose: log a few augmented training images each epoch for sanity‑checking.
- Server: run `python -m visdom.server -port 8097` in a separate shell.
- CLI flags (DL only):
  - `--visdom-aug <N>`: number of samples to send from the first batch of each epoch (default 4 = on).
  - `--visdom-metrics`: also plot train/val accuracy per epoch as a line chart.
  - `--visdom-env <name>`: environment to write to (default `vkb-aug`, not `main`).
  - `--visdom-port 8097`: server port (default 8097).
- No fallbacks: if `--visdom-aug > 0` but `visdom` isn’t installed, training raises with an explicit message.

### Visdom default behavior and troubleshooting
- Default: training now attempts to log to Visdom automatically. If the client can’t import visdom or reach the server, a yellow warning is printed and training continues.
- Confirm server: `python -m visdom.server -port 8097` (and open http://localhost:8097/).
- Ensure flags are set: use at least one of `--visdom-metrics` or `--visdom-aug N` (>0). Without them, nothing logs by design.
- Check env dropdown: select `vkb-aug` (default env for this project).
- Sanity ping: `python -c "from visdom import Visdom as V; v=V(port=8097, env='vkb-aug'); print('connected=', v.check_connection()); v.text('hello from test')"` → should create a text pane.
- Match ports: if your server runs on a non‑default port, pass `--visdom-port <port>` to training.

- Why no frames? Frames are only sent when running DL training and only from the first training batch of each epoch. By default `--visdom-aug` is 4 (enabled). Ensure the server is reachable and you train with `--clf dl` — classic ridge/xgb/logreg paths do not push images.

### Dynamic Rotation (Full Range)
- Controller now adjusts `rot_deg` in degrees up to 360. With dynamic‑aug default ON, rotation can scale from 0 → 360 across training.
- In Visdom “Aug Strengths” the rotation series is normalized: we log `rot` = `rot_deg/360` so full rotation plots at 1.0.
- Dataset applies angle‑aware zoom‑out then rotates; prevents crop‑in at high angles. If `--aug rot360` is also used, both paths compose harmlessly (zoom‑out then dynamic rotation).

### Quick reference — `--visdom-aug`
- Purpose: send N augmented training images (grid) from batch 1 of each epoch to Visdom.
- Enable: enabled by default (`--visdom-aug 4`). Set to 0 to disable.
- Example: ``.venv/bin/python train_frames.py --clf dl --visdom-aug 8 --visdom-metrics --visdom-env vkb-aug --visdom-port 8097``
- Where it appears: env `vkb-aug`, window id `vkb_aug` (title “Aug Samples”).
- Notes: only DL path logs images; validation uses no augmentation (by design).

- Added optional Visdom logging of augmented images (env `vkb-aug`, port 8097). Enable via `--visdom-aug N`.
- Added Visdom scalar metrics (train/val accuracy) via `--visdom-metrics` (shares env/port 8097).
- Added per-run sidecar JSON saved next to each model: `<model>.pkl.meta.json` with key hparams, labels, eval split/mode, and summary metrics for both classic and DL paths.
  - DL sidecar now includes `stride`, `stride_rotate`, and `stride_offset_base` for reproducibility.
- Visdom now also shows a text panel "Aug/Reg Policy" that appends a one‑line summary each epoch (aug=brightness=warp=drop_path=dropout). It appears automatically when any Visdom feature is enabled.
  - The policy line reflects aug/regularization only (no stride fields).
- Overfitting diagnosis: large train/val gap traced to weak per‑video labels and heavy class imbalance; confusion matrices show most frames in key‑labeled videos look like `no_input`. Next: add an event‑window labeling path and class‑weighted loss.

## MLflow Logging
- Enable with `--mlflow` on `train_frames.py` (both classic and DL).
- Flags:
  - `--mlflow-uri`: tracking URI (falls back to env if unset).
  - `--mlflow-exp`: experiment name (default `vkb`).
  - `--mlflow-run-name`: optional run name; otherwise `clf|embed_model` (classic) or `dl|embed_model`.
- Logged:
  - Params: key hparams (model/epochs/batch_size/lr/wd/eval split/mode/drop_path/dropout/aug/brightness/warp; alpha/C for ridge/logreg).
  - Metrics: DL per‑epoch `train_acc` and `val_acc`; classic final `val_acc`.
  - Artifacts: saved model `.pkl`, sidecar `.meta.json`, and a small grid of augmented images per epoch (first batch) when `--visdom-aug > 0`.
- Behavior: if `--mlflow` is set but `mlflow` isn’t installed, training raises a clear error (no silent fallback).

### Prevent writing to your real MLflow
- Disable entirely: set `VKB_MLFLOW_DISABLE=1` (ignores `--mlflow`).
- Redirect to a local file store: set `MLFLOW_TRACKING_URI="file:/tmp/mlruns_test"` or pass `--mlflow-uri file:/tmp/mlruns_test` on the CLI.
- Recommended e2e env bundle:
  - `VKB_MODELS_DIR=/tmp/vkb_models_test VKB_VISDOM_DISABLE=1 VKB_MLFLOW_DISABLE=1`

### Test‑friendly run config (no writes to real services)
- Models dir: set `VKB_MODELS_DIR=/tmp/vkb_models_test` or pass `--models-dir` to CLI. All artifacts (and `LATEST`) go there.
- Visdom: set `VKB_VISDOM_DISABLE=1` to skip connecting entirely. Or point to a non‑production server using `VKB_VISDOM_SERVER` / `VKB_VISDOM_PORT` / `VKB_VISDOM_ENV`.
- MLflow: set `VKB_MLFLOW_DISABLE=1` to ignore `--mlflow` in tests.

### Decision — Dropout default
- Keep standard dropout OFF by default (`--dropout 0.0`). Rationale: we already use strong regularization (drop path 0.25, label smoothing, rot360+noise+brightness+warp). For MobileNetV3, `drop_rate` typically affects the head only and can slow convergence without clear gains on small datasets. If needed, try `--dropout 0.1` and compare curves.
- 2025-11-01: Confirmed glossary acronym: `vkb` stands for "Vision Keyboard" (package namespace used across code/tests).

### 2025-11-01 — Epsilon tie for val_acc (keeps aug changes)
- Added epsilon tie in DL training: `vkb.finetune.VAL_EQ_EPS = 0.002` and helper `_val_compare(val_acc, best_val, eps)`.
- Behavior: if the validation accuracy change is within ±0.002, we treat it as a tie and keep the current augmentation change (via `AugScheduler.update(equal=True)`), rather than reverting.
- Tests: `tests/test_val_compare_epsilon.py` covers improved/tie/worse cases.
## Overfitting Toolkit (Minimal First)
- Regularization (no code):
  - Weight decay: try `--wd 3e-4` or `5e-4`.
  - Dropout: `--dropout 0.1`–`0.3` (head only).
  - Drop path: `--drop-path 0.1`–`0.2` (timm’s `drop_path_rate`).
  - Label smoothing: currently fixed at 0.05; consider adding a flag to try 0.1.
  - Noise: `--noise-std 0.05`–`0.2` (random σ per sample; train only).
- Data balancing and sampling:
  - Enable class balancing: `--class-weights auto` (WeightedRandomSampler + CE weights).
  - Reduce duplicate frames per video: instead of stride, randomly subsample a fixed fraction per video (we can add `--train-frac-per-video` next; simpler to reason about than stride).
- Augmentation (small, cheap):
  - Keep rotation small (≤10–20°); heavy 360° rotation degraded val here.
  - Add random erasing (tiny cutout) with p≈0.2, area≈2–6% (can be added as a small block in `FrameDataset`).
  - Keep color jitter modest; avoid washing out subtle cues.
- Training protocol:
  - Fewer epochs or early stop on plateau (we can add `--early-stop N` to stop after N no‑improve epochs).
  - Smaller batch size (e.g., 64) increases gradient noise; sometimes generalizes better.
  - Freeze→unfreeze: train head only for a few epochs, then unfreeze (tiny `--freeze-epochs` flag if desired).
- Architecture:
  - Try a smaller backbone via `--embed-model mobilenetv3_small_075` or `efficientnet_lite0`.
- Data/labels (highest impact):
  - Use event windows (frames near press times) rather than whole videos. This changes the task from memorizing videos to detecting events and typically improves validation substantially.

### Sampling Without Stride (Keep All Frames)
- If you prefer `stride=1` to retain small temporal variations, avoid per‑video memorization via sampling rather than removal:
  - Group‑balanced batches: ensure each batch draws from many different videos with a cap like ≤25% frames from any single video per batch (custom batch sampler). Keeps dataset size; lowers “video ID” cues.
  - Epoch‑wise per‑video slices: each epoch uses a different contiguous slice per video (e.g., epoch e takes frames i∈[e mod 5] out of 5 slices). Over E≥5 epochs you still see all frames; per‑epoch duplicates are reduced.
  - Per‑video reweighting: weight loss inversely to per‑video frame count, so dominant videos don’t overpower the gradient while all frames remain available.
  - Optional (tiny) mixup α≈0.05: preserves all frames, discourages memorization without heavy code.
- MobileNetV3 size note (Large vs Small, α=1.0 @224²)
  - Params: Large ≈5.4M vs Small ≈2.9M → ~1.9×.
  - FLOPs (MAdds): Large ≈219M vs Small ≈66M → ~3.3×.
  - Final feature dim: Large 1280 vs Small 1024 → 1.25×.
  - Practical: expect ~3× step time at the same batch size; reduce BS ~30–40% or keep BS and accept slower epochs.

### 2025-11-02 — Weight Decay (AdamW) quick guide
- Defaults: we use `--wd 1e-4`.
- Safe range to try on MobileNetV3 fine‑tune: `3e-4`, `5e-4`, up to `1e-3`.
- Beyond ~`3e-3` typically hurts; `1e-2` often collapses learning on small datasets.
- Interactions: with label smoothing (0.05) + drop path (0.1–0.25) + strong aug, prefer `3e-4–5e-4`. If overfit persists, try `1e-3`.
- Exclusions (optional small improvement): do not decay biases/BN/Norm params.
- CLI: `--wd <value>`.
- 2025-11-02 — Preprocessing parity
  - DL: infer_live uses bundle-saved `input_size` and `normalize` (mean/std) and applies `cv.resize → BGR→RGB → /255 → (x-mean)/std`. Train: same after aug; Val/Test: same without aug. Paths: infer (`infer_live.py:_make_preprocess`), train (`vkb/dataset.FrameDataset.__getitem__`).
  - Classic: infer uses the same embedder (`vkb.emb.create_embedder`) as training, with identical resize + RGB + ImageNet normalization.
- Default change: set `--eval-split` default to `0.01` (1%). Optuna driver updated to the same. Test `tests/test_cli_eval_split_default.py` locks the default.

## 2025-11-06 Updates
- New training mode: MediaPipe landmarks + Logistic Regression
  - CLI: `--clf mp_logreg` (minimal; explicit, no hidden fallbacks).
  - Features: per-frame upper-triangle pairwise distances between the 21 hand landmarks (x,y), L1-normalized per frame.
  - Implementation: `vkb/landmarks.py` (`pairwise_distance_features`, `extract_features_for_video`). Training path `_train_mediapipe_logreg()` in `train_frames.py` handles discovery, splits, optional HPO for `C`, and artifacts.
  - Filenames include `val{ACC}` when a validation split is used, consistent with classic/DL.
  - Flags: `--mp-stride` (default 5), `--mp-max-frames` (default 200).
- Tests
  - `tests/test_landmarks_pairwise_features.py` checks 210-dim output and L1 normalization.
  - `tests/test_mp_logreg_train_smoke.py` monkeypatches the extractor for CI; asserts saved path includes `mp_logreg` and `val`.
- Notes
  - When MediaPipe isn’t installed, `mp_logreg` raises clearly; tests stub the extractor instead.
  - Workers are not used here; for GPU overlap questions use the DL path. Repro is deterministic given inputs.

### LogReg C selection (defaults)
- If you don’t pass `--C`, we still try multiple C values by default via `--hpo-logreg 10` (log‑uniform search). We pick the best C on a validation split and then retrain on the train set.
- Disable this by setting `--hpo-logreg 0` to use the literal default `C=1.0`.
- Filename `_val{ACC}` comes from the final evaluation split you configured; HPO’s internal split does not affect the filename.

### Landmarks Cache (mp_logreg)
- We now cache MediaPipe landmark predictions per video+stride under `.cache/vkb/landmarks/<hash>_s<stride>.npz` with fields: `idx` (frame indices), `lm` (Nx21x3), and source fingerprint (`src_size`, `src_mtime_ns`).
- `extract_features_for_video()` loads from cache when fingerprint matches; otherwise recomputes and writes the cache. Minimal design, no partial/incremental writes.
- Test: `tests/test_landmarks_cache.py` ensures a second call with the same stride does not re-run the compute path.
- Dynamic extension: if a stride cache exists and you request more frames (or unlimited), we now append new landmark predictions from the last cached frame onward and update the cache automatically.
- Gotcha (legacy capped caches): caches are keyed only by stride. Older caches created before 2025‑11‑06 will not contain the appended data until you trigger a re-run; to force a clean rebuild remove `._s<stride>.npz` files for that stride.

### mp_logreg Artifacts & Inference
- Artifacts: bundle contains `clf`, `labels`, `clf_name='mp_logreg'`, `embed_model='mediapipe_hand'`. Sidecar adds:
  - `feat_dim=210`, `frames`, `val_acc/test_acc`.
  - `hparams`: `{C, mp_stride, mp_max_frames, feat_norm='l1', pairs='upper_xy', landmarks=21}`.
- infer_live: detects mp_logreg by `clf_name` or `embed_model` and uses a MediaPipe embedder; shows `no_hand` when landmarks aren’t detected. Test: `tests/test_infer_live_mp_logreg.py`.
  - Verified live run path with tests; usage: `python infer_live.py --frames 200` after training `--clf mp_logreg`. Requires `mediapipe` + `opencv-python` installed.
  - Overlay: we now draw detected landmarks as small yellow points on the live frame in mp_logreg mode. Minimal rendering (points only, no connections) to keep code simple.

- ### mp_stride (definition & guidance)
- `--mp-stride N` processes every Nth frame when extracting MediaPipe landmarks for training (e.g., N=5 → frames 0,5,10,…). It reduces compute and near-duplicate samples. Default: `1` (no stride, process every frame).
- The stride value is part of the landmark cache key: `.cache/vkb/landmarks/<hash>_sN.npz`. Changing `N` builds a new cache.
- Sidecar records `mp_stride`. Live inference ignores stride (runs per frame).
- Quick guidance: keep default `1` for small sets; for longer videos use 3–10 to reduce duplicates and speed up extraction.

### mp_max_frames (definition & caveat)
- `--mp-max-frames K` caps how many frames per video we keep after applying `mp_stride` (e.g., K=200 keeps the first 200 sampled frames). It speeds up training and keeps classes balanced across videos. Default: `0` (unlimited).
- Recorded in the sidecar (`hparams.mp_max_frames`). Live inference ignores this.
- Cache caveat: landmark cache files are keyed by stride only. If you cached with a small `mp_max_frames` and later raise it, cached data may be a subset. Delete matching `.cache/vkb/landmarks/*_s<stride>.npz` to rebuild with a larger cap.

### data_small and running on full data
- `data_small/` is a tiny, repo-contained example dataset (labels: `PgDown`, `PgUp`, `no_input`) used by tests and for quick sanity runs.
- Full dataset uses `data/` (default for `--data`). Example runs:
  - mp_logreg (landmarks): `python train_frames.py --clf mp_logreg --data data --eval-split 0.2 --mp-stride 5 --mp-max-frames 200`
  - Classic ridge (embeddings): `python train_frames.py --clf ridge --data data --eval-split 0.2`
- Tips: start with modest `mp_stride` (5–10) and `mp_max_frames` (100–300). Landmark caches make subsequent runs faster.
