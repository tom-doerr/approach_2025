#!/usr/bin/env python3
import argparse, os
from vkb.io import make_output_path, update_latest_symlink


def parse_args():
    p = argparse.ArgumentParser(description="Record webcam video with label.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--label", help="label used as subdirectory under data/")
    g.add_argument("--buttons", help="comma-separated labels; shows on-screen buttons; clicking starts recording into that label")
    p.add_argument("--add-button-right-index-inward", action="store_true", help="when using --buttons, also add a button labeled 'right_index_inward'")
    p.add_argument("--no-preview", action="store_true", help="disable GUI preview (use Ctrl-C to stop)")
    p.add_argument("--cam-index", type=int, default=0, help="camera index (default 0)")
    p.add_argument("--list-modes", action="store_true", help="list all formats/resolutions/fps via v4l2-ctl and exit (Linux only)")
    return p.parse_args()


def record(label: str, preview: bool = True, cam_index: int = 0):
    import cv2 as cv  # local import to keep tests simple
    from rich.console import Console
    path = make_output_path(label, ".mp4")
    cons = Console()
    # Best-effort: list full capabilities if available (Linux + v4l2-ctl)
    try:
        import sys as _sys
        if str(getattr(_sys, 'platform', '')) == 'linux':
            try:
                txt = _list_modes_linux(cam_index)
                if txt:
                    try:
                        cons.print("[dim]Capabilities (v4l2-ctl):[/]")
                    except Exception:
                        pass
                    print(txt)
            except Exception:
                pass
    except Exception:
        pass
    cap = cv.VideoCapture(int(cam_index))
    if not cap.isOpened():
        raise RuntimeError("camera not available")
    # Print basic device info
    try:
        bname = cap.getBackendName() if hasattr(cap, 'getBackendName') else "?"
        cons.print(f"[dim]Camera 0 backend={bname}[/]")
    except Exception:
        pass
    # Default request: first MJPG mode (if available); else fall back to 1920x1080@30.
    try:
        choice = None
        try:
            choice = _pick_first_mjpg_mode(cam_index)
        except Exception:
            choice = None
        if hasattr(cv, 'CAP_PROP_FOURCC'):
            cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
        if choice is not None:
            ww, hh, ff = choice
            cap.set(getattr(cv,'CAP_PROP_FRAME_WIDTH',3), ww)
            cap.set(getattr(cv,'CAP_PROP_FRAME_HEIGHT',4), hh)
            cap.set(getattr(cv,'CAP_PROP_FPS',5), ff)
        else:
            cap.set(getattr(cv,'CAP_PROP_FRAME_WIDTH',3), 1920)
            cap.set(getattr(cv,'CAP_PROP_FRAME_HEIGHT',4), 1080)
            cap.set(getattr(cv,'CAP_PROP_FPS',5), 30)
    except Exception:
        pass
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("no frame from camera")
    # Print current mode
    try:
        cw = int(cap.get(getattr(cv, 'CAP_PROP_FRAME_WIDTH', 3)) or frame.shape[1])
        ch = int(cap.get(getattr(cv, 'CAP_PROP_FRAME_HEIGHT', 4)) or frame.shape[0])
        cfps = float(cap.get(getattr(cv, 'CAP_PROP_FPS', 5)) or 0.0)
        cons.print(f"[dim]Current: {cw}x{ch}@{int(round(cfps)) if cfps>0 else '?'}[/]")
    except Exception:
        pass
    # Probe a few common modes and show what appears to work
    try:
        modes = []
        res = [(3840,2160),(2560,1440),(1920,1080),(1280,720),(640,480)]
        fpss = [60, 30]
        ow = int(cap.get(getattr(cv,'CAP_PROP_FRAME_WIDTH',3)) or 0)
        oh = int(cap.get(getattr(cv,'CAP_PROP_FRAME_HEIGHT',4)) or 0)
        ofps = float(cap.get(getattr(cv,'CAP_PROP_FPS',5)) or 0.0)
        for ww, hh in res:
            for ff in fpss:
                try:
                    if hasattr(cap,'set'):
                        cap.set(getattr(cv,'CAP_PROP_FRAME_WIDTH',3), ww)
                        cap.set(getattr(cv,'CAP_PROP_FRAME_HEIGHT',4), hh)
                        cap.set(getattr(cv,'CAP_PROP_FPS',5), ff)
                    # Read once to apply
                    _ = cap.read()
                    rw = int(cap.get(getattr(cv,'CAP_PROP_FRAME_WIDTH',3)) or 0)
                    rh = int(cap.get(getattr(cv,'CAP_PROP_FRAME_HEIGHT',4)) or 0)
                    rfps = float(cap.get(getattr(cv,'CAP_PROP_FPS',5)) or 0.0)
                    if rw and rh and abs(rw-ww) <= 8 and abs(rh-hh) <= 8:
                        modes.append((rw, rh, int(round(rfps)) if rfps>0 else None))
                except Exception:
                    pass
        # Restore original
        try:
            if ow>0: cap.set(getattr(cv,'CAP_PROP_FRAME_WIDTH',3), ow)
            if oh>0: cap.set(getattr(cv,'CAP_PROP_FRAME_HEIGHT',4), oh)
            if ofps>0: cap.set(getattr(cv,'CAP_PROP_FPS',5), ofps)
        except Exception:
            pass
        if modes:
            uniq = {}
            for mw, mh, mf in modes:
                uniq[(mw, mh, mf)] = True
            modes = sorted(uniq.keys(), key=lambda t: (t[0]*t[1], t[2] or 0), reverse=True)
            cons.print("[dim]Modes:[/]")
            for mw, mh, mf in modes:
                cons.print(f"[dim] - {mw}x{mh}@{mf if mf is not None else '?'}[/]")
    except Exception:
        pass
    h, w = frame.shape[:2]
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    # Choose writer FPS from capture if available; otherwise 30.
    try:
        writer_fps = float(cfps) if (cfps and cfps > 0) else 30.0
    except Exception:
        writer_fps = 30.0
    try:
        cons.print(f"[dim]Using: {w}x{h}@{int(round(writer_fps))}[/]")
    except Exception:
        pass
    out = cv.VideoWriter(path, fourcc, writer_fps, (w, h))
    win = f"rec: {label} (press q or ESC)"
    if preview:
        cons.print(f"[bold cyan]Recording[/] to [magenta]{path}[/]. Press [bold]q[/]/[bold]ESC[/] to stop.")
    else:
        cons.print(f"[bold cyan]Recording[/] to [magenta]{path}[/]. Press [bold]Ctrl-C[/] to stop (no preview).")
    from time import perf_counter
    t0 = perf_counter(); fc = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out.write(frame); fc += 1
            if preview:
                # On-frame indicator of what is being recorded
                try:
                    dt = int(max(0.0, perf_counter() - t0))
                    mm, ss = dt // 60, dt % 60
                    cv.putText(frame, f"REC: {label} {mm:02d}:{ss:02d} {fc}f", (10, 24), getattr(cv,'FONT_HERSHEY_SIMPLEX',0), 0.8, (0,0,255), 2)
                except Exception:
                    pass
                cv.imshow(win, frame)
                k = cv.waitKey(1) & 0xFF
                if k in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    cap.release(); out.release();
    link = update_latest_symlink(path)
    if preview:
        cv.destroyAllWindows()
    cons.print(f"[bold green]Saved[/]: [magenta]{path}[/]")
    cons.print(f"[bold]Latest link[/]: [cyan]{link}[/] → [magenta]{os.path.basename(path)}[/]")


def _layout_buttons(labels, w, h):
    # Single row at bottom; equal widths
    n = max(1, len(labels))
    bw = max(1, w // n); bh = max(24, min(80, h // 8)); y0 = h - bh
    boxes = []
    for i, lab in enumerate(labels):
        x0 = i * bw; x1 = (i + 1) * bw - 1
        boxes.append((lab, x0, y0, x1, h - 1))
    return boxes


def record_buttons(labels_csv: str, cam_index: int = 0):
    import cv2 as cv
    from rich.console import Console
    labels = [s.strip() for s in labels_csv.split(',') if s.strip()]
    if not labels:
        raise ValueError("--buttons provided but empty")
    cons = Console()
    cap = cv.VideoCapture(int(cam_index))
    if not cap.isOpened():
        raise RuntimeError("camera not available")
    ok, frame = cap.read()
    if not ok:
        cap.release(); raise RuntimeError("no frame from camera")
    h, w = frame.shape[:2]
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    fps = float(cap.get(getattr(cv, 'CAP_PROP_FPS', 5)) or 30.0)
    win = "rec: buttons (click to start; q/ESC to stop)"
    path = None; out = None; active = None
    boxes = _layout_buttons(labels + ["STOP"], w, h)

    from time import perf_counter
    t0 = None
    fc = 0

    def on_mouse(evt, x, y, flags, param):
        nonlocal out, path, active, t0
        if evt != getattr(cv, 'EVENT_LBUTTONDOWN', 1):
            return
        for lab, x0, y0, x1, y1 in boxes:
            if x0 <= x <= x1 and y0 <= y <= y1:
                if lab == "STOP":
                    if out is not None:
                        # finish current recording now
                        try: out.release()
                        except Exception: pass
                        link = update_latest_symlink(path)
                        try:
                            cons.print(f"[bold green]Saved[/]: [magenta]{path}[/]")
                            cons.print(f"[bold]Latest link[/]: [cyan]{link}[/] → [magenta]{os.path.basename(path)}[/]")
                        except Exception:
                            pass
                        out = None; path = None; active = None; t0 = None; fc = 0
                    return
                # Toggle behavior: clicking the active label stops recording
                if out is not None and lab == active:
                    try: out.release()
                    except Exception: pass
                    link = update_latest_symlink(path)
                    try:
                        cons.print(f"[bold green]Saved[/]: [magenta]{path}[/]")
                        cons.print(f"[bold]Latest link[/]: [cyan]{link}[/] → [magenta]{os.path.basename(path)}[/]")
                    except Exception:
                        pass
                    out = None; path = None; active = None; t0 = None; fc = 0
                    return
                if out is None:
                    p = make_output_path(lab, ".mp4")
                    out = cv.VideoWriter(p, fourcc, fps, (w, h))
                    path = p; active = lab; t0 = perf_counter(); fc = 0
                    try: cons.print(f"[bold cyan]Recording[/] label [magenta]{lab}[/] -> {p}")
                    except Exception: pass
                break

    cv.namedWindow(win)
    try:
        cv.setMouseCallback(win, on_mouse)
    except Exception:
        pass
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # overlay buttons
            try:
                for lab, x0, y0, x1, y1 in boxes:
                    col = (0,0,255) if lab==active else ((0,0,200) if lab=="STOP" else (0,255,0))
                    cv.rectangle(frame, (x0, y0), (x1, y1), col, 2)
                    cv.putText(frame, lab, (x0+6, max(y0+20, y0+6)), getattr(cv,'FONT_HERSHEY_SIMPLEX',0), 0.6, (255,255,255), 1)
                if active and t0 is not None:
                    dt = int(max(0.0, perf_counter() - t0))
                    mm, ss = dt // 60, dt % 60
                    cv.putText(frame, f"REC: {active} {mm:02d}:{ss:02d} {fc}f", (10, 24), getattr(cv,'FONT_HERSHEY_SIMPLEX',0), 0.8, (0,0,255), 2)
            except Exception:
                pass
            if out is not None:
                out.write(frame)
                fc += 1
            cv.imshow(win, frame)
            k = cv.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break
    except KeyboardInterrupt:
        pass
    cap.release()
    if out is not None:
        out.release()
        link = update_latest_symlink(path)
        try:
            cons.print(f"[bold green]Saved[/]: [magenta]{path}[/]")
            cons.print(f"[bold]Latest link[/]: [cyan]{link}[/] → [magenta]{os.path.basename(path)}[/]")
        except Exception:
            pass
    try:
        cv.destroyAllWindows()
    except Exception:
        pass


def _list_modes_linux(cam_index: int):
    import shutil, subprocess, sys
    if sys.platform != 'linux':
        raise RuntimeError("--list-modes requires Linux (v4l2)")
    if shutil.which('v4l2-ctl') is None:
        raise RuntimeError("v4l2-ctl not found. Install v4l-utils.")
    dev = f"/dev/video{int(cam_index)}"
    out = subprocess.run(['v4l2-ctl','-d',dev,'--list-formats-ext'], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or "v4l2-ctl failed")
    return out.stdout


def _pick_first_mjpg_mode(cam_index: int):
    try:
        txt = _list_modes_linux(cam_index)
    except Exception:
        return None
    w = h = fps = None
    in_mjpg = False
    for raw in txt.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "'MJPG'" in line:
            in_mjpg = True
            w = h = fps = None
            continue
        if not in_mjpg:
            continue
        if line.startswith("Size:") and "Discrete" in line and "x" in line and w is None:
            try:
                part = line.split("Discrete",1)[1].strip().split()[0]
                w, h = [int(x) for x in part.split('x')[:2]]
            except Exception:
                w = h = None
        elif line.startswith("Interval:") and "fps" in line and fps is None:
            try:
                seg = line.split('(')[-1]
                val = seg.split()[0]
                fps = float(val)
            except Exception:
                fps = None
        if w and h and fps:
            return (int(w), int(h), int(round(float(fps))))
    return None


def main():
    args = parse_args()
    if args.list_modes:
        txt = _list_modes_linux(args.cam_index)
        print(txt)
        return
    if args.buttons:
        if args.no_preview:
            raise SystemError("--buttons requires preview (GUI)")
        labels_csv = args.buttons
        if getattr(args, 'add_button_right_index_inward', False):
            parts = [s.strip() for s in labels_csv.split(',') if s.strip()]
            if 'right_index_inward' not in parts:
                parts.append('right_index_inward')
            labels_csv = ','.join(parts)
        record_buttons(labels_csv, cam_index=args.cam_index)
    else:
        record(args.label, preview=not args.no_preview, cam_index=args.cam_index)


if __name__ == "__main__":
    main()
