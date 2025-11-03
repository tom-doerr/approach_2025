import os
from dataclasses import dataclass
from datetime import datetime


def safe_label(s: str) -> str:
    return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in s.strip())


def make_output_path(label: str, ext: str = ".mp4") -> str:
    lab = safe_label(label)
    out_dir = os.path.join("data", lab)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"{ts}{ext}")


def list_videos(root: str):
    import glob
    vids = []
    for lab in sorted(os.listdir(root)):
        d = os.path.join(root, lab)
        if not os.path.isdir(d):
            continue
        for p in glob.glob(os.path.join(d, "*.mp4")):
            vids.append((p, lab))
    return vids


def update_latest_symlink(video_path: str, link_name: str = "latest") -> str:
    d = os.path.dirname(video_path)
    link = os.path.join(d, link_name)
    target = os.path.basename(video_path)
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(target, link)
    return link


@dataclass
class IO:
    data_root: str = "data"
    timefmt: str = "%Y%m%d_%H%M%S"

    def safe_label(self, s: str) -> str:
        return safe_label(s)

    def make_output_path(self, label: str, ext: str = ".mp4") -> str:
        lab = safe_label(label)
        out_dir = os.path.join(self.data_root, lab)
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime(self.timefmt)
        return os.path.join(out_dir, f"{ts}{ext}")

    def list_videos(self, root: str | None = None):
        return list_videos(root or self.data_root)

    def update_latest_symlink(self, video_path: str, link_name: str = "latest") -> str:
        return update_latest_symlink(video_path, link_name)
