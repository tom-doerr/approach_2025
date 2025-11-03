def create_embedder(model_name: str = "mobilenetv3_small_100"):
    import cv2 as cv  # local import to avoid test deps
    import torch, timm, inspect
    kw = dict(pretrained=True, num_classes=0)
    try:
        sig = inspect.signature(timm.create_model)
        if 'global_pool' in sig.parameters:
            kw['global_pool'] = 'avg'
    except Exception:
        pass
    m = timm.create_model(model_name, **kw).eval()
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def embed(frame):
        fr = cv.cvtColor(cv.resize(frame, (224, 224)), cv.COLOR_BGR2RGB)
        t = torch.from_numpy(fr).float().permute(2, 0, 1) / 255.0
        with torch.no_grad():
            return m(((t - mean) / std).unsqueeze(0)).squeeze(0).cpu().numpy()

    return embed


# Minimal OO wrapper
from dataclasses import dataclass


@dataclass
class Embedder:
    model_name: str = "mobilenetv3_small_100"

    def __post_init__(self):
        self._fn = create_embedder(self.model_name)

    def embed(self, frame):
        return self._fn(frame)

    def __call__(self, frame):  # convenience
        return self.embed(frame)
